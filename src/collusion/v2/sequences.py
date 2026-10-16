"""Deal each seed's episode sequence from the exported pool tasks.

One sequence per trajectory seed, in upstream's ``{"pairs": [{"tasks": [a, b]}]}``
format, written as ``data/sequences/repNNN_sampled_manifest.json`` (gitignored):

- EPISODES episodes x 2 tasks, from 2 x EPISODES distinct pool functions, so no
  function repeats and a trajectory never holds both versions of one;
- every episode pairs one HumanEval and one MBPP function, and which agent gets
  which source is random, so sources are balanced within each trajectory;
- exactly half the tasks are buggy (the split within each source differs by at
  most one);
- deterministic in the seed (and the excluded set). Arms share sequences by
  seed, and every arm 3 branch from a checkpoint continues the same sequence.

``--exclude`` takes a calibration drops.json (Step 3) and never deals its pool
functions. data/sequences/sequences_meta.json records, per seed, the file's
sha256 and the calibration it was dealt under; the launcher checks both.

    python -m collusion.v2.sequences --seeds 1-24 [--exclude data/calibration/calib/drops.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from collusion.v2 import HARNESS_ROOT, POOL_MANIFEST, SEQUENCE_DIR

# 25 episodes: up to 13 before the arm 2 deadline plus 12 post-branch episodes.
EPISODES = 25
SOURCES = ("humaneval", "mbpp")
META_FILENAME = "sequences_meta.json"
# Fields of a calibration's drops.json that identify what it measured.
CALIBRATION_FIELDS = ("run_id", "settings_hash", "model_settings", "tasks_sha256", "samples")


def parse_seeds(text: str) -> list[int]:
    """Parse "1-24", "3", or "1,5,9-12" into a sorted list of distinct seeds."""
    seeds: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = (int(value) for value in part.split("-", 1))
            seeds.update(range(low, high + 1))
        elif part:
            seeds.add(int(part))
    if not seeds or min(seeds) < 1:
        raise ValueError(f"seeds must be positive integers, got {text!r}")
    return sorted(seeds)


def _twins_by_source(entries: list[dict]) -> dict[str, list[dict[str, dict]]]:
    """{source: [{"clean": entry, "buggy": entry}, ...]} sorted by pool id."""
    twins: dict[str, dict[str, dict]] = defaultdict(dict)
    for entry in entries:
        twins[entry["pool_id"]][entry["version"]] = entry
    out: dict[str, list[dict[str, dict]]] = defaultdict(list)
    for pool_id in sorted(twins):
        pair = twins[pool_id]
        if set(pair) != {"clean", "buggy"}:
            raise ValueError(f"{pool_id} lacks a clean/buggy twin in the manifest")
        out[pair["clean"]["source"]].append(pair)
    return out


def deal(
    entries: list[dict],
    seed: int,
    episodes: int = EPISODES,
    exclude: frozenset[str] = frozenset(),
) -> list[list[str]]:
    """One seed's sequence: a list of [alice_task_id, bob_task_id] pairs.

    Pool ids in ``exclude`` are never dealt; with none excluded the result is
    the same as before exclusion existed.
    """
    rng = random.Random(f"collusion-v2-sequence-{seed}")
    twins = _twins_by_source([e for e in entries if e["pool_id"] not in exclude])
    for source in SOURCES:
        if len(twins[source]) < episodes:
            raise ValueError(
                f"need {episodes} {source} functions, the pool has {len(twins[source])}"
            )
    # Half of all tasks are buggy; split that between sources as evenly as possible.
    buggy_total = episodes
    first_share = buggy_total // 2 + rng.randrange(2) * (buggy_total % 2)
    buggy_by_source = {SOURCES[0]: first_share, SOURCES[1]: buggy_total - first_share}
    picked: dict[str, list[dict]] = {}
    for source in SOURCES:
        functions = rng.sample(twins[source], episodes)
        versions = ["buggy"] * buggy_by_source[source] + ["clean"] * (
            episodes - buggy_by_source[source]
        )
        rng.shuffle(versions)
        picked[source] = [pair[version] for pair, version in zip(functions, versions)]
    pairs = []
    for index in range(episodes):
        tasks = [picked[source][index]["task_id"] for source in SOURCES]
        if rng.randrange(2):
            tasks.reverse()
        pairs.append(tasks)
    return pairs


def sequence_path(seed: int, root: Path = HARNESS_ROOT) -> Path:
    return root / SEQUENCE_DIR / f"rep{seed:03d}_sampled_manifest.json"


def meta_path(root: Path = HARNESS_ROOT) -> Path:
    return root / SEQUENCE_DIR / META_FILENAME


def read_sequence_meta(root: Path = HARNESS_ROOT) -> dict[str, dict]:
    """{str(seed): {"sha256", "episodes", "calibration"}} for every dealt seed."""
    path = meta_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["seeds"]


def load_exclusion(path: Path) -> tuple[frozenset[str], dict]:
    """The pool ids a calibration drops, and the calibration's identity."""
    drops = json.loads(Path(path).read_text(encoding="utf-8"))
    if drops.get("partial"):
        raise SystemExit(
            f"{path} comes from a partial calibration run (sizing or unfinished); "
            "only a complete run's drops can be excluded"
        )
    identity = {field: drops[field] for field in CALIBRATION_FIELDS}
    identity["drops_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    identity["drop_pool_ids"] = sorted(drops["drop_pool_ids"])
    return frozenset(drops["drop_pool_ids"]), identity


def write_sequences(
    seeds: list[int],
    root: Path = HARNESS_ROOT,
    episodes: int = EPISODES,
    exclude_path: Path | None = None,
) -> list[Path]:
    entries = json.loads((root / POOL_MANIFEST).read_text(encoding="utf-8"))
    exclude, calibration = (
        load_exclusion(exclude_path) if exclude_path else (frozenset(), None)
    )
    meta = read_sequence_meta(root)
    paths = []
    for seed in seeds:
        path = sequence_path(seed, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        pairs = deal(entries, seed, episodes, exclude)
        document = {"pairs": [{"tasks": pair} for pair in pairs]}
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        meta[str(seed)] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "episodes": episodes,
            "calibration": calibration,
        }
        paths.append(path)
    target = meta_path(root)
    scratch = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    ordered = {key: meta[key] for key in sorted(meta, key=int)}
    scratch.write_text(json.dumps({"seeds": ordered}, indent=2) + "\n", encoding="utf-8")
    os.replace(scratch, target)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="1-24", help='e.g. "1-24" or "1,5,9-12"')
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--exclude", type=Path, default=None,
                        help="A complete calibration's drops.json; its pool ids are not dealt.")
    args = parser.parse_args()
    paths = write_sequences(
        parse_seeds(args.seeds), episodes=args.episodes, exclude_path=args.exclude
    )
    excluded = f", excluding {args.exclude}" if args.exclude else ""
    print(
        f"Wrote {len(paths)} sequences of {args.episodes} episodes to "
        f"{paths[0].parent}{excluded}"
    )


if __name__ == "__main__":
    main()
