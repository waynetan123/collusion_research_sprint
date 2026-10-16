"""Export the pool as code-analysis tasks for the forked harness.

Each pool record becomes two tasks (clean and buggy), each a single Python file
holding the stored code (spec docstring included) under a neutral name, plus one
manifest in upstream's format. Agents see the file path and function name, so
nothing in either may hint at the source dataset or at which version is buggy:

- task ids are ``fn_<NNN>_task`` with NNN a seeded random permutation of
  1..2N, so the visible display id ("1-NNN") carries no order information;
- files are ``data/tasks/fn_<NNN>.py`` (gitignored; regenerate on each machine);
- EvalPlus inputs are never exported. Ground truth is the record's version.

Extra manifest fields (pool_id, source, bug_type, bug_origin) are for analysis
only; the harness never shows them to agents.

    python -m collusion.v2.export_tasks
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from collusion.pool.io import POOL_FILE, load_pool
from collusion.v2 import HARNESS_ROOT, POOL_MANIFEST, POOL_TASK_DIR

# Seeds the task-number permutation. Changing it renames every task.
EXPORT_SEED = 20261016
VERSIONS = ("clean", "buggy")


def task_entries(records: list[dict], seed: int = EXPORT_SEED) -> list[dict]:
    """Manifest entries for every (record, version), numbered by a seeded permutation."""
    numbers = list(range(1, 2 * len(records) + 1))
    random.Random(seed).shuffle(numbers)
    width = max(3, len(str(len(numbers))))
    entries = []
    ordered = sorted(records, key=lambda record: record["id"])
    for index, (record, version) in enumerate(
        (record, version) for record in ordered for version in VERSIONS
    ):
        name = f"fn_{numbers[index]:0{width}d}"
        buggy = version == "buggy"
        entries.append(
            {
                "task_id": f"{name}_task",
                # Twins share spec_key, so the loader never pairs them in one episode.
                "spec_key": record["id"],
                "task_type": "code_analysis",
                "is_buggy": buggy,
                "expected_verdict": "fail" if buggy else "pass",
                "code_path": f"{POOL_TASK_DIR}/{name}.py",
                "function_name": record["entry_point"],
                "pool_id": record["id"],
                "version": version,
                "source": record["source"],
                "bug_type": record["bug_type"],
                "bug_origin": record["bug_origin"],
            }
        )
    entries.sort(key=lambda entry: entry["task_id"])
    return entries


def file_text(record: dict, version: str) -> str:
    """The exported file: the stored code exactly, plus a final newline."""
    return record[f"{version}_code"] + "\n"


def export(
    records: list[dict],
    root: Path = HARNESS_ROOT,
    seed: int = EXPORT_SEED,
) -> list[dict]:
    """Write every task file and the manifest; remove stale task files."""
    by_id = {record["id"]: record for record in records}
    entries = task_entries(records, seed)
    task_dir = root / POOL_TASK_DIR
    task_dir.mkdir(parents=True, exist_ok=True)
    wanted = set()
    for entry in entries:
        path = root / entry["code_path"]
        wanted.add(path.name)
        path.write_text(
            file_text(by_id[entry["pool_id"]], entry["version"]), encoding="utf-8"
        )
    for stale in task_dir.glob("fn_*.py"):
        if stale.name not in wanted:
            stale.unlink()
    (root / POOL_MANIFEST).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, default=POOL_FILE)
    args = parser.parse_args()
    records = load_pool(args.pool)
    entries = export(records)
    buggy = sum(entry["is_buggy"] for entry in entries)
    print(
        f"Exported {len(entries)} tasks ({buggy} buggy) from {len(records)} pool "
        f"records to {HARNESS_ROOT / POOL_TASK_DIR}"
    )


if __name__ == "__main__":
    main()
