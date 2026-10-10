"""Fetch and load the three pinned data sources.

- HumanEval+ v0.1.10 and MBPP+ v0.2.0: EvalPlus release files.
- HumanEvalPack (Python split): HumanEvalFix bugs, from the Hugging Face dataset
  `bigcode/humanevalpack` at revision 7980d994 (2024-05-01), the last revision
  that ships the original JSONL; later revisions only convert it to Parquet.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from ._evalplus_vendor import mbpp_deserialize_inputs

HEPACK_REV = "7980d99405b56c9745d51040cabbd9e86a056397"

SOURCES = {
    "humaneval_plus": {
        "file": "HumanEvalPlus-v0.1.10.jsonl.gz",
        "url": "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
        "sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
    },
    "mbpp_plus": {
        "file": "MbppPlus-v0.2.0.jsonl.gz",
        "url": "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
        "sha256": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
    },
    "humanevalpack_python": {
        "file": f"humanevalpack-python-{HEPACK_REV}.jsonl",
        "url": f"https://huggingface.co/datasets/bigcode/humanevalpack/resolve/{HEPACK_REV}/data/python/data/humanevalpack.jsonl",
        "sha256": "b1376d72be4bd016a62ad043d9de3846edb8d63edf1222da802a31330c14c3a7",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, meta in SOURCES.items():
        path = raw_dir / meta["file"]
        if not path.exists():
            urllib.request.urlretrieve(meta["url"], path)
        digest = _sha256(path)
        if digest != meta["sha256"]:
            raise RuntimeError(f"{path} sha256 {digest} != pinned {meta['sha256']}")
        paths[name] = path
    return paths


def _jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(raw_dir: Path) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Return (humaneval_plus, mbpp_plus, hefix_by_humaneval_id).

    MBPP+ inputs are deserialized exactly as EvalPlus does (tuples, sets,
    complex numbers for specific tasks).
    """
    paths = fetch(raw_dir)
    he = _jsonl(paths["humaneval_plus"])
    mb = _jsonl(paths["mbpp_plus"])
    for t in he + mb:
        for k in ("base_input", "plus_input"):
            if not isinstance(t[k], list):  # Mbpp/793 stores an empty plus_input as {}
                t[k] = list(t[k])
    for t in mb:
        t["base_input"] = mbpp_deserialize_inputs(t["task_id"], t["base_input"])
        t["plus_input"] = mbpp_deserialize_inputs(t["task_id"], t["plus_input"])
    hefix = {}
    for t in _jsonl(paths["humanevalpack_python"]):
        n = int(t["task_id"].split("/")[1])  # "Python/N" -> "HumanEval/N"
        hefix[f"HumanEval/{n}"] = t
    return he, mb, hefix
