"""Read the built pool. Use these loaders rather than plain json.

The pool is split in two files:

- data/pool/pool.jsonl: one record per function (code, spec, bug metadata,
  verification summary). Small; read by the harness for every patch.
- data/pool/pool_tests.jsonl.gz: per-function test data (inputs, expected
  outputs, per-input timeouts), needed only for fix testing. Values are
  codec-encoded and some integers exceed Python's default 4300-digit
  str->int limit, which `collusion.pool.codec` lifts on import.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from . import codec

ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = ROOT / "data" / "pool"
POOL_FILE = POOL_DIR / "pool.jsonl"
TESTS_FILE = POOL_DIR / "pool_tests.jsonl.gz"
TEST_FIELDS = ("inputs", "expected", "timeouts")


def load_pool(path: Path = POOL_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tests(path: Path = TESTS_FILE, ids: set[str] | None = None) -> dict[str, dict]:
    """{id: {"inputs": [...], "expected": [...], "timeouts": [...]}} with values decoded."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if ids is not None and rec["id"] not in ids:
                continue
            out[rec["id"]] = {
                "inputs": codec.decode(rec["inputs"]),
                "expected": codec.decode(rec["expected"]),
                "timeouts": rec["timeouts"],
            }
    return out


def write_pool(records: list[dict], pool_path: Path = POOL_FILE, tests_path: Path = TESTS_FILE) -> None:
    """Write full records (with codec-encoded test fields) as the two-file pool."""
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pool_path, "w", encoding="utf-8") as fp, gzip.open(tests_path, "wt", encoding="utf-8") as ft:
        for r in records:
            fp.write(json.dumps({k: v for k, v in r.items() if k not in TEST_FIELDS},
                                sort_keys=True, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": r["id"], **{k: r[k] for k in TEST_FIELDS}}, sort_keys=True) + "\n")
