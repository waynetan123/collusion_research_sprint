"""Independent post-build verification of data/pool (Step 1 acceptance checks).

Reads only the files on disk (via collusion.pool.io) and re-checks every record:

1. clean passes every stored input; buggy fails >= 1 (sandboxed, re-run from
   the stored strings and the decoded test sidecar);
2. the docstring examples, parsed from `clean_code` itself, pass on both versions;
3. no source leaks: dataset names, type annotations, typing imports;
4. clean/buggy still differ by one localized edit at the recorded location;
5. `spec` equals the docstring embedded in both versions.

Usage: python -m collusion.pool.check_pool [--workers W]
"""

from __future__ import annotations

import argparse
import ast
import os
from collections import Counter

from .astdiff import diff_pair
from .build import REFERENCE_TIMEOUT, parallel_map
from .finalize import _leaks
from .io import load_pool, load_tests
from .sandbox import OK, judge, run_calls
from ._evalplus_vendor import outputs_match


def _examples(code: str, ep: str) -> list[tuple[str, str]]:
    fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == ep)
    lines = [ln.strip() for ln in (ast.get_docstring(fn) or "").splitlines()]
    return [(lines[k][4:], lines[k + 1]) for k in range(len(lines) - 1) if lines[k].startswith(">>> ")]


def check_record(job: tuple[dict, dict]) -> dict:
    rec, tests = job
    ep, src, atol = rec["entry_point"], rec["source"], rec["atol"]
    problems = []

    res = run_calls(rec["clean_code"], ep, tests["inputs"], REFERENCE_TIMEOUT, mode="batch", max_timeouts=1)
    if not judge(res, tests["expected"], atol, ep, src).all_pass:
        problems.append("clean fails a stored input")
    res = run_calls(rec["buggy_code"], ep, tests["inputs"], tests["timeouts"], mode="batch", max_timeouts=10)
    if judge(res, tests["expected"], atol, ep, src).n_fail == 0:
        problems.append("buggy passes every stored input")

    exs = _examples(rec["clean_code"], ep)
    if len(exs) < 2:
        problems.append(f"only {len(exs)} docstring examples")
    calls = []
    for call, expected in exs:
        node = ast.parse(call, mode="eval").body
        calls.append(([ast.literal_eval(a) for a in node.args], ast.literal_eval(expected)))
    for code in (rec["clean_code"], rec["buggy_code"]):
        out = run_calls(code, ep, [a for a, _ in calls], 5.0)
        for (_, exp), r in zip(calls, out):
            if r.status != OK or not outputs_match(r.value, exp, atol, ep, src):
                problems.append("docstring example fails on " + ("clean" if code is rec["clean_code"] else "buggy"))

    leaks = _leaks(rec["clean_code"]) + _leaks(rec["buggy_code"])
    if rec["id"] in rec["clean_code"] or rec["id"] in rec["buggy_code"]:
        leaks.append("record id")
    if leaks:
        problems.append(f"leaks: {leaks}")

    d = diff_pair(rec["clean_code"], rec["buggy_code"])
    if not d.ok:
        problems.append(f"not a single edit: {d.reason}")
    elif [list(d.clean_lines or []), list(d.buggy_lines or [])] != [
        rec["bug_location"]["clean_lines"] or [], rec["bug_location"]["buggy_lines"] or []
    ]:
        problems.append("bug_location mismatch")

    for code in (rec["clean_code"], rec["buggy_code"]):
        fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == ep)
        if ast.get_docstring(fn) != rec["spec"]:
            problems.append("spec differs from embedded docstring")
            break

    return {"id": rec["id"], "problems": problems, "item": True} if not problems else \
        {"id": rec["id"], "problems": problems, "drop": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    pool = load_pool()
    tests = load_tests()
    assert set(tests) == {r["id"] for r in pool}, "pool and test sidecar ids differ"
    assert len({r["id"] for r in pool}) == len(pool), "duplicate ids"
    outs = parallel_map(check_record, [(r, tests[r["id"]]) for r in pool], args.workers, "check")
    bad = [o for o in outs if o["problems"]]
    print(f"records: {len(pool)}  passed: {len(pool) - len(bad)}  failed: {len(bad)}")
    print("by source:", dict(Counter(r["source"] for r in pool)))
    for o in bad:
        print(" ", o["id"], o["problems"])
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
