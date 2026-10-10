"""Final stage: render, insert docstrings, strictly re-verify, write outputs.

Ground truth in pool.jsonl is established on exactly the strings agents will
see: the one-edit check and the per-call sandboxed verification are re-run on
the final `clean_code` / `buggy_code` (docstring included).
"""

from __future__ import annotations

import ast
import difflib
import json
import platform
import random
import sys
from collections import Counter
from importlib import metadata

from . import codec, sources
from .astdiff import diff_pair
from .build import POOL, REFERENCE_TIMEOUT, SEED, funnel, load, parallel_map, save
from .io import write_pool
from .normalize import insert_docstring
from .sandbox import judge, run_calls
from .spec import MODEL, _check_examples

MAX_TIMEOUTS_FINAL = 10
FORBIDDEN_STRINGS = ("HumanEval", "Mbpp", "MBPP", "EvalPlus", "evalplus")

# Records removed after the manual spec review (docs/step1_decisions.md §13).
MANUAL_DROPS = {
    "HumanEval/163": "spec repeats 'inclusive' 5 times; the bug drops the upper endpoint",
}

# USD per million tokens for claude-opus-5-5 (input, output).
PRICE = (4.00, 20.00)


def _leaks(code: str) -> list[str]:
    out = [s for s in FORBIDDEN_STRINGS if s in code]
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            if n.returns or any(x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs):
                out.append(f"annotation in {n.name}")
        if isinstance(n, ast.AnnAssign):
            out.append("annotated assignment")
        if isinstance(n, ast.ImportFrom) and n.module == "typing":
            out.append("typing import")
    return out


def finalize_item(item: dict) -> dict:
    ep = item["entry_point"]

    def fail(reason, detail=""):
        return {"drop": {"id": item["id"], "source": item["source"], "stage": "finalize",
                         "reason": reason, "detail": str(detail)[:300]}}

    if item["id"] in MANUAL_DROPS:
        return fail("manual_review_hints_at_bug", MANUAL_DROPS[item["id"]])

    clean_code = insert_docstring(item["clean"], ep, item["spec"])
    buggy_code = insert_docstring(item["buggy"], ep, item["spec"])

    if ast.get_docstring(_fn(clean_code, ep)) is None:
        return fail("docstring_not_inserted")
    leaks = _leaks(clean_code) + _leaks(buggy_code)
    if leaks:
        return fail("source_leak", leaks)

    d = diff_pair(clean_code, buggy_code)
    if not d.ok:
        return fail("not_single_edit_after_render", d.reason)

    # Clean is trusted: generous limit (correctness check, avoids load-induced timeouts).
    res = run_calls(clean_code, ep, item["inputs"], REFERENCE_TIMEOUT, mode="call", max_timeouts=1)
    v_clean = judge(res, item["ref_outputs"], item["atol"], ep, item["source"])
    if not v_clean.all_pass:
        return fail("final_clean_fails", v_clean.first_failure)
    res = run_calls(buggy_code, ep, item["inputs"], item["timeouts"], mode="call", max_timeouts=MAX_TIMEOUTS_FINAL)
    v_buggy = judge(res, item["ref_outputs"], item["atol"], ep, item["source"])
    if v_buggy.n_fail == 0:
        return fail("final_buggy_passes_all_inputs")

    final = {**item, "clean": clean_code, "buggy": buggy_code}
    examples, _ = _check_examples(final, item["examples"])
    if len(examples) != len(item["examples"]):
        return fail("final_examples_fail")

    record = {
        "id": item["id"],
        "source": item["source"],
        "entry_point": ep,
        "spec": item["spec"],
        "clean_code": clean_code,
        "buggy_code": buggy_code,
        "bug_origin": item["bug_origin"],
        "bug_type": item["bug_type"],
        "bug_location": {
            "clean_lines": list(d.clean_lines) if d.clean_lines else None,
            "buggy_lines": list(d.buggy_lines) if d.buggy_lines else None,
            "n_edit_regions": d.n_regions,
        },
        "inputs": codec.encode(item["inputs"]),
        "n_base_inputs": item["n_base"],
        "expected": codec.encode(item["ref_outputs"]),
        "timeouts": [round(t, 4) for t in item["timeouts"]],
        "atol": item["atol"],
        "examples": item["examples"],
        "verification": {
            "clean": v_clean.as_dict(),
            "buggy": v_buggy.as_dict(),
            "mode": "process-per-call",
        },
        "he_clean_base": item.get("he_clean_base"),
        "mutant_detail": item.get("mutant_detail"),
        "n_valid_mutants": item.get("n_valid_mutants"),
        "spec_drafts": item["spec_drafts"],
    }
    # Round-trip check: what Step 2 will decode must equal what was verified.
    assert codec.same(codec.decode(json.loads(json.dumps(record["inputs"]))), item["inputs"])
    return {"record": record}


def _fn(code: str, name: str) -> ast.FunctionDef:
    for n in ast.parse(code).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def _llm_cost(log: list[dict]) -> dict:
    fresh = [m for m in log if not m.get("cached")]
    tin = sum(m["input_tokens"] for m in log)
    tout = sum(m["output_tokens"] for m in log)
    return {
        "calls": len(log),
        "calls_uncached_this_run": len(fresh),
        "input_tokens": tin,
        "output_tokens": tout,
        "est_cost_usd": round(tin / 1e6 * PRICE[0] + tout / 1e6 * PRICE[1], 2),
        "by_tag": dict(Counter(m["tag"].rstrip("0123456789") for m in log)),
        "served_by": dict(Counter(m["model"] for m in log)),
    }


def _review_sample(records: list[dict], n: int = 30) -> str:
    rng = random.Random(SEED)
    sample = rng.sample(records, min(n, len(records)))
    out = [
        "# Step 1 spec review sample",
        "",
        f"{len(sample)} random pool records (seed {SEED}). For each, check that the spec is",
        "complete and neutral: it should let you spot the bug, but it should NOT single out",
        "or hint at the changed behaviour. Mark each as OK / HINTS / INCOMPLETE.",
        "",
    ]
    for k, r in enumerate(sample, 1):
        diff = "\n".join(difflib.unified_diff(
            r["clean_code"].splitlines(), r["buggy_code"].splitlines(), "clean", "buggy", lineterm="", n=1))
        out += [
            f"## {k}. {r['id']} ({r['bug_type']}, {r['bug_origin']})  verdict: ____",
            "",
            "```python",
            r["clean_code"],
            "```",
            "",
            "```diff",
            diff,
            "```",
            "",
        ]
    return "\n".join(out)


def stage_finalize(workers: int) -> dict:
    prev = load("spec")
    outs = parallel_map(finalize_item, prev["items"], workers, "finalize")
    records = sorted((o["record"] for o in outs if "record" in o), key=lambda r: (r["source"], r["id"]))
    drops = prev["drops"] + [o["drop"] for o in outs if "drop" in o]
    kept_ids = {r["id"] for r in records}

    POOL.mkdir(parents=True, exist_ok=True)
    write_pool(records, POOL / "pool.jsonl", POOL / "pool_tests.jsonl.gz")
    with open(POOL / "mutants_all.jsonl", "w", encoding="utf-8") as f:
        for m in prev["all_mutants"]:
            if m["id"] in kept_ids:
                f.write(json.dumps(m, sort_keys=True) + "\n")
    with open(POOL / "drops.jsonl", "w", encoding="utf-8") as f:
        for d in drops:
            f.write(json.dumps(d, sort_keys=True) + "\n")
    # Never overwrite a completed manual review; a fresh sample goes alongside it.
    sample = POOL / "review_sample.md"
    if sample.exists() and "verdict: ____" not in sample.read_text():
        sample = POOL / "review_sample.new.md"
    sample.write_text(_review_sample(records))

    out = {"items": [{"source": r["source"], "bug_type": r["bug_type"], "he_clean_base": r["he_clean_base"],
                      "id": r["id"]} for r in records], "drops": drops}
    report = {
        "funnel": funnel(out),
        "bug_origin": dict(Counter(r["bug_origin"] for r in records)),
        "buggy_fail_kinds": dict(sum((Counter(r["verification"]["buggy"]["kinds"]) for r in records), Counter())),
        "buggy_fails_on_every_input": sum(
            r["verification"]["buggy"]["n_fail"] == r["verification"]["buggy"]["n_inputs"] for r in records),
        "spec_drafts": dict(Counter(r["spec_drafts"] for r in records)),
        "llm": {"model": MODEL, **_llm_cost(prev["llm_log"])},
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **{p: metadata.version(p) for p in ("anthropic", "numpy", "pydantic", "huggingface_hub")},
            "datasets": {k: {"file": v["file"], "sha256": v["sha256"]} for k, v in sources.SOURCES.items()},
        },
        "seed": SEED,
    }
    (POOL / "build_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    save("finalize", out)
    return out
