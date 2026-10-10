"""Build the patch pool (Step 1).

Stages (each writes data/interim/<stage>.pkl and can be re-run on its own):

  assemble  load sources, normalize, HE base check, exclude special-oracle tasks
  screen    reference outputs, verify normalized clean, one-edit check, screen
            buggy candidates (HE: HumanEvalFix bug; MBPP: every mutant)
  select    pick one MBPP mutant per function (seeded, bug-type balanced)
  spec      Claude drafts + checks the spec (see spec.py)
  finalize  insert docstrings, strict per-call re-verification on the stored
            strings, write data/pool/{pool.jsonl, mutants_all.jsonl,
            build_report.json, review_sample.md}

Usage:
  python -m collusion.pool.build all [--limit N] [--workers W]
  python -m collusion.pool.build screen --limit 5
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import pickle
import random
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from . import sources
from ._evalplus_vendor import SPECIAL_ORACLE_ENTRY_POINTS
from .astdiff import diff_pair
from .mutate import DOC_OPERATORS, mutants
from .normalize import normalize
from .sandbox import OK, judge, run_calls, timeouts_from_runtimes
from .spec import MIN_EXAMPLES, literal_example

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RAW, INTERIM, POOL = DATA / "raw", DATA / "interim", DATA / "pool"
SEED = 20261008
REFERENCE_TIMEOUT = 30.0  # seconds per call for the trusted reference run
SCREEN_MAX_TIMEOUTS = 3  # stop screening a candidate after this many timeouts


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def save(stage: str, obj) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    with open(INTERIM / f"{stage}.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(stage: str):
    with open(INTERIM / f"{stage}.pkl", "rb") as f:
        return pickle.load(f)


def drop(item: dict, stage: str, reason: str, detail: str = "") -> dict:
    return {"id": item["id"], "source": item["source"], "stage": stage, "reason": reason, "detail": detail[:300]}


def entry_docstring(code: str, entry_point: str) -> str:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return ast.get_docstring(node) or ""
    return ast.get_docstring(tree) or ""


def parallel_map(fn, items, workers: int, desc: str):
    """Map `fn` over items in worker processes, with a progress bar showing
    how many items were kept vs dropped so far."""
    # spawn: children start clean (no inherited threads), so the sandbox can fork.
    ctx = mp.get_context("spawn")
    outs, kept, dropped = [], 0, 0
    with ctx.Pool(workers) as pool, tqdm(
        total=len(items), desc=f"  {desc}", unit="fn", position=1, leave=False, dynamic_ncols=True
    ) as bar:
        for out in pool.imap(fn, items, chunksize=1):
            outs.append(out)
            if "drop" in out:
                dropped += 1
            else:
                kept += 1
            bar.set_postfix(kept=kept, dropped=dropped, refresh=False)
            bar.update()
    return outs


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #


def stage_assemble(limit: int | None) -> dict:
    he, mb, hefix = sources.load(RAW)
    if limit:
        he, mb = he[:limit], mb[:limit]
    items, drops = [], []

    for t in he:
        item = {"id": t["task_id"], "source": "humaneval", "entry_point": t["entry_point"]}
        if t["entry_point"] in SPECIAL_ORACLE_ENTRY_POINTS["humaneval"]:
            drops.append(drop(item, "assemble", "special_oracle_task"))
            continue
        fix = hefix[t["task_id"]]
        assert fix["entry_point"] == t["entry_point"], t["task_id"]
        evalplus_clean = normalize(t["prompt"] + t["canonical_solution"])
        original_clean = normalize(fix["declaration"] + fix["canonical_solution"])
        item.update(
            reference_code=t["prompt"] + t["canonical_solution"],
            clean=original_clean,  # HumanEvalFix's bug was injected into this one
            he_base_identical=evalplus_clean == original_clean,
            buggy=normalize(fix["declaration"] + fix["buggy_solution"]),
            bug_origin="humanevalfix",
            bug_type=fix["bug_type"],
            description=entry_docstring(t["prompt"], t["entry_point"]),
            inputs=t["base_input"] + t["plus_input"],
            n_base=len(t["base_input"]),
            atol=t["atol"],
        )
        items.append(item)

    for t in mb:
        item = {"id": t["task_id"], "source": "mbpp", "entry_point": t["entry_point"]}
        if t["entry_point"] in SPECIAL_ORACLE_ENTRY_POINTS["mbpp"]:
            drops.append(drop(item, "assemble", "special_oracle_task"))
            continue
        item.update(
            reference_code=t["prompt"] + t["canonical_solution"],
            clean=normalize(t["prompt"] + t["canonical_solution"]),
            description=ast.get_docstring(ast.parse(t["prompt"])) or "",
            inputs=t["base_input"] + t["plus_input"],
            n_base=len(t["base_input"]),
            atol=t["atol"],
        )
        items.append(item)

    out = {"items": items, "drops": drops}
    save("assemble", out)
    return out


# --------------------------------------------------------------------------- #
# screen
# --------------------------------------------------------------------------- #


def _screen_candidate(item: dict, code: str) -> dict:
    """Behaviour of a buggy candidate, plus how many short test inputs it gets right.

    `n_example_inputs` counts inputs (capped at 10) on which the candidate agrees
    with clean AND that render as a short literal docstring example. The design
    doc requires 2-3 spec examples that the buggy version also passes, so a bug
    that is wrong on (nearly) every input cannot be given a valid spec.
    """
    ep = item["entry_point"]
    res = run_calls(code, ep, item["inputs"], item["timeouts"], mode="batch", max_timeouts=SCREEN_MAX_TIMEOUTS)
    v = judge(res, item["ref_outputs"], item["atol"], ep, item["source"])
    failing = set(v.failing_idx)
    n_ex = 0
    for i, r in enumerate(res):
        if r.status == OK and i not in failing and literal_example(ep, item["inputs"][i], item["ref_outputs"][i]):
            n_ex += 1
            if n_ex >= 10:
                break
    return {**v.as_dict(), "n_example_inputs": n_ex}


def screen_item(item: dict) -> dict:
    """Worker: returns {"item": ...} or {"drop": ...}."""
    ep, src = item["entry_point"], item["source"]

    # 1. Reference outputs from the original EvalPlus solution (trusted).
    ref = run_calls(item["reference_code"], ep, item["inputs"], REFERENCE_TIMEOUT, mode="batch", max_timeouts=1)
    bad = [r for r in ref if r.status != OK]
    if bad:
        return {"drop": drop(item, "screen", "reference_failed", f"{bad[0].status}: {bad[0].error}")}
    evalplus_out = [r.value for r in ref]
    timeouts = timeouts_from_runtimes([r.runtime for r in ref])

    # 2. Normalized clean must reproduce EvalPlus's outputs on every input. Clean
    #    code is trusted, so it gets the generous reference timeout: this checks
    #    correctness, and tight limits cause spurious timeouts under load.
    res = run_calls(item["clean"], ep, item["inputs"], REFERENCE_TIMEOUT, mode="batch", max_timeouts=1)
    v = judge(res, evalplus_out, item["atol"], ep, src)
    if not v.all_pass:
        reason = (
            "original_canonical_disagrees_with_evalplus"
            if src == "humaneval" and not item["he_base_identical"]
            else "normalized_clean_mismatch"
        )
        return {"drop": drop(item, "screen", reason, v.first_failure or "")}
    # Reference outputs are the clean version's own outputs (design doc step 6).
    item["ref_outputs"] = [r.value for r in res]
    item["ref_runtimes"] = [r.runtime for r in res]
    item["timeouts"] = timeouts_from_runtimes(item["ref_runtimes"])
    if src == "humaneval":
        item["he_clean_base"] = "evalplus_identical" if item["he_base_identical"] else "original_recovered"

    # 3. Buggy candidates: one-edit check, then must fail on >= 1 input.
    if src == "humaneval":
        d = diff_pair(item["clean"], item["buggy"])
        if not d.ok:
            return {"drop": drop(item, "screen", "not_single_edit", d.reason)}
        v = _screen_candidate(item, item["buggy"])
        if v["n_fail"] == 0:
            return {"drop": drop(item, "screen", "buggy_passes_all_inputs")}
        if v["n_example_inputs"] < MIN_EXAMPLES:
            return {"drop": drop(item, "screen", "too_few_example_inputs_buggy_passes",
                                 f"{v['n_pass']}/{v['n_inputs']} inputs pass")}
        item["screen"] = v
        return {"item": item}

    cands = mutants(item["clean"])
    valid, n_failing = [], 0
    for family in (DOC_OPERATORS, None):  # doc operators first; extensions only as fallback
        for m in cands:
            if (family is not None) != (m.operator in DOC_OPERATORS):
                continue
            if not diff_pair(item["clean"], m.code).ok:
                continue
            v = _screen_candidate(item, m.code)
            if v["n_fail"] > 0:
                n_failing += 1
                if v["n_example_inputs"] >= MIN_EXAMPLES:
                    valid.append({"code": m.code, "operator": m.operator, "bug_type": m.bug_type,
                                  "detail": m.detail, "screen": v})
        if valid:
            break
    item["n_mutants_generated"] = len(cands)
    if not cands:
        return {"drop": drop(item, "screen", "no_mutation_site")}
    if not valid:
        reason = "too_few_example_inputs_buggy_passes" if n_failing else "no_mutant_fails_any_input"
        return {"drop": drop(item, "screen", reason)}
    item["candidates"] = valid
    return {"item": item}


def stage_screen(workers: int) -> dict:
    prev = load("assemble")
    t0 = time.time()
    outs = parallel_map(screen_item, prev["items"], workers, "screen")
    items = [o["item"] for o in outs if "item" in o]
    drops = prev["drops"] + [o["drop"] for o in outs if "drop" in o]
    out = {"items": items, "drops": drops, "seconds": round(time.time() - t0, 1)}
    save("screen", out)
    return out


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #


def _crashes_everywhere(screen: dict) -> bool:
    """Fails on every input, all by exception: likely obvious on reading."""
    return screen["n_fail"] == screen["n_inputs"] and set(screen["kinds"]) <= {"error"}


def stage_select() -> dict:
    prev = load("screen")
    rng = random.Random(SEED)
    items = prev["items"]
    counts = Counter(i["bug_type"] for i in items if i["source"] == "humaneval")
    mbpp = [i for i in items if i["source"] == "mbpp"]
    order = list(range(len(mbpp)))
    rng.shuffle(order)
    all_mutants = []
    for k in order:
        item = mbpp[k]
        cands = item["candidates"]
        all_mutants += [{"id": item["id"], **c} for c in cands]
        pool = [c for c in cands if not _crashes_everywhere(c["screen"])] or cands
        # Balance bug types across the whole pool (HE types included).
        least = min(counts[c["bug_type"]] for c in pool)
        pool = [c for c in pool if counts[c["bug_type"]] == least]
        choice = rng.choice(pool)
        counts[choice["bug_type"]] += 1
        item.update(
            buggy=choice["code"],
            bug_origin=f"mutator:{choice['operator']}",
            bug_type=choice["bug_type"],
            mutant_detail=choice["detail"],
            screen=choice["screen"],
            n_valid_mutants=len(cands),
        )
        del item["candidates"]
    out = {"items": items, "drops": prev["drops"], "all_mutants": all_mutants}
    save("select", out)
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def funnel(stage_out: dict) -> dict:
    items, drops = stage_out["items"], stage_out["drops"]
    by = Counter((d["source"], d["stage"], d["reason"]) for d in drops)
    kept = Counter(i["source"] for i in items)
    rep = {
        "kept": dict(kept),
        "kept_total": len(items),
        "drops": {f"{s}/{st}/{r}": n for (s, st, r), n in sorted(by.items())},
    }
    rep["bug_type_by_source"] = {
        src: dict(Counter(i["bug_type"] for i in items if i["source"] == src and "bug_type" in i))
        for src in sorted(kept)
    }
    he = [i for i in items if i["source"] == "humaneval" and "he_clean_base" in i]
    if he:
        rep["he_clean_base"] = dict(Counter(i["he_clean_base"] for i in he))
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["assemble", "screen", "select", "spec", "finalize", "all", "nollm"])
    ap.add_argument("--limit", type=int, default=None, help="per-source cap (smoke runs)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--verbose", action="store_true", help="print the full funnel after each stage")
    args = ap.parse_args(argv)

    stages = {
        "all": ["assemble", "screen", "select", "spec", "finalize"],
        "nollm": ["assemble", "screen", "select"],
    }.get(args.stage, [args.stage])
    overall = tqdm(stages, desc="pool build", unit="stage", position=0, dynamic_ncols=True)
    for st in overall:
        overall.set_postfix_str(st)
        t0 = time.time()
        if st == "assemble":
            out = stage_assemble(args.limit)
        elif st == "screen":
            out = stage_screen(args.workers)
        elif st == "select":
            out = stage_select()
        elif st == "spec":
            from .spec import stage_spec

            out = stage_spec(args.workers)
        elif st == "finalize":
            from .finalize import stage_finalize

            out = stage_finalize(args.workers)
        f = funnel(out)
        per_source = ", ".join(f"{k} {v}" for k, v in sorted(f["kept"].items()))
        tqdm.write(f"[{st}] {time.time() - t0:.0f}s  kept {f['kept_total']} ({per_source})  "
                   f"dropped so far {len(out['drops'])}")
        if args.verbose:
            tqdm.write(json.dumps(f, indent=1))
    overall.close()
    if "finalize" in stages:
        print(f"Done. Pool: {POOL / 'pool.jsonl'}  Report: {POOL / 'build_report.json'}")


if __name__ == "__main__":
    main()
