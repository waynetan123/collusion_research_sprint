"""Harness command line.

  python -m collusion.harness run CONFIG [--no-fixtest] [--fix-mode call|batch]
  python -m collusion.harness branch CONFIG --session SID [--round K] [--transforms t1,t2,...]
  python -m collusion.harness summary RUN_ID
  python -m collusion.harness verify CONFIG
  python -m collusion.harness fixtest CONFIG            (backfill untested fixes)
  python -m collusion.harness judge RUN_ID [--limit N]
  python -m collusion.harness handcheck RUN_ID [--n 50]
  python -m collusion.harness agreement RUN_ID
  python -m collusion.harness smoke CONFIG              (live vLLM: 1 seed, 5 rounds, checks)
  python -m collusion.harness fetch-tokenizer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from tqdm import tqdm

from . import config as C
from .logio import RunDir

DEFAULT_ARM3 = ["continue", "truncate", "wipe", "truncate_wipe", "blind"]


def _tokenizer(required: bool):
    from .tokens import LocalTokenizer, fetch_tokenizer

    try:
        fetch_tokenizer()
        return LocalTokenizer()
    except Exception as e:
        if required:
            raise
        print(f"note: no local tokenizer ({e}); fake token counts are estimates", file=sys.stderr)
        return None


def _progress():
    bars: dict = {}

    def cb(seed, rounds: dict):
        for name, r in rounds.items():
            key = (seed, name)
            if key not in bars:
                bars[key] = tqdm(desc=f"seed {seed} {name}", unit="round", position=len(bars), dynamic_ncols=True)
            bars[key].n = r
            bars[key].refresh()

    return cb, bars


def cmd_run(args) -> None:
    from .runner import Context, run_main

    cfg = C.load(args.config)
    ctx = Context(cfg, _tokenizer(required=False), fix_testing=not args.no_fixtest, fix_mode=args.fix_mode)
    cb, bars = _progress()

    async def go():
        try:
            return await run_main(ctx, cb)
        finally:
            await ctx.close()

    res = asyncio.run(go())
    for b in bars.values():
        b.close()
    print(json.dumps(res, indent=1))


def cmd_branch(args) -> None:
    from .runner import Context, checkpoint_round, run_branches

    cfg = C.load(args.config)
    transforms = args.transforms.split(",") if args.transforms else DEFAULT_ARM3
    bad = set(transforms) - set(C.TRANSFORMS)
    if bad:
        raise SystemExit(f"unknown transforms {sorted(bad)}")
    ctx = Context(cfg, _tokenizer(required=True), fix_testing=not args.no_fixtest, fix_mode=args.fix_mode)
    k = args.round if args.round is not None else checkpoint_round(ctx, args.session)
    cb, bars = _progress()

    async def go():
        try:
            return await run_branches(ctx, args.session, k, transforms, cb)
        finally:
            await ctx.close()

    res = asyncio.run(go())
    for b in bars.values():
        b.close()
    print(json.dumps(res, indent=1))


def cmd_summary(args) -> None:
    from .report import summarize

    rows = summarize(RunDir(args.run_id))
    keys = ["session", "status", "rounds", "partner_miss", "stranger_miss", "gap", "excess_partner_miss",
            "false_reject_partner", "first_attempt_fail_rate", "verdicts_imputed", "fix_reprompt_rate",
            "fix_missing_rate", "fix_pass_rate_correct_rejections", "mean_prompt_tokens", "max_prompt_tokens",
            "completion_tokens_per_round", "cached_token_share"]
    for row in rows:
        print("  ".join(f"{k}={row.get(k)}" for k in keys))
        for d in row["detector"]:
            print("     detector:", {k: v for k, v in d.items() if k != "type"})
    print(f"(full table: {RunDir(args.run_id).path / 'summary.json'})")


def cmd_verify(args) -> None:
    from collusion.pool.io import load_pool

    from .report import verify

    cfg = C.load(args.config)
    pool = {r["id"]: r for r in load_pool()}
    res = verify(RunDir(cfg.run_id), pool, cfg.game)
    print(json.dumps(res, indent=1))
    raise SystemExit(1 if res["problems"] else 0)


def cmd_fixtest(args) -> None:
    """Test every logged fix that has no fixtest event yet (e.g. after a crash)."""
    from collusion.pool.io import load_pool

    from .fixtest import FixTester

    cfg = C.load(args.config)
    run = RunDir(cfg.run_id)
    pool = {r["id"]: r for r in load_pool()}

    async def go():
        ft = FixTester(cfg.fix_workers, args.fix_mode)
        n = 0
        try:
            for sid in run.sessions():
                done = {(e["round"], e["agent"], e["label"]) for e in run.events(sid, "fixtest") if "fix_passes" in e}
                jobs = []
                for e in run.events(sid, "round"):
                    for lb, rv in e["reviews"].items():
                        key = (e["round"], e["agent"], lb)
                        if rv["verdict"] != "REJECT" or rv["fix_missing"] or rv["forced"] or key in done:
                            continue
                        rec = pool[rv["fn_id"]]
                        meta = {"type": "fixtest", "round": e["round"], "agent": e["agent"], "label": lb,
                                "slot": rv["slot"], "fn_id": rv["fn_id"], "buggy": rv["buggy"]}
                        jobs.append((meta, ft.submit(rv["fn_id"], rec["entry_point"], rec["source"], rec["atol"], rv["fix"])))
                for meta, fut in jobs:
                    run.append(sid, [{**meta, **(await fut)}])
                    n += 1
        finally:
            ft.close()
        return n

    print(f"tested {asyncio.run(go())} fixes")


def cmd_judge(args) -> None:
    from collusion.pool.io import load_pool

    from .judge import run_judge

    pool = {r["id"]: r for r in load_pool()}
    print(json.dumps(run_judge(RunDir(args.run_id), pool, limit=args.limit), indent=1))


def cmd_handcheck(args) -> None:
    from collusion.pool.io import load_pool

    from .judge import write_handcheck

    pool = {r["id"]: r for r in load_pool()}
    print(write_handcheck(RunDir(args.run_id), pool, args.n))


def cmd_agreement(args) -> None:
    from .judge import agreement

    print(json.dumps(agreement(RunDir(args.run_id)), indent=1))


def cmd_smoke(args) -> None:
    """Live check against vLLM: tokenizer agreement, a short lockstep run, cache hits."""
    from collusion.pool.io import load_pool

    from . import render
    from .backend import VLLMBackend
    from .runner import Context, run_main
    from .schedule import make_schedule
    from .state import SessionState

    cfg = C.load(args.config)
    if cfg.backend.kind != "vllm":
        raise SystemExit("smoke needs backend.kind = 'vllm'")
    # arm3_rounds=0: no arm 3 follows a smoke run, so arm 2 plays all rounds (its deadline is cap - arm3_rounds).
    cfg = cfg.replace(seeds=cfg.seeds[:1], max_rounds=args.rounds, arm3_rounds=0, run_id=cfg.run_id + "-smoke")
    tok = _tokenizer(required=True)
    pool_list = load_pool()
    pool = {r["id"]: r for r in pool_list}

    async def go():
        be = VLLMBackend(cfg)
        try:
            m0 = await be.metrics()
            sched = make_schedule(pool_list, cfg.seeds[0], args.rounds, cfg.game.buggy_share, cfg.game.block_size)
            mism = []
            for arm in ("arm1", "arm2"):
                st = SessionState.new("tok", cfg.run_id, cfg.seeds[0], arm, 0, 1, "", "")
                for a in ("A", "B"):
                    msgs = render.review_messages(st, a, sched[1], pool, cfg.game)
                    local, remote = tok.count_chat(msgs), await be.tokenize_chat(msgs)
                    if local != remote:
                        mism.append((arm, a, local, remote))
            print("tokenizer check:", "OK" if not mism else f"MISMATCH {mism}")
        finally:
            await be.aclose()
        ctx = Context(cfg, tok, fix_testing=True, fix_mode=args.fix_mode)
        cb, bars = _progress()
        try:
            res = await run_main(ctx, cb)
            for b in bars.values():
                b.close()
            m1 = await ctx.backend.metrics()
        finally:
            await ctx.close()
        return res, {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}

    res, dm = asyncio.run(go())
    print(json.dumps(res, indent=1))
    print("vLLM metric deltas:", json.dumps(dm, indent=1))
    q, h = dm.get("prefix_cache_queries", 0), dm.get("prefix_cache_hits", 0)
    print(f"prefix cache hit rate: {h / q:.3f}" if q else "prefix cache: no queries reported")
    cmd_summary(argparse.Namespace(run_id=cfg.run_id))


def cmd_fetch_tokenizer(args) -> None:
    from .tokens import fetch_tokenizer

    print(fetch_tokenizer())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m collusion.harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("config")
    p.add_argument("--no-fixtest", action="store_true")
    p.add_argument("--fix-mode", default="call", choices=["call", "batch"])
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("branch")
    p.add_argument("config")
    p.add_argument("--session", required=True)
    p.add_argument("--round", type=int, default=None, help="default: the session's confirmed checkpoint")
    p.add_argument("--transforms", default=None, help=f"comma-separated; default {','.join(DEFAULT_ARM3)}")
    p.add_argument("--no-fixtest", action="store_true")
    p.add_argument("--fix-mode", default="call", choices=["call", "batch"])
    p.set_defaults(fn=cmd_branch)

    for name, fn in (("summary", cmd_summary), ("judge", cmd_judge), ("handcheck", cmd_handcheck),
                     ("agreement", cmd_agreement)):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        if name == "judge":
            p.add_argument("--limit", type=int, default=None)
        if name == "handcheck":
            p.add_argument("--n", type=int, default=50)
        p.set_defaults(fn=fn)

    p = sub.add_parser("verify")
    p.add_argument("config")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("fixtest")
    p.add_argument("config")
    p.add_argument("--fix-mode", default="call", choices=["call", "batch"])
    p.set_defaults(fn=cmd_fixtest)

    p = sub.add_parser("smoke")
    p.add_argument("config")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--fix-mode", default="call", choices=["call", "batch"])
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("fetch-tokenizer")
    p.set_defaults(fn=cmd_fetch_tokenizer)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
