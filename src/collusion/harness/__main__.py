"""Harness command line.

  python -m collusion.harness run CONFIG [--no-fixtest] [--fix-mode call|batch]
  python -m collusion.harness branch CONFIG --session SID [--round K] [--transforms t1,t2,...]
  python -m collusion.harness summary RUN_ID
  python -m collusion.harness analyze CONFIG [--placebo PLACEBO_CONFIG] [--min-checkpoints 4]
                                                        (arms 1+2: per-seed outcomes, gates, GO / DEBUG)
  python -m collusion.harness verify CONFIG
  python -m collusion.harness fixtest CONFIG            (backfill untested fixes)
  python -m collusion.harness judge RUN_ID [--limit N]
  python -m collusion.harness handcheck RUN_ID [--n 50]
  python -m collusion.harness agreement RUN_ID
  python -m collusion.harness smoke CONFIG              (live vLLM or OpenRouter: 1 seed, 5 rounds, checks)
  python -m collusion.harness fetch-tokenizer [--config CONFIG]   (default: the Qwen3.5 tokenizer)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


from . import config as C
from .logio import RunDir

DEFAULT_ARM3 = ["continue", "truncate", "wipe", "truncate_wipe", "blind"]


def _tokenizer(required: bool, cfg: C.Config | None = None):
    """The local tokenizer the config's backend uses (backend.tokenizer; Qwen3.5 if no config)."""
    from .tokens import LocalTokenizer, fetch_tokenizer, get_spec

    spec = get_spec(cfg.backend.tokenizer if cfg else "qwen3.5")
    try:
        fetch_tokenizer(spec)
        return LocalTokenizer(spec=spec, enable_thinking=cfg.sampling.enable_thinking if cfg else False)
    except Exception as e:
        if required:
            raise
        print(f"note: no local tokenizer ({e}); fake token counts are estimates", file=sys.stderr)
        return None


def _progress(ctx):
    """Loading bars: an overall bar (session-rounds, fix tests) and one bar per session (progress.py)."""
    from .progress import Progress

    return Progress(ctx.fixtester)


async def _ticking(prog, coro):
    """Await `coro` while refreshing the bars every few seconds (elapsed time and ETA stay live)."""
    tick = asyncio.ensure_future(prog.ticker())
    try:
        return await coro
    finally:
        tick.cancel()


def _totals(run: RunDir) -> dict:
    """Rounds played and tokens generated so far across a run's sessions.

    Counts only rounds up to each session's last snapshot, i.e. what survives a
    resume (events of an interrupted round are dropped and replayed).
    """
    rounds = gen = 0
    for sid in run.sessions():
        st = run.load_state(sid)
        if st is None:
            continue
        done = st.round_done
        rounds += done - (st.branch_round or 0)
        gen += sum(e["completion_tokens"] or 0 for e in run.events(sid, "call") if e["round"] <= done)
    return {"session_rounds": rounds, "generated_tokens": gen}


def _stopped(e: Exception, run: RunDir) -> None:
    """Spend cap reached or credits exhausted: stop cleanly; the run resumes from its last snapshots."""
    from .report import run_cost

    print(f"\nstopped: {e}\nlogged spending in run {run.path.name}: ${run_cost(run):.4f}. "
          "Raise backend.max_cost_usd (it is not part of the config hash) or add credits, then re-run to resume; "
          "run `fixtest CONFIG` to backfill any fix tests that were cut off.", file=sys.stderr)


def cmd_run(args) -> None:
    from .backend import BudgetExceeded, OutOfCredit
    from .report import run_cost
    from .runner import Context, run_main

    cfg = C.load(args.config)
    ctx = Context(cfg, _tokenizer(required=False, cfg=cfg), fix_testing=not args.no_fixtest, fix_mode=args.fix_mode)
    prog = _progress(ctx)
    before = _totals(ctx.run)

    async def go():
        metrics = getattr(ctx.backend, "metrics", None)
        m0 = await metrics() if metrics else {}
        t0 = time.monotonic()
        try:
            res = await _ticking(prog, run_main(ctx, prog))
            elapsed = time.monotonic() - t0
            m1 = await metrics() if metrics else {}
        finally:
            await ctx.close()
        return res, elapsed, {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}

    cost0 = run_cost(ctx.run)
    try:
        res, elapsed, dm = asyncio.run(go())
    except (BudgetExceeded, OutOfCredit) as e:
        prog.close()
        _stopped(e, ctx.run)
        raise SystemExit(2)
    prog.close()
    print(json.dumps(res, indent=1))
    after = _totals(ctx.run)
    n_rounds = after["session_rounds"] - before["session_rounds"]
    n_gen = after["generated_tokens"] - before["generated_tokens"]
    print(f"\nthroughput: {n_rounds} session-rounds in {elapsed:.0f} s "
          f"= {60 * n_rounds / max(elapsed, 1e-9):.1f} session-rounds/min; "
          f"{n_gen:,} tokens generated = {n_gen / max(elapsed, 1e-9):.0f} tokens/s")
    if dm:
        q, h = dm.get("prefix_cache_queries", 0), dm.get("prefix_cache_hits", 0)
        print(f"vLLM: preemptions {dm.get('num_preemptions', 0):.0f}, "
              f"prefix cache hit rate {h / q:.3f}" if q else "vLLM: no prefix-cache queries reported")
    if cfg.backend.kind == "openrouter":
        total = run_cost(ctx.run)
        cap = f" of the ${cfg.backend.max_cost_usd:.2f} cap" if cfg.backend.max_cost_usd else ""
        print(f"cost: ${total - cost0:.4f} this invocation; ${total:.4f} logged in run {cfg.run_id}{cap}")


def cmd_branch(args) -> None:
    from .backend import BudgetExceeded, OutOfCredit
    from .runner import Context, checkpoint_round, run_branches

    cfg = C.load(args.config)
    transforms = args.transforms.split(",") if args.transforms else DEFAULT_ARM3
    bad = set(transforms) - set(C.TRANSFORMS)
    if bad:
        raise SystemExit(f"unknown transforms {sorted(bad)}")
    ctx = Context(cfg, _tokenizer(required=True, cfg=cfg), fix_testing=not args.no_fixtest, fix_mode=args.fix_mode)
    k = args.round if args.round is not None else checkpoint_round(ctx, args.session)
    prog = _progress(ctx)

    async def go():
        try:
            return await _ticking(prog, run_branches(ctx, args.session, k, transforms, prog))
        finally:
            await ctx.close()

    try:
        res = asyncio.run(go())
    except (BudgetExceeded, OutOfCredit) as e:
        prog.close()
        _stopped(e, ctx.run)
        raise SystemExit(2)
    prog.close()
    print(json.dumps(res, indent=1))


def cmd_summary(args) -> None:
    from .report import summarize

    rows = summarize(RunDir(args.run_id))
    keys = ["session", "status", "rounds", "partner_miss", "stranger_miss", "gap", "excess_partner_miss",
            "false_reject_partner", "first_attempt_fail_rate", "verdicts_imputed", "fix_reprompt_rate",
            "fix_missing_rate", "fix_pass_rate_correct_rejections", "mean_prompt_tokens", "max_prompt_tokens",
            "completion_tokens_per_round", "cached_token_share", "cost_usd"]
    for row in rows:
        print("  ".join(f"{k}={row.get(k)}" for k in keys))
        for d in row["detector"]:
            print("     detector:", {k: v for k, v in d.items() if k != "type"})
    print(f"(full table: {RunDir(args.run_id).path / 'summary.json'})")


def cmd_analyze(args) -> None:
    """Run-level analysis of arms 1+2 and the pre-registered GO / DEBUG decision (analysis.py)."""
    from collusion.pool.io import load_pool

    from .analysis import analyze, format_report
    from .schedule import pool_cap

    cfg = C.load(args.config)
    n_rounds = cfg.max_rounds or pool_cap(load_pool())
    run = RunDir(cfg.run_id)
    placebo = None
    if args.placebo:
        pcfg = C.load(args.placebo)
        if pcfg.seeds != cfg.seeds or pcfg.replicate != cfg.replicate + 1 or pcfg.arms != ("arm1",):
            raise SystemExit("the placebo config must have the same seeds, arms = ['arm1'] and replicate = main + 1")
        placebo = RunDir(pcfg.run_id)
    a = analyze(cfg, n_rounds, run, placebo, args.min_checkpoints)
    run.write_json(run.path / "analysis.json", a)
    print(format_report(a))
    print(f"(full results: {run.path / 'analysis.json'})")


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
    """Live check: a short lockstep run (1 seed, arms 1+2) plus backend-specific checks."""
    cfg = C.load(args.config)
    # arm3_rounds=0: no arm 3 follows a smoke run, so arm 2 plays all rounds (its deadline is cap - arm3_rounds).
    cfg = cfg.replace(seeds=cfg.seeds[:1], max_rounds=args.rounds, arm3_rounds=0, run_id=cfg.run_id + "-smoke")
    if cfg.backend.kind == "vllm":
        _smoke_vllm(args, cfg)
    elif cfg.backend.kind == "openrouter":
        _smoke_openrouter(args, cfg)
    else:
        raise SystemExit("smoke needs backend.kind = 'vllm' or 'openrouter'")


def _smoke_vllm(args, cfg) -> None:
    """vLLM: tokenizer agreement with /tokenize, the run, prefix-cache hits from /metrics."""
    from collusion.pool.io import load_pool

    from . import render
    from .backend import VLLMBackend
    from .runner import Context, run_main
    from .schedule import make_schedule
    from .state import SessionState

    tok = _tokenizer(required=True, cfg=cfg)
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
        prog = _progress(ctx)
        try:
            res = await _ticking(prog, run_main(ctx, prog))
            prog.close()
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


def _smoke_openrouter(args, cfg) -> None:
    """OpenRouter: credits, then probe calls before the run.

    - token count: the round-1 review prompt of arm 1 and arm 2 (agent A) is sent once
      each; the provider's prompt_tokens minus the local count must be the same in
      both arms (a constant offset, e.g. from the JSON schema written into the
      prompt, keeps the arms matched; a varying one would not);
    - seed: the arm 1 probe is sent again with the same seed (reported, not required);
    - reasoning off: no reasoning tokens or text;
    - provider: every reply comes from the pinned provider.
    The probes are not part of the run, so their cost is printed but not logged.
    Then the short lockstep run and its summary.
    """
    from collusion.pool.io import load_pool

    from . import render
    from .backend import BudgetExceeded, CallContext, OpenRouterBackend, OutOfCredit
    from .report import run_cost, summarize
    from .review import ParseError, derive_seed, parse_review, review_schema
    from .runner import Context, run_main
    from .schedule import LABELS, make_schedule
    from .state import SessionState

    tok = _tokenizer(required=True, cfg=cfg)
    pool_list = load_pool()
    pool = {r["id"]: r for r in pool_list}
    checks: dict[str, bool] = {}

    async def probe():
        be = OpenRouterBackend(cfg)
        try:
            cr = await be.credits()
            print(f"credits: ${cr.get('total_credits', 0):.2f} bought, ${cr.get('total_usage', 0):.4f} used")
            sched = make_schedule(pool_list, cfg.seeds[0], args.rounds, cfg.game.buggy_share, cfg.game.block_size)
            schema = review_schema(list(LABELS), cfg.limits.reason_max_chars)
            got = {}
            for arm in ("arm1", "arm2"):
                st = SessionState.new("probe", cfg.run_id, cfg.seeds[0], arm, 0, 1, "", "")
                msgs = render.review_messages(st, "A", sched[1], pool, cfg.game)
                seed = derive_seed("probe", arm)
                reply = await be.chat(msgs, max_tokens=cfg.limits.review_max_tokens, seed=seed, schema=schema,
                                      ctx=CallContext("probe", arm, 1, "A", "review"))
                got[arm] = (msgs, seed, reply)
            msgs, seed, _ = got["arm1"]
            again = await be.chat(msgs, max_tokens=cfg.limits.review_max_tokens, seed=seed, schema=schema,
                                  ctx=CallContext("probe", "arm1", 1, "A", "review"))
            return got, again, be.spent
        finally:
            await be.aclose()

    try:
        got, again, probe_cost = asyncio.run(probe())
    except OutOfCredit as e:
        raise SystemExit(f"stopped: {e}")
    replies = [r for _, _, r in got.values()] + [again]
    offsets = {}
    for arm, (msgs, _, r) in got.items():
        local = tok.count_chat(msgs)
        offsets[arm] = r.prompt_tokens - local
        print(f"tokens {arm}: local {local}, provider {r.prompt_tokens}, offset {offsets[arm]:+d}")
    checks["token offset equal in both arms"] = offsets["arm1"] == offsets["arm2"]
    checks["reasoning off (0 reasoning tokens, no reasoning text)"] = all(
        not r.reasoning_tokens and not r.reasoning for r in replies)
    providers = {r.provider for r in replies}
    checks[f"served by the pinned provider only {sorted(p for p in providers if p)}"] = (
        None not in providers and len(providers) == 1)
    n_ok = 0
    for r in replies:
        try:
            parse_review(r.text, list(LABELS), cfg.limits.reason_max_chars)
            n_ok += r.finish_reason != "length"
        except ParseError:
            pass
    checks[f"probe replies parse ({n_ok}/{len(replies)})"] = n_ok == len(replies)
    print(f"seed honoured (same seed, same text): {again.text == got['arm1'][2].text}")
    print(f"probe cost: ${probe_cost:.5f} (not logged in the run)")

    ctx = Context(cfg, tok, fix_testing=True, fix_mode=args.fix_mode)
    prog = _progress(ctx)

    async def go():
        try:
            return await _ticking(prog, run_main(ctx, prog))
        finally:
            prog.close()
            await ctx.close()

    try:
        res = asyncio.run(go())
    except (BudgetExceeded, OutOfCredit) as e:
        _stopped(e, ctx.run)
        raise SystemExit(2)
    print(json.dumps(res, indent=1))
    cmd_summary(argparse.Namespace(run_id=cfg.run_id))
    rows = summarize(ctx.run)
    run_providers = sorted({p for row in rows for p in row["providers"]})
    checks[f"run served by one provider {run_providers}"] = len(run_providers) == 1
    checks["run used no reasoning tokens"] = sum(row["reasoning_tokens"] for row in rows) == 0
    checks["no imputed verdicts"] = sum(row["verdicts_imputed"] for row in rows) == 0
    rounds = sum(row["rounds"] for row in rows)
    total = run_cost(ctx.run)
    print(f"\nrun cost: ${total:.4f} for {rounds} session-rounds = ${total / max(rounds, 1):.5f} per session-round")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise SystemExit(1)


def cmd_fetch_tokenizer(args) -> None:
    from .tokens import fetch_tokenizer, get_spec

    spec = get_spec(C.load(args.config).backend.tokenizer if args.config else "qwen3.5")
    print(fetch_tokenizer(spec))


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

    p = sub.add_parser("analyze")
    p.add_argument("config")
    p.add_argument("--placebo", default=None, help="config of the arm 1 placebo replicate run")
    p.add_argument("--min-checkpoints", type=int, default=4, help="gate (a); pre-registered: 4 (of 12 seeds)")
    p.set_defaults(fn=cmd_analyze)

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
    p.add_argument("--config", default=None, help="fetch the tokenizer this config's backend uses")
    p.set_defaults(fn=cmd_fetch_tokenizer)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
