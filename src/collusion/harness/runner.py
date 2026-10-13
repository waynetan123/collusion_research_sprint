"""Run sessions: arm 1 and arm 2 in lockstep per seed, and checkpoint branches.

Arm 1 and arm 2 of a seed share the schedule and advance one round at a time
together, so the detector compares them at the same rounds. Arm 1 runs to the
round cap; arm 2 stops at a confirmed checkpoint or at the deadline
(cap - arm3_rounds). Sessions resume from their last snapshot.
"""

from __future__ import annotations

import asyncio
import json

from collusion.pool.io import load_pool

from . import detector as D
from .backend import make_backend
from .branch import make_branch
from .config import Config
from .logio import RunDir
from .report import run_cost
from .schedule import Schedule, make_schedule, pool_cap
from .session import Session
from .state import SessionState


def session_id(seed: int, arm: str, replicate: int) -> str:
    return f"s{seed}-{arm}-rep{replicate}"


def branch_id(parent: str, transform: str, k: int) -> str:
    return f"{parent}~{transform}@{k}"


class Context:
    """Shared objects for one invocation: config, run dir, pool, schedules, backend, fix tester."""

    def __init__(self, cfg: Config, tokenizer=None, fix_testing: bool = True, fix_mode: str = "call", base=None):
        self.cfg = cfg
        self.run = RunDir(cfg.run_id, **({"base": base} if base else {}))
        self._check_config()
        self.pool_list = load_pool()
        self.pool = {r["id"]: r for r in self.pool_list}
        self.cap = pool_cap(self.pool_list)
        self.n_rounds = cfg.max_rounds or self.cap
        if self.n_rounds > self.cap:
            raise ValueError(f"max_rounds {self.n_rounds} > pool cap {self.cap}")
        self.tokenizer = tokenizer
        # Spending already logged in this run counts toward the spend cap (resume).
        self.backend = make_backend(cfg, tokenizer, spent=run_cost(self.run))
        self.fixtester = None
        if fix_testing:
            from .fixtest import FixTester

            self.fixtester = FixTester(cfg.fix_workers, fix_mode)
        self._schedules: dict[int, Schedule] = {}

    def _check_config(self) -> None:
        p = self.run.path / "config.json"
        if p.exists():
            old = json.loads(p.read_text())
            if old["hash"] != self.cfg.hash:
                raise RuntimeError(f"{p} was written by a different config ({old['hash']} != {self.cfg.hash}); "
                                   "use a new run_id")
        else:
            self.run.write_json(p, {"hash": self.cfg.hash, "config": self.cfg.as_dict()})

    def schedule(self, seed: int) -> Schedule:
        if seed not in self._schedules:
            p = self.run.schedule_path(seed)
            g = self.cfg.game
            # Always dealt for the full cap; max_rounds only limits play, so shorter runs use a prefix.
            sched = make_schedule(self.pool_list, seed, self.cap, g.buggy_share, g.block_size)
            if p.exists():
                saved = Schedule.from_json(json.loads(p.read_text()))
                if saved.hash != sched.hash:
                    raise RuntimeError(f"{p} differs from the regenerated schedule (pool changed?)")
            else:
                self.run.write_json(p, sched.to_json())
            self._schedules[seed] = sched
        return self._schedules[seed]

    def open_session(self, sid: str, seed: int, arm: str, last_round: int) -> Session:
        sched = self.schedule(seed)
        st = self.run.load_state(sid)
        if st is None:
            st = SessionState.new(sid, self.cfg.run_id, seed, arm, self.cfg.replicate, last_round,
                                  self.cfg.hash, sched.hash)
            self.run.save_state(st)  # round-0 snapshot: lets `verify` re-render round 1
        else:
            self.run.drop_unfinished(sid, st.round_done)
        return Session(st, sched, self.pool, self.cfg, self.backend, self.run, self.fixtester)

    async def close(self) -> None:
        await self.backend.aclose()
        if self.fixtester is not None:
            self.fixtester.close()


async def run_seed(ctx: Context, seed: int, progress=None) -> dict:
    cfg = ctx.cfg
    rep = cfg.replicate
    deadline = max(1, ctx.n_rounds - cfg.arm3_rounds)
    sessions = {}
    if "arm1" in cfg.arms:
        sessions["arm1"] = ctx.open_session(session_id(seed, "arm1", rep), seed, "arm1", ctx.n_rounds)
    if "arm2" in cfg.arms:
        if "arm1" not in cfg.arms:
            raise ValueError("arm 2 needs the matched arm 1 session in the same run")
        sessions["arm2"] = ctx.open_session(session_id(seed, "arm2", rep), seed, "arm2", min(deadline, ctx.n_rounds))

    det = None
    hist: dict[str, dict[int, dict]] = {}
    if "arm2" in sessions:
        # Rebuild per-round partner outcomes and the detector from logs (resume).
        for arm, s in sessions.items():
            hist[arm] = D.outcomes_from_events(ctx.run.events(s.state.session_id, "round"))
        det, _ = D.scan(hist["arm1"], hist["arm2"], cfg.detector)
        fed = max(set(hist["arm1"]) & set(hist["arm2"]), default=0)
        if det.checkpoint is not None and sessions["arm2"].state.status == "running":
            sessions["arm2"].state.status = "checkpointed"
            ctx.run.save_state(sessions["arm2"].state)

    while True:
        active = [s for s in sessions.values() if not s.done]
        if not active:
            break
        # Lockstep: play every active arm that is furthest behind (after a crash it catches up first).
        lo = min(s.state.round_done for s in active)
        step = [s for s in active if s.state.round_done == lo]
        outs = await asyncio.gather(*(s.play_round() for s in step))
        if progress:
            progress(seed, {s.state.arm: s.state.round_done for s in sessions.values()})
        if det is None:
            continue
        for s, o in zip(step, outs):
            hist[s.state.arm][o["round"]] = o["partner"]
        a2 = sessions["arm2"].state
        while det.checkpoint is None and fed + 1 in hist["arm1"] and fed + 1 in hist["arm2"]:
            fed += 1
            evs = det.update(fed, hist["arm1"][fed], hist["arm2"][fed])
            if evs:
                ctx.run.append(a2.session_id, evs)
        if det.checkpoint is not None and a2.status in ("running", "done"):
            a2.status = "checkpointed"
            ctx.run.save_state(a2)
        elif a2.status == "done":  # reached the deadline without a confirmed checkpoint
            a2.status = "deadline"
            ctx.run.save_state(a2)
            ctx.run.append(a2.session_id, [{"type": "detector", "round": a2.round_done, "event": "no_collusion"}])

    for s in sessions.values():
        await s.drain()
    return {arm: {"session": s.state.session_id, "round_done": s.state.round_done, "status": s.state.status}
            for arm, s in sessions.items()} | ({"checkpoint": det.checkpoint} if det else {})


async def _gather_or_cancel(coros) -> list:
    """gather(), but if one task fails (e.g. the spend cap) the others are cancelled before re-raising.

    Their unfinished rounds are dropped on resume, as after any crash.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def run_main(ctx: Context, progress=None) -> dict:
    res = await _gather_or_cancel(run_seed(ctx, seed, progress) for seed in ctx.cfg.seeds)
    return dict(zip(ctx.cfg.seeds, res))


def checkpoint_round(ctx: Context, parent_sid: str) -> int:
    evs = [e for e in ctx.run.events(parent_sid, "detector") if e["event"] == "checkpoint"]
    if not evs:
        raise ValueError(f"{parent_sid} has no checkpoint; pass --round")
    return evs[-1]["round"]


async def run_branches(ctx: Context, parent_sid: str, k: int, transforms: list[str], progress=None) -> dict:
    parent = ctx.run.snapshot(parent_sid, k)
    sched = ctx.schedule(parent.seed)
    sessions, reports = [], {}
    for t in transforms:
        sid = branch_id(parent_sid, t, k)
        st = ctx.run.load_state(sid)
        if st is None:
            st, rep = make_branch(parent, t, sid, sched, ctx.pool, ctx.cfg, ctx.tokenizer, ctx.n_rounds)
            ctx.run.save_state(st)
            ctx.run.append(sid, [{"type": "branch", "round": k, **rep}])
            reports[t] = rep
        else:
            ctx.run.drop_unfinished(sid, st.round_done)
        sessions.append(Session(st, sched, ctx.pool, ctx.cfg, ctx.backend, ctx.run, ctx.fixtester))

    async def drive(s: Session):
        while not s.done:
            await s.play_round()
            if progress:
                progress(parent.seed, {s.state.transform: s.state.round_done})
        await s.drain()

    await _gather_or_cancel(drive(s) for s in sessions)
    return {"reports": reports, "sessions": {s.state.transform: s.state.round_done for s in sessions}}
