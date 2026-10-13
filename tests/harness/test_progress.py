"""Loading bars: totals, early stops, resumes and the overall bar."""

import asyncio
import io

from tqdm import tqdm

from collusion.harness import config as C
from collusion.harness.progress import Progress
from collusion.harness.runner import Context, run_main

# tqdm's monitor thread would make later sandbox tests fork from a multi-threaded process.
tqdm.monitor_interval = 0


def test_bars_track_totals_early_stop_and_resume():
    p = Progress(file=io.StringIO())
    p(1, {"arm1": (0, 111, "running"), "arm2": (0, 61, "running")})
    assert p.overall.total == 172 and p.overall.n == 0
    p(1, {"arm1": (49, 111, "running"), "arm2": (49, 61, "checkpointed")})  # arm 2 stops early
    assert p.bars[(1, "arm2")].total == 49 and p.overall.total == 160 and p.overall.n == 98
    p(2, {"arm1": (30, 111, "running")})  # a resumed session starts at its played rounds
    assert p.bars[(2, "arm1")].n == 30 and p.overall.n == 128 and p.overall.total == 271
    p.close()


def test_run_reports_every_session_to_completion(tmp_path, pool_list):
    cfg = C.from_dict({"run_id": "pb", "seeds": [1, 2], "max_rounds": 50, "arm3_rounds": 0,
                       "fake": {"arm1": {"kind": "honest"}, "arm2": {"kind": "colluder", "collude_from": 5}}})
    ctx = Context(cfg, None, fix_testing=False, base=tmp_path)
    p = Progress(file=io.StringIO())

    async def go():
        try:
            return await run_main(ctx, p)
        finally:
            await ctx.close()

    asyncio.run(go())
    assert {k: (b.n, b.total) for k, b in p.bars.items()} == {
        (1, "arm1"): (50, 50), (1, "arm2"): (49, 49), (2, "arm1"): (50, 50), (2, "arm2"): (49, 49)}
    assert p.overall.n == p.overall.total == 198
    assert p.status[(1, "arm2")] == "checkpointed" and p.status[(1, "arm1")] == "done"
    p.close()
