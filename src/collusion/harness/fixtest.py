"""Fix testing: run each rejection's fix on the stored EvalPlus inputs.

Uses the Step 1 sandbox (`sandbox.run_calls` + `sandbox.judge`) in the same
mode as pool verification (process per call by default). Runs in spawn-started
worker processes: they hold no threads, so the sandbox may fork. Results never
reach agents or scores; they are logged as `fixtest` events (fix quality).

Test data comes from the pool sidecar via `collusion.pool.io.load_tests`, split
once into per-function pickles so each worker loads only what it needs.
"""

from __future__ import annotations

import asyncio
import functools
import multiprocessing as mp
import pickle
import time
from pathlib import Path

from collusion.pool import io as pool_io

from .review import clean_fix

CACHE = pool_io.ROOT / "data" / "interim" / "tests_by_id"
MAX_TIMEOUTS = 10  # as in Step 1 final verification


def _cache_path(fn_id: str) -> Path:
    return CACHE / (fn_id.replace("/", "__") + ".pkl")


def ensure_test_cache() -> None:
    """Split pool_tests.jsonl.gz into per-id pickles (once; rebuilt if the sidecar is newer)."""
    stamp = CACHE / ".source_mtime"
    src_mtime = str(pool_io.TESTS_FILE.stat().st_mtime_ns)
    if stamp.exists() and stamp.read_text() == src_mtime:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    for fn_id, t in pool_io.load_tests().items():
        with open(_cache_path(fn_id), "wb") as f:
            pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL)
    stamp.write_text(src_mtime)


@functools.lru_cache(maxsize=32)
def _tests(fn_id: str) -> dict:
    with open(_cache_path(fn_id), "rb") as f:
        return pickle.load(f)


def run_fix_test(job: dict) -> dict:
    """job: {fn_id, entry_point, source, atol, fix, mode}. Runs in a worker process."""
    from collusion.pool.sandbox import judge, run_calls

    t0 = time.time()
    t = _tests(job["fn_id"])
    res = run_calls(clean_fix(job["fix"]), job["entry_point"], t["inputs"], t["timeouts"],
                    mode=job.get("mode", "call"), max_timeouts=MAX_TIMEOUTS)
    v = judge(res, t["expected"], job["atol"], job["entry_point"], job["source"])
    return {"fix_passes": v.all_pass, "n_pass": v.n_pass, "n_inputs": v.n_inputs, "kinds": dict(sorted(v.kinds.items())),
            "first_failure": v.first_failure, "seconds": round(time.time() - t0, 2)}


class FixTester:
    """Background fix testing on a spawn-started `multiprocessing.Pool`.

    (`concurrent.futures.ProcessPoolExecutor` is avoided: it probes
    SC_SEM_NSEMS_MAX, which some sandboxes forbid; Step 1 used the same Pool.)
    """

    def __init__(self, workers: int, mode: str = "call"):
        ensure_test_cache()
        self.mode = mode
        self.pool = mp.get_context("spawn").Pool(workers)

    def submit(self, fn_id: str, entry_point: str, source: str, atol: float, fix: str) -> asyncio.Future:
        job = {"fn_id": fn_id, "entry_point": entry_point, "source": source, "atol": atol, "fix": fix, "mode": self.mode}
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def ok(res):
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(res))

        def err(exc):
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_exception(exc))

        self.pool.apply_async(run_fix_test, (job,), callback=ok, error_callback=err)
        return fut

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
