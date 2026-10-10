"""Sandboxed execution of a candidate function on many inputs.

Two isolation modes:

- ``"call"``: one forked child process per call. This is the strict mode the
  design doc requires for verification ("each call in a separate process with a
  timeout and with arguments deep-copied").
- ``"batch"``: one forked child runs all inputs, each under a SIGALRM per-call
  timeout. It is ~100x faster and used for screening (reference outputs, mutant
  candidates). If the child hangs or dies, the remaining inputs are re-run in
  ``"call"`` mode, so results are always complete.

Children deep-copy arguments, run in a throwaway temp dir with stdio sent to
/dev/null and destructive os/shutil/subprocess functions disabled (EvalPlus's
``reliability_guard``). This guards against accidents, not adversaries.

Forking is only safe from a process without extra threads. Callers that hold
threads (e.g. HTTP clients) should invoke this from a spawn-started worker.
"""

from __future__ import annotations

import copy
import os
import pickle
import select
import shutil
import signal
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ._evalplus_vendor import outputs_match, reliability_guard

OK, ERROR, TIMEOUT, CRASH, SKIPPED = "ok", "error", "timeout", "crash", "skipped"

MIN_TIMEOUT = 1.0  # seconds; EvalPlus's DEFAULT_MIN_TIME_LIMIT
TIMEOUT_FACTOR = 10.0  # x reference runtime (EvalPlus uses 4; we are more lenient)
MEM_LIMIT_BYTES = 4 * 1024**3  # enforced on Linux only
_SPAWN_SLACK = 0.5  # seconds added to a per-call deadline for fork/exec overhead
_FRAME = struct.Struct("!Q")


@dataclass
class CallResult:
    status: str
    value: Any = None
    error: str | None = None
    runtime: float = 0.0


class _Timeout(BaseException):
    pass


def timeouts_from_runtimes(runtimes: Sequence[float]) -> list[float]:
    return [max(MIN_TIMEOUT, TIMEOUT_FACTOR * t) for t in runtimes]


# --------------------------------------------------------------------------- #
# child side
# --------------------------------------------------------------------------- #


def _alarm(signum, frame):
    raise _Timeout()


def _child_main(code_obj, entry_point, items, alarm, max_timeouts, workdir, wfd) -> None:
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    os.chdir(workdir)
    if sys.platform.startswith("linux"):
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    signal.signal(signal.SIGALRM, _alarm)
    reliability_guard()

    def emit(obj) -> None:
        try:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:  # unpicklable output
            i, _, _, _, rt = obj
            data = pickle.dumps((i, ERROR, None, f"UnpicklableOutput: {e!r}"[:300], rt))
        msg = _FRAME.pack(len(data)) + data
        view = memoryview(msg)
        while view:
            n = os.write(wfd, view)
            view = view[n:]

    globs: dict[str, Any] = {"__name__": "__sandbox__"}
    try:
        exec(code_obj, globs)
        fn = globs[entry_point]
    except BaseException as e:
        for i, _, _ in items:
            emit((i, ERROR, None, f"LoadError: {type(e).__name__}: {e}"[:300], 0.0))
        return

    n_timeouts = 0
    for i, inp, limit in items:
        if n_timeouts >= max_timeouts:
            emit((i, SKIPPED, None, "max_timeouts reached", 0.0))
            continue
        status, out, err = OK, None, None
        args = copy.deepcopy(inp)
        t0 = time.perf_counter()  # runtime covers the call only, as in EvalPlus
        try:
            if alarm:
                signal.setitimer(signal.ITIMER_REAL, limit)
            try:
                out = fn(*args)
            finally:
                if alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        except _Timeout:
            status, err = TIMEOUT, "per-call timeout"
            n_timeouts += 1
        except BaseException as e:
            status, err = ERROR, f"{type(e).__name__}: {e}"[:300]
        emit((i, status, out, err, time.perf_counter() - t0))


# --------------------------------------------------------------------------- #
# parent side
# --------------------------------------------------------------------------- #


def _fork_run(code_obj, entry_point, items, alarm, max_timeouts, workdir, deadline):
    """Run `items` [(index, input, timeout)] in one child; return {index: CallResult}."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(r)
            _child_main(code_obj, entry_point, items, alarm, max_timeouts, workdir, w)
        finally:
            os._exit(0)
    os.close(w)
    results: dict[int, CallResult] = {}
    buf = bytearray()
    end = time.monotonic() + deadline
    killed = False
    try:
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0 or not select.select([r], [], [], remaining)[0]:
                os.kill(pid, signal.SIGKILL)
                killed = True
                break
            chunk = os.read(r, 1 << 20)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= _FRAME.size:
                (n,) = _FRAME.unpack_from(buf)
                if len(buf) < _FRAME.size + n:
                    break
                i, status, out, err, rt = pickle.loads(bytes(buf[_FRAME.size : _FRAME.size + n]))
                del buf[: _FRAME.size + n]
                results[i] = CallResult(status, out, err, rt)
    finally:
        os.close(r)
        os.waitpid(pid, 0)
    return results, killed


def run_calls(
    code: str,
    entry_point: str,
    inputs: Sequence[Sequence[Any]],
    timeouts: Sequence[float] | float,
    *,
    mode: str = "call",
    max_timeouts: int = 10,
) -> list[CallResult]:
    """Call `entry_point(*inp)` defined by `code` for every input."""
    n = len(inputs)
    limits = [float(timeouts)] * n if isinstance(timeouts, (int, float)) else list(timeouts)
    assert len(limits) == n
    try:
        code_obj = compile(code, "<patch>", "exec")
    except SyntaxError as e:
        return [CallResult(ERROR, error=f"SyntaxError: {e}"[:300]) for _ in range(n)]

    workdir = tempfile.mkdtemp(prefix="sbx_")
    try:
        results: dict[int, CallResult] = {}
        if mode == "batch" and n:
            items = [(i, inputs[i], limits[i]) for i in range(n)]
            deadline = sum(limits) + 5.0
            results, _ = _fork_run(code_obj, entry_point, items, True, max_timeouts, workdir, deadline)
        elif mode != "call" and mode != "batch":
            raise ValueError(f"unknown mode {mode!r}")

        n_timeouts = sum(1 for res in results.values() if res.status == TIMEOUT)
        for i in range(n):
            if i in results:
                continue
            if n_timeouts >= max_timeouts:
                results[i] = CallResult(SKIPPED, error="max_timeouts reached")
                continue
            got, killed = _fork_run(
                code_obj, entry_point, [(i, inputs[i], limits[i])], False, 1, workdir,
                limits[i] + _SPAWN_SLACK,
            )
            if i in got:
                results[i] = got[i]
            elif killed:
                results[i] = CallResult(TIMEOUT, error="per-call timeout", runtime=limits[i])
            else:
                results[i] = CallResult(CRASH, error="child exited without a result")
            if results[i].status == TIMEOUT:
                n_timeouts += 1
        return [results[i] for i in range(n)]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# verification against reference outputs
# --------------------------------------------------------------------------- #


def short_repr(x: Any, limit: int = 120) -> str:
    try:
        r = repr(x)
    except Exception as e:  # e.g. ints beyond sys.get_int_max_str_digits()
        r = f"<unrepresentable {type(x).__name__}: {type(e).__name__}>"
    return r if len(r) <= limit else r[: limit - 3] + "..."


@dataclass
class Verdict:
    n_inputs: int
    n_pass: int
    n_fail: int  # wrong output + error + timeout + crash
    kinds: dict[str, int] = field(default_factory=dict)  # wrong/error/timeout/crash/skipped
    failing_idx: list[int] = field(default_factory=list)
    first_failure: str | None = None

    @property
    def all_pass(self) -> bool:
        return self.n_pass == self.n_inputs

    def as_dict(self, max_idx: int = 50) -> dict:
        return {
            "n_inputs": self.n_inputs,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "kinds": dict(sorted(self.kinds.items())),
            "failing_idx": self.failing_idx[:max_idx],
            "first_failure": self.first_failure,
        }


def judge(
    results: Sequence[CallResult],
    expected: Sequence[Any],
    atol: float,
    entry_point: str,
    dataset: str,
) -> Verdict:
    kinds: dict[str, int] = {}
    failing: list[int] = []
    first = None
    n_pass = 0
    for i, (res, exp) in enumerate(zip(results, expected, strict=True)):
        if res.status == OK:
            if outputs_match(res.value, exp, atol, entry_point, dataset):
                n_pass += 1
                continue
            kind, why = "wrong", f"input #{i}: got {short_repr(res.value)} expected {short_repr(exp)}"
        else:
            kind, why = res.status, f"input #{i}: {res.status}: {res.error}"
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind != SKIPPED:
            failing.append(i)
        if first is None:
            first = why
    n = len(results)
    return Verdict(n, n_pass, n - n_pass - kinds.get(SKIPPED, 0), kinds, failing, first)
