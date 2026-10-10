"""Small pieces vendored from EvalPlus v0.3.1 (https://github.com/evalplus/evalplus).

EvalPlus is licensed under Apache-2.0; `evalplus/eval/__init__.py` additionally
carries the MIT notice of OpenAI's human-eval. We vendor instead of depending on
the `evalplus` package because it pulls in transformers, openai, datasets, etc.

Vendored, with minimal changes:
- `mbpp_deserialize_inputs` from evalplus/data/mbpp.py (verbatim logic).
- Special-oracle task lists from evalplus/eval/_special_oracle.py.
- Output comparison semantics from `unsafe_execute` in evalplus/eval/__init__.py,
  rewritten as a pure function; `np.allclose` is re-implemented in pure Python so
  sandboxes need not import numpy before forking.
- `reliability_guard` from evalplus/eval/utils.py (which derives from human-eval).
"""

from __future__ import annotations

import math
from typing import Any

# --------------------------------------------------------------------------- #
# Special oracles (evalplus/eval/_special_oracle.py)
# --------------------------------------------------------------------------- #

# For tasks whose outputs are not serializable, EvalPlus only checks "not None".
MBPP_OUTPUT_NOT_NONE_TASKS = ["check_str", "text_match_three", "text_starta_endb"]

# Tasks whose list outputs are compared as sets.
MBPP_OUTPUT_SET_EQ_TASKS = [
    "similar_elements",  # Mbpp/2
    "find_char_long",  # Mbpp/7
    "common_in_nested_lists",  # Mbpp/111
    "extract_singly",  # Mbpp/140
    "larg_nnum",  # Mbpp/232
    "intersection_array",  # Mbpp/249
    "find_dissimilar",  # Mbpp/579
    "Diff",  # Mbpp/769
]

# Tasks EvalPlus judges with a bespoke oracle rather than the canonical output.
# The pool excludes these (plus the NOT_NONE tasks): "differs from the clean
# output" would not reliably mean "buggy" for them.
SPECIAL_ORACLE_ENTRY_POINTS = {
    "mbpp": ["are_equivalent", "sum_div", "surface_Area", "digit_distance_nums"]
    + MBPP_OUTPUT_NOT_NONE_TASKS,
    "humaneval": ["find_zero"],
}


# --------------------------------------------------------------------------- #
# MBPP+ input deserialization (evalplus/data/mbpp.py)
# --------------------------------------------------------------------------- #


def mbpp_deserialize_inputs(task_id: str, inputs: list) -> list:
    task_id = int(task_id.split("/")[-1])
    if task_id in [
        2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429, 470, 560, 579,
        596, 616, 630, 726, 740, 744, 809,
    ]:
        modified_inputs = [[tuple(lst) for lst in inp] for inp in inputs]

    elif task_id in [63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780]:
        modified_inputs = [
            [[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs
        ]

    elif task_id in [75, 413, 444, 753]:
        modified_inputs = [
            [[tuple(lst) for lst in inp[0]]] + [inp[1]] for inp in inputs
        ]

    elif task_id == 106 or task_id == 750:
        modified_inputs = [[inp[0]] + [tuple(inp[1])] for inp in inputs]

    elif task_id == 115:
        modified_inputs = [
            [
                [
                    set(item) if isinstance(item, list) and len(item) else {}
                    for item in inp[0]
                ]
            ]
            for inp in inputs
        ]

    elif task_id == 124:
        modified_inputs = [(float(inp[0]), complex(inp[1])) for inp in inputs]

    elif task_id in [250, 405, 446, 617, 720, 763, 808]:
        modified_inputs = [[tuple(inp[0])] + [inp[1]] for inp in inputs]

    elif task_id in [259, 401, 445]:
        modified_inputs = [
            [[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs
        ]
        modified_inputs = [[tuple(lst) for lst in inp] for inp in modified_inputs]

    elif task_id == 278:
        modified_inputs = [
            [[tuple(item) if isinstance(item, list) else item for item in inp[0]]]
            for inp in inputs
        ]
        modified_inputs = [[tuple(lst) for lst in inp] for inp in modified_inputs]

    elif task_id == 307:
        modified_inputs = [[tuple(inp[0])] + [inp[1], inp[2]] for inp in inputs]

    elif task_id == 722:
        modified_inputs = [
            [{key: tuple(value) for key, value in inp[0].items()}] + inp[1:]
            for inp in inputs
        ]

    elif task_id == 252:
        modified_inputs = [[complex(inp[0])] for inp in inputs]

    elif task_id in [580, 615, 791]:

        def turn_all_list_into_tuple(inp):
            if isinstance(inp, list):
                return tuple([turn_all_list_into_tuple(item) for item in inp])
            return inp

        modified_inputs = [turn_all_list_into_tuple(inp) for inp in inputs]

    else:
        modified_inputs = inputs

    return modified_inputs


# --------------------------------------------------------------------------- #
# Output comparison (evalplus/eval/__init__.py: is_floats, unsafe_execute)
# --------------------------------------------------------------------------- #


def is_floats(x: Any) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    return False


def _isclose(a: Any, b: Any, rtol: float, atol: float) -> bool:
    # numpy.isclose semantics: |a - b| <= atol + rtol * |b|, NaN never close.
    try:
        if isinstance(a, complex) or isinstance(b, complex):
            return abs(a - b) <= atol + rtol * abs(b)
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * abs(b)


def _allclose(out: Any, exp: Any, rtol: float, atol: float) -> bool:
    """Pure-Python stand-in for np.allclose on scalars and (nested) sequences."""
    if isinstance(exp, (list, tuple)):
        if not isinstance(out, (list, tuple)) or len(out) != len(exp):
            return False
        return all(_allclose(o, e, rtol, atol) for o, e in zip(out, exp))
    return _isclose(out, exp, rtol, atol)


def outputs_match(out: Any, exp: Any, atol: float, entry_point: str, dataset: str) -> bool:
    """True iff `out` is accepted for expected `exp`, following EvalPlus semantics.

    Tasks with bespoke oracles are excluded from the pool, so only the
    set-equality special case is kept here.
    """
    try:
        exact_match = bool(out == exp)
    except Exception:
        exact_match = False

    if dataset == "mbpp" and entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
        try:
            exact_match = set(out) == set(exp)
        except Exception:
            return False

    if atol == 0 and is_floats(exp):
        atol = 1e-6  # EvalPlus enforces atol for float comparison
    if not exact_match and atol != 0:
        if type(out) is not type(exp):
            return False
        if isinstance(exp, (list, tuple)) and len(out) != len(exp):
            return False
        return _allclose(out, exp, rtol=1e-07, atol=atol)
    return exact_match


# --------------------------------------------------------------------------- #
# reliability_guard (evalplus/eval/utils.py, from openai/human-eval)
# --------------------------------------------------------------------------- #


def reliability_guard() -> None:
    """Disable destructive functions in a sandbox child process.

    WARNING: this is NOT a security sandbox. It prevents accidental damage by
    generated code, as in human-eval/EvalPlus. Memory limits are applied by the
    caller (RLIMIT_AS works on Linux; macOS does not enforce it).
    """
    import builtins
    import os
    import shutil
    import subprocess
    import sys

    os.environ["OMP_NUM_THREADS"] = "1"

    builtins.exit = None
    builtins.quit = None

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None
    for name in ("lchflags", "lchmod"):
        if hasattr(os, name):
            setattr(os, name, None)

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore

    builtins.help = None

    for mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[mod] = None  # type: ignore
