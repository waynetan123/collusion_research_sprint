import ast
import math

import pytest

from collusion.pool import codec
from collusion.pool._evalplus_vendor import outputs_match
from collusion.pool.astdiff import diff_pair
from collusion.pool.mutate import BUG_TYPE, DOC_OPERATORS, mutants
from collusion.pool.normalize import insert_docstring, normalize, strip_entry_docstring
from collusion.pool.render import docstring_problems, render_docstring, render_patch
from collusion.pool.sandbox import CRASH, ERROR, OK, SKIPPED, TIMEOUT, judge, run_calls

# --------------------------------------------------------------------------- #
# codec
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    [
        [1, "a", None, True, 2.5],
        (1, (2, 3), [4]),
        {1, 2, 3},
        frozenset({"x"}),
        {1: "a", "b": (2,), (3, 4): [5]},
        complex(1, -2),
        b"\x00\xff",
        [float("inf"), -math.inf],
        [{"__t__": "tuple", "v": []}],  # user data that looks like a tag
        10**5000,
    ],
)
def test_codec_roundtrip(value):
    assert codec.same(codec.loads(codec.dumps(value)), value)


def test_pool_io_roundtrip(tmp_path):
    import json

    from collusion.pool.io import load_pool, load_tests, write_pool

    big = 10**5000  # beyond Python's default int-string limit
    recs = [{"id": "X/1", "spec": "s", "inputs": codec.encode([[big], [(1, 2)]]),
             "expected": codec.encode([1, {3}]), "timeouts": [1.0, 1.0]}]
    write_pool(recs, tmp_path / "p.jsonl", tmp_path / "t.jsonl.gz")
    pool = load_pool(tmp_path / "p.jsonl")
    assert pool == [{"id": "X/1", "spec": "s"}]
    json.loads((tmp_path / "p.jsonl").read_text())  # main file stays plain-json readable
    t = load_tests(tmp_path / "t.jsonl.gz")["X/1"]
    assert codec.same(t["inputs"], [[big], [(1, 2)]]) and t["expected"] == [1, {3}]


def test_codec_nan():
    assert math.isnan(codec.loads(codec.dumps(float("nan"))))


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #

SRC = '''
from typing import List, Optional
import math

def f(xs: List[int], k: Optional[int] = None) -> int:
    """Doc."""
    # comment
    total: int = 0
    seen: set
    "stray string"
    for x in xs:
        total += x
    return total
'''


def test_normalize_strips_annotations_docstrings_comments():
    out = normalize(SRC)
    assert "List" not in out and "Optional" not in out and "typing" not in out
    assert "->" not in out and ":" not in out.splitlines()[2].split("(")[1].split(")")[0]
    assert '"""' not in out and "comment" not in out and "stray" not in out
    assert "import math" in out
    assert "total = 0" in out


def test_normalize_idempotent():
    out = normalize(SRC)
    assert normalize(out) == out


def test_normalize_keeps_used_typing():
    src = "from typing import List\n\ndef f(x):\n    return List[int]\n"
    assert "from typing import List" in normalize(src)


def test_insert_and_strip_docstring():
    code = normalize("def f(a):\n    return a + 1\n")
    doc = "Add one.\n\nArgs:\n    a (int): value"
    with_doc = insert_docstring(code, "f", doc)
    fn = ast.parse(with_doc).body[0]
    assert ast.get_docstring(fn) == doc
    assert strip_entry_docstring(with_doc, "f") == code


# --------------------------------------------------------------------------- #
# mutate
# --------------------------------------------------------------------------- #


def _ops(code):
    return {(m.operator, m.detail) for m in mutants(normalize(code))}


def test_each_operator_fires():
    code = """
def f(xs, n):
    if not xs:
        return 0
    s = 0
    for i in range(1, n):
        if xs[i] < n and i > 0:
            s = s + xs[i]
    return s
"""
    ops = {op for op, _ in _ops(code)}
    assert DOC_OPERATORS <= ops
    assert "constant_shift" in ops


def test_function_swap_and_bug_types():
    ops = _ops("def f(xs):\n    return max(xs, key=len).lower()\n")
    assert ("function_swap", "max->min") in ops
    assert ("function_swap", ".lower->.upper") in ops
    assert BUG_TYPE["guard_delete"] == "missing logic"
    assert BUG_TYPE["boundary_shift"] == "value misuse"
    assert BUG_TYPE["swap_operator"] == "operator misuse"


def test_mutants_are_distinct_single_edits():
    code = normalize("def f(a, b):\n    if a < b:\n        return a - b\n    return a + b\n")
    ms = mutants(code)
    assert len({m.code for m in ms}) == len(ms)
    assert all(m.code != code for m in ms)
    assert all(diff_pair(code, m.code).ok for m in ms)


def test_imports_not_mutated():
    ms = mutants("import math\n\ndef f(x):\n    return math.floor(x)\n")
    assert all("import math" in m.code for m in ms)


# --------------------------------------------------------------------------- #
# astdiff
# --------------------------------------------------------------------------- #


def test_single_edit_accepted_with_location():
    d = diff_pair("def f(a, b):\n    x = 1\n    return a < b\n", "def f(a, b):\n    x = 1\n    return a <= b\n")
    assert d.ok and d.clean_lines == (3, 3) and d.buggy_lines == (3, 3)


def test_two_sites_same_statement_accepted():
    d = diff_pair("def f(a, b, i, j):\n    return a[i] < b[j]\n", "def f(a, b, i, j):\n    return a[j] < b[i]\n")
    assert d.ok and d.n_regions == 2


def test_two_statements_rejected():
    d = diff_pair(
        "def f(a):\n    x = a + 1\n    y = a + 2\n    return x + y\n",
        "def f(a):\n    x = a - 1\n    y = a - 2\n    return x + y\n",
    )
    assert not d.ok


def test_deletion_located():
    d = diff_pair("def f(a):\n    if a:\n        return 1\n    return 2\n", "def f(a):\n    return 2\n")
    assert d.ok and d.clean_lines == (2, 3) and d.buggy_lines == (2, 2)


def test_identical_rejected():
    assert not diff_pair("def f():\n    return 1\n", "def f():\n    return 1\n").ok


# --------------------------------------------------------------------------- #
# sandbox
# --------------------------------------------------------------------------- #

SANDBOX_CODE = """
def f(x, lst):
    lst.append(1)
    if x == 1:
        while True:
            pass
    if x == 2:
        raise ValueError("boom")
    if x == 3:
        import os
        os._exit(3)
    return x, len(lst)
"""


@pytest.mark.parametrize("mode", ["call", "batch"])
def test_sandbox_statuses(mode):
    inputs = [[0, []], [1, []], [2, []], [3, []], [4, []]]
    res = run_calls(SANDBOX_CODE, "f", inputs, 0.3, mode=mode)
    assert [r.status for r in res] == [OK, TIMEOUT, ERROR, CRASH, OK]
    assert res[0].value == (0, 1) and res[4].value == (4, 1)  # deep-copied per call
    assert inputs[0] == [0, []]  # caller's inputs untouched


def test_sandbox_max_timeouts_skips_rest():
    res = run_calls("def f(x):\n    while True: pass\n", "f", [[i] for i in range(4)], 0.2, max_timeouts=2)
    assert [r.status for r in res] == [TIMEOUT, TIMEOUT, SKIPPED, SKIPPED]


def test_sandbox_syntax_error():
    res = run_calls("def f(:\n", "f", [[1]], 1)
    assert res[0].status == ERROR and "SyntaxError" in res[0].error


def test_sandbox_guard_blocks_destructive_calls(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("x")
    code = f"import os\ndef f():\n    os.remove({str(target)!r})\n"
    res = run_calls(code, "f", [[]], 1)
    assert res[0].status == ERROR and target.exists()


def test_judge_counts():
    from collusion.pool.sandbox import CallResult

    res = [CallResult(OK, 1), CallResult(OK, 2), CallResult(ERROR, error="E"), CallResult(SKIPPED)]
    v = judge(res, [1, 3, 3, 4], 0, "f", "humaneval")
    assert v.n_pass == 1 and v.n_fail == 2 and v.failing_idx == [1, 2]
    assert v.kinds == {"wrong": 1, "error": 1, "skipped": 1}


# --------------------------------------------------------------------------- #
# comparison (vendored EvalPlus semantics)
# --------------------------------------------------------------------------- #


def test_outputs_match_float_tolerance():
    assert outputs_match(0.1 + 0.2, 0.3, 0, "f", "humaneval")  # atol forced to 1e-6 for floats
    assert not outputs_match(0.31, 0.3, 0, "f", "humaneval")
    assert outputs_match([1.0, 2.0000001], [1.0, 2.0], 0, "f", "mbpp")
    assert not outputs_match(1, 1.0000001, 0, "f", "mbpp")  # int vs float: type must match for allclose


def test_outputs_match_set_eq_task():
    assert outputs_match((5, 4), (4, 5), 0, "similar_elements", "mbpp")
    assert not outputs_match((5, 4), (4, 5), 0, "other", "mbpp")


def test_outputs_match_nan_and_types():
    assert not outputs_match(float("nan"), float("nan"), 0, "f", "mbpp")
    assert not outputs_match("1", 1, 0, "f", "mbpp")


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #


def test_render_docstring_template():
    fields = {
        "summary": "Return the sum of the list.",
        "details": ["An empty list gives 0."],
        "params": [{"name": "xs", "type": "list of int", "description": "Numbers."}],
        "returns": {"type": "int", "description": "The sum."},
    }
    doc = render_docstring(fields, "total", [{"args": "[1, 2]", "expected": "3"}])
    assert doc.splitlines()[0] == "Return the sum of the list."
    assert "Args:\n    xs (list of int): Numbers." in doc
    assert "    >>> total([1, 2])\n    3" in doc
    assert docstring_problems(doc) == []
    assert docstring_problems("a\\nb") == ["contains a backslash"]


def test_render_patch_has_no_free_text_slot():
    assert render_patch("X", "def f():\n    pass") == "Patch X:\n```python\ndef f():\n    pass\n```"
