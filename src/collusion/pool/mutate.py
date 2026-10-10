"""AST mutator for MBPP+ buggy versions.

Every mutant is a single-site edit from one of the five operator families named
in the design doc, each mapped to a HumanEvalFix bug category:

| operator           | example                               | bug_type        |
|--------------------|---------------------------------------|-----------------|
| boundary_shift     | `<` <-> `<=`, `>` <-> `>=`            | value misuse    |
| range_endpoint     | `range(a, b)` -> `range(a, b + 1)`    | value misuse    |
| guard_delete       | drop `if c: return/raise/break/...`   | missing logic   |
| negate_condition   | `if c` -> `if not c`                  | operator misuse |
| swap_operator      | `+` <-> `-`, `and` <-> `or`, `==` ... | operator misuse |

Two extension families exist for functions that have no valid mutant from the
five above (many MBPP+ solutions are builtin one-liners such as
`max(lists, key=sum)`). The build only falls back to them in that case:

| operator           | example                               | bug_type        |
|--------------------|---------------------------------------|-----------------|
| constant_shift     | `x[0]` -> `x[1]`, `reverse=True` -> `False` | value misuse |
| function_swap      | `min` <-> `max`, `lower` <-> `upper`  | function misuse |

Import statements are never mutated. Mutants identical to the original or to
each other are dropped.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Callable, Iterator

BUG_TYPE = {
    "boundary_shift": "value misuse",
    "range_endpoint": "value misuse",
    "guard_delete": "missing logic",
    "negate_condition": "operator misuse",
    "swap_operator": "operator misuse",
    "constant_shift": "value misuse",
    "function_swap": "function misuse",
}
DOC_OPERATORS = frozenset(
    {"boundary_shift", "range_endpoint", "guard_delete", "negate_condition", "swap_operator"}
)

_BUILTIN_SWAP = {"min": "max", "max": "min", "any": "all", "all": "any"}
_METHOD_SWAP = {
    "lower": "upper", "upper": "lower",
    "startswith": "endswith", "endswith": "startswith",
    "lstrip": "rstrip", "rstrip": "lstrip",
    "find": "rfind", "rfind": "find",
    "isupper": "islower", "islower": "isupper",
    "nlargest": "nsmallest", "nsmallest": "nlargest",
    "floor": "ceil", "ceil": "floor",
    "intersection": "union", "union": "intersection",
    "ljust": "rjust", "rjust": "ljust",
    "keys": "values", "values": "keys",
}

_BOUNDARY = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
_CMP_SWAP = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE,
    ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
_BIN_SWAP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Div: ast.Mult,
    ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult,
    ast.LShift: ast.RShift, ast.RShift: ast.LShift,
    ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd,
}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}
_GUARD_EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


@dataclass(frozen=True)
class Mutant:
    code: str
    operator: str
    bug_type: str
    detail: str  # human-readable description of the edit, for logs only


def _nodes(tree: ast.AST) -> list[ast.AST]:
    """Stable pre-order list of mutable nodes (imports and their children excluded)."""
    out: list[ast.AST] = []

    def walk(n: ast.AST) -> None:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return
        out.append(n)
        for child in ast.iter_child_nodes(n):
            walk(child)

    walk(tree)
    return out


def _plus_minus_one(expr: ast.expr, delta: int) -> ast.expr:
    if isinstance(expr, ast.Constant) and type(expr.value) is int:
        return ast.Constant(expr.value + delta)
    op = ast.Add() if delta > 0 else ast.Sub()
    return ast.BinOp(left=copy.deepcopy(expr), op=op, right=ast.Constant(1))


def _negate(expr: ast.expr) -> ast.expr:
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return expr.operand
    return ast.UnaryOp(op=ast.Not(), operand=expr)


# Each site generator yields (operator, detail, apply) where apply(node, parent_map)
# mutates the copied tree in place. Sites are addressed by pre-order index so the
# same site can be located in a fresh deep copy.
Apply = Callable[[ast.AST, dict], None]


def _sites(node: ast.AST) -> Iterator[tuple[str, str, Apply]]:
    if isinstance(node, ast.Compare):
        for k, op in enumerate(node.ops):
            if type(op) in _BOUNDARY:
                new = _BOUNDARY[type(op)]

                def f(n, _, k=k, new=new):
                    n.ops[k] = new()

                yield "boundary_shift", f"{type(op).__name__}->{new.__name__}", f
            if type(op) in _CMP_SWAP:
                new = _CMP_SWAP[type(op)]

                def f(n, _, k=k, new=new):
                    n.ops[k] = new()

                yield "swap_operator", f"{type(op).__name__}->{new.__name__}", f

    elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _BIN_SWAP:
        new = _BIN_SWAP[type(node.op)]

        def f(n, _, new=new):
            n.op = new()

        yield "swap_operator", f"{type(node.op).__name__}->{new.__name__}", f

    elif isinstance(node, ast.BoolOp):
        new = _BOOL_SWAP[type(node.op)]

        def f(n, _, new=new):
            n.op = new()

        yield "swap_operator", f"{type(node.op).__name__}->{new.__name__}", f

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and 1 <= len(node.args) <= 3
        and not node.keywords
    ):
        positions = [0] if len(node.args) == 1 else [0, 1]  # (stop) or (start, stop)
        for k in positions:
            for delta in (+1, -1):
                def f(n, _, k=k, delta=delta):
                    n.args[k] = _plus_minus_one(n.args[k], delta)

                which = "stop" if len(node.args) == 1 or k == 1 else "start"
                yield "range_endpoint", f"range {which}{delta:+d}", f

    if isinstance(node, (ast.If, ast.While, ast.IfExp)):
        def f(n, _):
            n.test = _negate(n.test)

        yield "negate_condition", f"negate {type(node).__name__}.test", f

    if isinstance(node, ast.comprehension):
        for k in range(len(node.ifs)):
            def f(n, _, k=k):
                n.ifs[k] = _negate(n.ifs[k])

            yield "negate_condition", "negate comprehension if", f

    if (
        isinstance(node, ast.If)
        and not node.orelse
        and len(node.body) == 1
        and isinstance(node.body[0], _GUARD_EXITS)
    ):
        def f(n, parents):
            parent, field = parents[id(n)]
            body = getattr(parent, field)
            body.remove(n)
            if not body and field == "body":
                body.append(ast.Pass())

        yield "guard_delete", f"delete guard -> {type(node.body[0]).__name__}", f

    # ---- extension families (fallback only) ---- #
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        def f(n, _):
            n.value = not n.value

        yield "constant_shift", f"{node.value}->{not node.value}", f
    elif isinstance(node, ast.Constant) and type(node.value) is int:
        for delta in (+1, -1):
            def f(n, _, delta=delta):
                n.value = n.value + delta

            yield "constant_shift", f"{node.value}{delta:+d}", f

    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _BUILTIN_SWAP:
            new = _BUILTIN_SWAP[fn.id]

            def f(n, _, new=new):
                n.func.id = new

            yield "function_swap", f"{fn.id}->{new}", f
        elif isinstance(fn, ast.Attribute) and fn.attr in _METHOD_SWAP:
            new = _METHOD_SWAP[fn.attr]

            def f(n, _, new=new):
                n.func.attr = new

            yield "function_swap", f".{fn.attr}->.{new}", f


def _parent_map(tree: ast.AST) -> dict[int, tuple[ast.AST, str]]:
    parents = {}
    for p in ast.walk(tree):
        for field, value in ast.iter_fields(p):
            if isinstance(value, list):
                for c in value:
                    if isinstance(c, ast.AST):
                        parents[id(c)] = (p, field)
    return parents


def mutants(code: str) -> list[Mutant]:
    """All distinct single-site mutants of normalized `code` (all families).

    Callers filter on `operator in DOC_OPERATORS` to apply the fallback rule.
    """
    original = ast.unparse(ast.parse(code))
    tree = ast.parse(original)
    seen = {original}
    out: list[Mutant] = []
    for idx, node in enumerate(_nodes(tree)):
        for j, (operator, detail, _) in enumerate(_sites(node)):
            mtree = copy.deepcopy(tree)
            mnode = _nodes(mtree)[idx]
            _, _, apply = list(_sites(mnode))[j]
            apply(mnode, _parent_map(mtree))
            ast.fix_missing_locations(mtree)
            try:
                mcode = ast.unparse(mtree)
                compile(mcode, "<mutant>", "exec")
            except (SyntaxError, ValueError):
                continue
            if mcode in seen:
                continue
            seen.add(mcode)
            out.append(Mutant(mcode, operator, BUG_TYPE[operator], detail))
    return out
