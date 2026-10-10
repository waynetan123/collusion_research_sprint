"""Check that a clean/buggy pair differs by one localized edit (normalization step 4).

The two syntax trees are walked in parallel and every minimal differing region
is collected:

- a node whose type differs, or a primitive field that differs, is one region;
- a statement list whose lengths differ is one region if the change is a single
  contiguous run of inserted/deleted statements (common prefix + common suffix
  cover the shorter list); otherwise the pair is rejected;
- a non-statement list whose lengths differ (e.g. call arguments) is one region.

A pair passes if all regions sit inside the same innermost statement (so
`a[i] < b[j]` -> `a[j] < b[i]` counts as one edit, but edits on two different
lines do not) or there is exactly one region.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class Region:
    clean: ast.AST | list | None
    buggy: ast.AST | list | None
    clean_stmt: ast.stmt | None  # innermost enclosing statement in the clean tree
    buggy_stmt: ast.stmt | None
    clean_anchor: ast.AST | None = None  # nearest positioned node, for line spans
    buggy_anchor: ast.AST | None = None


@dataclass
class DiffResult:
    ok: bool
    n_regions: int
    reason: str
    clean_lines: tuple[int, int] | None = None
    buggy_lines: tuple[int, int] | None = None


class _Reject(Exception):
    pass


def _span(x) -> tuple[int, int] | None:
    nodes = x if isinstance(x, list) else [x]
    nodes = [n for n in nodes if hasattr(n, "lineno")]
    if not nodes:
        return None
    return min(n.lineno for n in nodes), max(n.end_lineno for n in nodes)


def _diff(a, b, sa, sb, pa, pb, out: list[Region]) -> None:
    """pa/pb: nearest ancestor-or-self with a line number (fallback for spans)."""
    if isinstance(a, ast.AST) and isinstance(b, ast.AST):
        if type(a) is not type(b):
            out.append(Region(a, b, sa, sb, pa, pb))
            return
        na = a if isinstance(a, ast.stmt) else sa
        nb = b if isinstance(b, ast.stmt) else sb
        qa = a if hasattr(a, "lineno") else pa
        qb = b if hasattr(b, "lineno") else pb
        for (fname, va), (_, vb) in zip(ast.iter_fields(a), ast.iter_fields(b)):
            if fname in ("ctx", "type_comment"):
                continue
            _diff(va, vb, na, nb, qa, qb, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b):
            for x, y in zip(a, b):
                _diff(x, y, sa, sb, pa, pb, out)
            return
        is_stmts = any(isinstance(x, ast.stmt) for x in a + b)
        if not is_stmts:
            out.append(Region(a, b, sa, sb, pa, pb))
            return
        da, db = [ast.dump(x) for x in a], [ast.dump(x) for x in b]
        p = 0
        while p < min(len(a), len(b)) and da[p] == db[p]:
            p += 1
        s = 0
        while s < min(len(a), len(b)) - p and da[-1 - s] == db[-1 - s]:
            s += 1
        if p + s < min(len(a), len(b)):
            raise _Reject("statement list changed in more than one contiguous run")
        ra, rb = a[p : len(a) - s], b[p : len(b) - s]
        # Anchor an empty side to the statement now at the gap, else the one before
        # it, else the enclosing node.
        def anchor(lst, q):
            if p < len(lst):
                return lst[p]
            return lst[p - 1] if p > 0 else q

        out.append(Region(ra, rb, None, None, anchor(a, pa), anchor(b, pb)))
        return
    if a != b:
        out.append(Region(pa, pb, sa, sb, pa, pb))


def diff_pair(clean_code: str, buggy_code: str) -> DiffResult:
    ta, tb = ast.parse(clean_code), ast.parse(buggy_code)
    regions: list[Region] = []
    try:
        _diff(ta, tb, None, None, ta, tb, regions)
    except _Reject as e:
        return DiffResult(False, -1, str(e))
    if not regions:
        return DiffResult(False, 0, "identical")
    if len(regions) > 1:
        stmts = {(id(r.clean_stmt), id(r.buggy_stmt)) for r in regions}
        if len(stmts) != 1 or regions[0].clean_stmt is None:
            return DiffResult(False, len(regions), "edits span more than one statement")

    def lines(side: str):
        spans = []
        for r in regions:
            sp = _span(getattr(r, side)) or _span(getattr(r, side + "_anchor"))
            if sp:
                spans.append(sp)
        if not spans:
            return None
        return min(s[0] for s in spans), max(s[1] for s in spans)

    return DiffResult(True, len(regions), "single localized edit", lines("clean"), lines("buggy"))
