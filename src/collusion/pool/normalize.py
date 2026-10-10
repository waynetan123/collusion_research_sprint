"""One formatting standard for every clean and buggy function (normalization step 3).

`normalize` parses the code, removes docstrings and other bare string
statements, strips type annotations, drops `typing` imports that become unused,
and regenerates source with `ast.unparse` (which also drops comments). Clean and
buggy versions then differ only in the bug, and style no longer reveals the
source dataset. `insert_docstring` adds the templated spec back afterwards.
"""

from __future__ import annotations

import ast

INDENT = "    "


def _is_str_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


class _Stripper(ast.NodeTransformer):
    """Remove bare string statements and annotations."""

    def generic_visit(self, node):
        super().generic_visit(node)
        for name in ("body", "orelse", "finalbody"):
            body = getattr(node, name, None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                kept = [s for s in body if not _is_str_expr(s)]
                if not kept and name == "body":
                    kept = [ast.Pass()]
                setattr(node, name, kept)
        return node

    def _strip_args(self, args: ast.arguments) -> None:
        for a in args.posonlyargs + args.args + args.kwonlyargs:
            a.annotation = None
        for a in (args.vararg, args.kwarg):
            if a is not None:
                a.annotation = None

    def visit_FunctionDef(self, node):
        node.returns = None
        self._strip_args(node.args)
        node.type_params = []
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._strip_args(node.args)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None  # bare declaration `x: int`
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def _used_names(tree: ast.AST) -> set[str]:
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            base = n
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                names.add(base.id)
    return names


def _prune_typing_imports(tree: ast.Module) -> None:
    used = _used_names(tree)

    def prune(body: list[ast.stmt]) -> list[ast.stmt]:
        out = []
        for s in body:
            if isinstance(s, ast.ImportFrom) and s.module == "typing":
                s.names = [a for a in s.names if (a.asname or a.name) in used]
                if not s.names:
                    continue
            elif isinstance(s, ast.Import):
                s.names = [
                    a for a in s.names if a.name != "typing" or (a.asname or a.name) in used
                ]
                if not s.names:
                    continue
            out.append(s)
        return out

    tree.body = prune(tree.body)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.body = prune(n.body) or [ast.Pass()]


def normalize(code: str) -> str:
    tree = ast.parse(code)
    tree = _Stripper().visit(tree)
    _prune_typing_imports(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"top-level function {name!r} not found")


def insert_docstring(code: str, entry_point: str, docstring: str) -> str:
    """Insert `docstring` as the entry point's docstring and re-unparse.

    `docstring` is plain text (no indentation); it is indented to the function
    body so `ast.unparse` renders a conventional multi-line docstring.
    """
    tree = ast.parse(code)
    fn = find_function(tree, entry_point)
    if fn.body and _is_str_expr(fn.body[0]):
        raise ValueError("function already has a docstring")
    lines = docstring.strip("\n").split("\n")
    body = "\n".join((INDENT + ln) if ln else "" for ln in lines)
    text = "\n" + body + "\n" + INDENT
    fn.body.insert(0, ast.Expr(ast.Constant(text)))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def strip_entry_docstring(code: str, entry_point: str) -> str:
    """Inverse of insert_docstring (used to compare final code to normalized code)."""
    tree = ast.parse(code)
    fn = find_function(tree, entry_point)
    if fn.body and _is_str_expr(fn.body[0]):
        fn.body = fn.body[1:] or [ast.Pass()]
    return ast.unparse(tree)
