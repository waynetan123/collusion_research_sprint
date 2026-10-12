"""Fixed templates for specs (docstrings) and patch descriptions.

Every docstring is rendered from structured fields with one template, so specs
from both datasets look alike and carry no free-form channel. Patch
descriptions shown to agents are likewise fixed templates around code that the
harness (not an agent) supplies, so agents cannot pass messages through them.
"""

from __future__ import annotations

import textwrap

WIDTH = 76


def _wrap(text: str, indent: str = "", hanging: str | None = None) -> list[str]:
    hanging = indent if hanging is None else hanging
    return textwrap.wrap(
        " ".join(text.split()), WIDTH, initial_indent=indent, subsequent_indent=hanging,
        break_long_words=False, break_on_hyphens=False,
    ) or [indent.rstrip()]


def render_docstring(fields: dict, entry_point: str, examples: list[dict]) -> str:
    """fields: {summary, details[], params[{name,type,description}], returns{type,description}}.
    examples: [{args, expected}] already verified by execution."""
    lines = _wrap(fields["summary"])
    if fields.get("details"):
        lines.append("")
        for d in fields["details"]:
            lines += _wrap(d, "- ", "  ")
    if fields.get("params"):
        lines += ["", "Args:"]
        for p in fields["params"]:
            lines += _wrap(f"{p['name']} ({p['type']}): {p['description']}", "    ", "        ")
    lines += ["", "Returns:"]
    lines += _wrap(f"{fields['returns']['type']}: {fields['returns']['description']}", "    ", "        ")
    if examples:
        lines += ["", "Examples:"]
        for ex in examples:
            lines.append(f"    >>> {entry_point}({ex['args']})")
            lines.append(f"    {ex['expected']}")
    return "\n".join(lines)


def docstring_problems(text: str) -> list[str]:
    """Characters that would make ast.unparse render the docstring badly."""
    problems = []
    if "\\" in text:
        problems.append("contains a backslash")
    if '"""' in text or "'''" in text:
        problems.append("contains a triple quote")
    if any(ord(c) < 32 and c != "\n" for c in text):
        problems.append("contains a control character")
    return problems


def render_patch(label: str, code: str, author: str | None = None) -> str:
    """The only way patch code is presented to an agent.

    `author` is a harness-chosen tag (e.g. "Agent B", "external contributor"),
    never agent-supplied text.
    """
    head = f"Patch {label}" if author is None else f"Patch {label} ({author})"
    return f"{head}:\n```python\n{code}\n```"
