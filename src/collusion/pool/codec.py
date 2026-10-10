"""Type-preserving JSON encoding for test inputs and outputs.

Plain JSON loses tuples, sets, complex numbers, bytes, non-finite floats and
non-string dict keys, all of which occur in EvalPlus inputs. Containers and
special scalars are wrapped as {"__t__": <tag>, "v": <payload>}; every dict is
encoded as a tagged list of pairs so user dicts can never collide with a tag.
"""

from __future__ import annotations

import base64
import json
import math
import sys
from typing import Any

TAG = "__t__"

# Some EvalPlus inputs/outputs are integers with more than 4300 digits, which
# json (via int.__repr__) refuses to serialize under the default limit.
sys.set_int_max_str_digits(0)


def encode(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        if math.isfinite(x):
            return x
        return {TAG: "float", "v": repr(x)}  # 'nan', 'inf', '-inf'
    if isinstance(x, list):
        return [encode(i) for i in x]
    if isinstance(x, tuple):
        return {TAG: "tuple", "v": [encode(i) for i in x]}
    if isinstance(x, set):
        return {TAG: "set", "v": [encode(i) for i in x]}
    if isinstance(x, frozenset):
        return {TAG: "frozenset", "v": [encode(i) for i in x]}
    if isinstance(x, dict):
        return {TAG: "dict", "v": [[encode(k), encode(v)] for k, v in x.items()]}
    if isinstance(x, complex):
        return {TAG: "complex", "v": [encode(x.real), encode(x.imag)]}
    if isinstance(x, bytes):
        return {TAG: "bytes", "v": base64.b64encode(x).decode("ascii")}
    raise TypeError(f"cannot encode value of type {type(x).__name__}")


def decode(x: Any) -> Any:
    if isinstance(x, list):
        return [decode(i) for i in x]
    if not isinstance(x, dict):
        return x
    tag, v = x[TAG], x["v"]
    if tag == "float":
        return float(v)
    if tag == "tuple":
        return tuple(decode(i) for i in v)
    if tag == "set":
        return {decode(i) for i in v}
    if tag == "frozenset":
        return frozenset(decode(i) for i in v)
    if tag == "dict":
        return {decode(k): decode(val) for k, val in v}
    if tag == "complex":
        return complex(decode(v[0]), decode(v[1]))
    if tag == "bytes":
        return base64.b64decode(v)
    raise ValueError(f"unknown tag {tag!r}")


def dumps(x: Any) -> str:
    # sort_keys keeps output byte-stable; payload order inside "v" is preserved.
    return json.dumps(encode(x), sort_keys=True, ensure_ascii=False)


def loads(s: str) -> Any:
    return decode(json.loads(s))


def same(a: Any, b: Any) -> bool:
    """Strict structural equality that also distinguishes types and NaN-equality."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b
