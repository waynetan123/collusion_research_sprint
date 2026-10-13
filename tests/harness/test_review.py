import asyncio
import json

import pytest

from collusion.harness.backend import CallContext, Reply
from collusion.harness.config import Limits
from collusion.harness.review import ParseError, clean_fix, fix_ok, parse_review, review_schema, run_review

GOOD_FIX = "def f(x):\n    return x + 1"


class Scripted:
    """Backend returning queued texts; records calls."""

    def __init__(self, *texts, finish="stop"):
        self.texts = list(texts)
        self.calls = []
        self.finish = finish

    async def chat(self, messages, *, max_tokens, seed, schema, ctx):
        self.calls.append({"kind": ctx.kind, "seed": seed, "messages": messages, "schema": schema})
        text = self.texts.pop(0)
        finish = self.finish if isinstance(self.finish, str) else self.finish.pop(0)
        return Reply(text, finish, 10, 5, None, 0.0)


def rv(v, fix="", reason="r"):
    return {"verdict": v, "reason": reason, "fix": fix}


def run(backend, labels=("X", "Y")):
    ctx = CallContext("s", "arm2", 1, "A", "review")
    return asyncio.run(run_review(backend, [{"role": "user", "content": "p"}], list(labels),
                                  {lb: "f" for lb in labels}, ctx, Limits(), ("s", 1, "A")))


def test_schema_requires_each_label_once():
    s = review_schema(["X", "Y"], 600)
    assert s["required"] == ["X", "Y"] and s["additionalProperties"] is False
    assert s["properties"]["X"]["properties"]["verdict"]["enum"] == ["APPROVE", "REJECT"]


def test_schema_puts_reason_before_verdict():
    """Guided decoding generates properties in schema order: reason first, then the verdict."""
    one = review_schema(["X", "Y"], 1000)["properties"]["X"]
    assert list(one["properties"]) == ["reason", "verdict", "fix"] == one["required"]
    assert one["properties"]["reason"]["maxLength"] == 1000


def test_parse_review_errors():
    with pytest.raises(ParseError):
        parse_review("not json", ["X", "Y"], 600)
    with pytest.raises(ParseError):
        parse_review(json.dumps({"X": rv("APPROVE")}), ["X", "Y"], 600)
    with pytest.raises(ParseError):
        parse_review(json.dumps({"X": rv("MAYBE"), "Y": rv("APPROVE")}), ["X", "Y"], 600)
    got = parse_review(json.dumps({"X": rv("APPROVE", reason="a" * 900), "Y": rv("REJECT", GOOD_FIX)}), ["X", "Y"], 600)
    assert len(got["X"]["reason"]) == 600 and got["Y"]["fix"] == GOOD_FIX


def test_fix_ok_and_fences():
    assert fix_ok(GOOD_FIX, "f")
    assert fix_ok("```python\n" + GOOD_FIX + "\n```", "f")
    assert clean_fix("```python\n" + GOOD_FIX + "\n```") == GOOD_FIX
    assert not fix_ok("", "f")
    assert not fix_ok("def g(x):\n    return x", "f")
    assert not fix_ok("def f(x) return", "f")
    assert fix_ok("import math\n\ndef f(x):\n    return math.floor(x)", "f")  # imports allowed


def test_clean_reply_needs_one_call():
    b = Scripted(json.dumps({"X": rv("APPROVE"), "Y": rv("REJECT", GOOD_FIX)}))
    out = run(b)
    assert not out.verdict_imputed and out.n_resamples == 0 and len(b.calls) == 1
    assert out.reviews["Y"]["fix"] == GOOD_FIX and not out.reviews["Y"]["fix_missing"]


def test_resample_with_new_seed_then_success():
    b = Scripted("{bad", '{"X": {}}', json.dumps({"X": rv("APPROVE"), "Y": rv("APPROVE")}))
    out = run(b)
    assert out.n_resamples == 2 and not out.verdict_imputed
    assert len({c["seed"] for c in b.calls}) == 3
    assert [c.ok for c in out.calls] == [False, False, True]


def test_truncated_reply_is_resampled():
    good = json.dumps({"X": rv("APPROVE"), "Y": rv("APPROVE")})
    b = Scripted(good, good, finish=["length", "stop"])
    out = run(b)
    assert out.n_resamples == 1 and out.calls[0].error == "cut off at max_tokens"


def test_all_attempts_fail_imputes_reject():
    b = Scripted(*["{bad"] * 4)
    out = run(b)
    assert out.verdict_imputed and len(b.calls) == 4  # 1 + 3 re-samples
    assert all(r["verdict"] == "REJECT" for r in out.reviews.values())


def test_missing_fix_reprompted_once_and_recovered():
    b = Scripted(json.dumps({"X": rv("REJECT", ""), "Y": rv("REJECT", "oops")}),
                 json.dumps({"X": {"fix": GOOD_FIX}, "Y": {"fix": "still bad"}}))
    out = run(b)
    assert [c["kind"] for c in b.calls] == ["review", "fix"]
    assert b.calls[1]["schema"]["required"] == ["X", "Y"]
    # The re-prompt continues the review exchange.
    assert [m["role"] for m in b.calls[1]["messages"]] == ["user", "assistant", "user"]
    assert out.reviews["X"]["fix"] == GOOD_FIX and not out.reviews["X"]["fix_missing"]
    assert out.reviews["Y"]["fix_missing"] and out.reviews["Y"]["verdict"] == "REJECT"  # verdict still counts
    assert all(out.reviews[lb]["fix_reprompted"] for lb in "XY")


def test_approval_fix_is_dropped():
    out = run(Scripted(json.dumps({"X": rv("APPROVE", GOOD_FIX), "Y": rv("APPROVE")})))
    assert out.reviews["X"]["fix"] == ""


def test_fix_with_invalid_escape_is_usable_without_warnings(recwarn):
    from collusion.harness.review import fix_ok

    fix = 'import re\n\ndef f(s):\n    return re.findall("\\w+", s)\n'
    assert fix_ok(fix, "f")
    assert not [w for w in recwarn if issubclass(w.category, SyntaxWarning)]
