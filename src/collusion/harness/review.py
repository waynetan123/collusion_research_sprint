"""Structured reviews: schema, parsing, re-sampling and fix enforcement.

- Review replies follow a JSON schema keyed by patch label, so guided decoding
  yields exactly one review per patch, each written reason first, then verdict,
  then fix. Verdicts go to scoring and to the
  author's ledger as APPROVE / REJECT only.
- An unparseable reply (bad JSON, missing label, bad verdict, or cut off at the
  token cap) is re-sampled with a new seed, up to `max_resamples` times; every
  attempt is logged. If all attempts fail the verdicts are imputed as REJECT
  with `verdict_imputed = true`; analysis excludes these rows.
- A REJECT whose fix is empty or does not parse into `def <entry_point>` gets
  one re-prompt (a continuation asking only for the missing fixes). If it is
  still missing, `fix_missing = true` and the verdict still counts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field

from .backend import Backend, CallContext, Reply

VERDICTS = ("APPROVE", "REJECT")


def review_schema(labels: list[str], reason_max_chars: int) -> dict:
    one = {
        "type": "object",
        # Property order is generation order under guided decoding: the reason comes
        # first so the verdict follows the reasoning instead of being rationalised after it.
        "properties": {
            "reason": {"type": "string", "maxLength": reason_max_chars},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "fix": {"type": "string"},
        },
        "required": ["reason", "verdict", "fix"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {lb: one for lb in labels}, "required": list(labels),
            "additionalProperties": False}


def fix_schema(labels: list[str]) -> dict:
    one = {"type": "object", "properties": {"fix": {"type": "string"}}, "required": ["fix"],
           "additionalProperties": False}
    return {"type": "object", "properties": {lb: one for lb in labels}, "required": list(labels),
            "additionalProperties": False}


class ParseError(ValueError):
    pass


def parse_review(text: str, labels: list[str], reason_max_chars: int) -> dict[str, dict]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ParseError("reply is not a JSON object")
    out = {}
    for lb in labels:
        r = obj.get(lb)
        if not isinstance(r, dict):
            raise ParseError(f"missing review for {lb}")
        v = r.get("verdict")
        if v not in VERDICTS:
            raise ParseError(f"bad verdict for {lb}: {v!r}")
        reason, fix = r.get("reason", ""), r.get("fix", "")
        if not isinstance(reason, str) or not isinstance(fix, str):
            raise ParseError(f"non-string reason/fix for {lb}")
        out[lb] = {"verdict": v, "reason": reason[:reason_max_chars], "fix": fix}
    return out


_FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def clean_fix(fix: str) -> str:
    """Strip a surrounding markdown code fence, if any."""
    m = _FENCE.match(fix)
    return (m.group(1) if m else fix).strip("\n")


def fix_ok(fix: str, entry_point: str) -> bool:
    """A usable fix parses and defines `entry_point` at top level."""
    code = clean_fix(fix)
    if not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point for n in tree.body)


def derive_seed(*parts) -> int:
    """Sampling seed for one request, derived from its identity (recorded per request)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def prompt_sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reply_event_fields(r: Reply) -> dict:
    """Per-call fields of a `call` event. OpenRouter extras are added only when reported,
    so vLLM and fake-backend events are unchanged."""
    out = {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
           "cached_tokens": r.cached_tokens, "finish_reason": r.finish_reason, "latency": round(r.latency, 3),
           "text": r.text}
    for k in ("reasoning", "cost", "reasoning_tokens", "provider"):
        v = getattr(r, k)
        if v is not None and v != "":
            out[k] = v
    return out


@dataclass
class CallLog:
    kind: str
    attempt: int
    seed: int
    prompt_sha256: str
    reply: Reply
    ok: bool
    error: str | None = None

    def as_event(self) -> dict:
        return {"kind": self.kind, "attempt": self.attempt, "seed": self.seed, "prompt_sha256": self.prompt_sha256,
                "ok": self.ok, "error": self.error, **reply_event_fields(self.reply)}


@dataclass
class ReviewOutcome:
    reviews: dict[str, dict]  # label -> {verdict, reason, fix, fix_reprompted, fix_missing}
    reply_text: str  # accepted reply (the assistant turn continued by the notes call); "" if imputed
    verdict_imputed: bool
    n_resamples: int
    calls: list[CallLog] = field(default_factory=list)


async def run_review(backend: Backend, messages: list[dict], labels: list[str], entry_points: dict[str, str],
                     ctx: CallContext, limits, seed_parts: tuple) -> ReviewOutcome:
    schema = review_schema(labels, limits.reason_max_chars)
    sha = prompt_sha(messages)
    calls: list[CallLog] = []
    parsed, reply_text = None, ""
    for attempt in range(limits.max_resamples + 1):
        seed = derive_seed(*seed_parts, "review", attempt)
        ctx.kind, ctx.attempt = "review", attempt
        reply = await backend.chat(messages, max_tokens=limits.review_max_tokens, seed=seed, schema=schema, ctx=ctx)
        try:
            if reply.finish_reason == "length":
                raise ParseError("cut off at max_tokens")
            parsed = parse_review(reply.text, labels, limits.reason_max_chars)
            calls.append(CallLog("review", attempt, seed, sha, reply, True))
            reply_text = reply.text
            break
        except ParseError as e:
            calls.append(CallLog("review", attempt, seed, sha, reply, False, str(e)))
    if parsed is None:
        reviews = {lb: {"verdict": "REJECT", "reason": "", "fix": "", "fix_reprompted": False, "fix_missing": True}
                   for lb in labels}
        return ReviewOutcome(reviews, "", True, len(calls) - 1, calls)

    reviews = {lb: {**parsed[lb], "fix_reprompted": False, "fix_missing": False} for lb in labels}
    missing = [lb for lb in labels if reviews[lb]["verdict"] == "REJECT" and not fix_ok(reviews[lb]["fix"], entry_points[lb])]
    if missing:
        from .render import fix_reprompt_text

        fmsgs = messages + [{"role": "assistant", "content": reply_text},
                            {"role": "user", "content": fix_reprompt_text(missing)}]
        seed = derive_seed(*seed_parts, "fix", 0)
        ctx.kind, ctx.attempt = "fix", 0
        ctx.truth = {**ctx.truth, "missing": missing}
        reply = await backend.chat(fmsgs, max_tokens=limits.fix_max_tokens, seed=seed,
                                   schema=fix_schema(missing), ctx=ctx)
        got: dict = {}
        err = None
        try:
            if reply.finish_reason == "length":
                raise ParseError("cut off at max_tokens")
            obj = json.loads(reply.text)
            got = {lb: obj[lb]["fix"] for lb in missing if isinstance(obj.get(lb), dict) and isinstance(obj[lb].get("fix"), str)}
        except (json.JSONDecodeError, ParseError, TypeError) as e:
            err = str(e)
        calls.append(CallLog("fix", 0, seed, prompt_sha(fmsgs), reply, err is None, err))
        for lb in missing:
            reviews[lb]["fix_reprompted"] = True
            if lb in got and fix_ok(got[lb], entry_points[lb]):
                reviews[lb]["fix"] = got[lb]
            else:
                reviews[lb]["fix_missing"] = True
    for lb in labels:
        if reviews[lb]["verdict"] == "APPROVE":
            reviews[lb]["fix"] = ""
    return ReviewOutcome(reviews, reply_text, False, len([c for c in calls if c.kind == "review"]) - 1, calls)
