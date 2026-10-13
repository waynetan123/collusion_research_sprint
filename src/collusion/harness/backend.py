"""Model backends.

- `VLLMBackend`: vLLM's OpenAI-compatible server (chat completions with JSON-schema
  structured output, thinking disabled through the chat template), plus its
  /tokenize and /metrics endpoints.
- `FakeBackend`: scripted agents for tests, dry runs and the Step 4 noise
  simulator. It reads ground truth from the call context, which the real
  backend never sees.

Both return a `Reply`. A global semaphore caps in-flight requests.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Config, FakePolicy


@dataclass
class Reply:
    text: str
    finish_reason: str  # "stop" | "length" | ...
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None  # prefix-cache hits, when the server reports them
    latency: float
    reasoning: str | None = None  # reasoning trace (thinking mode only); logged, never fed back


@dataclass
class CallContext:
    session_id: str
    arm: str
    round: int
    agent: str
    kind: str  # "review" | "fix" | "notes"
    attempt: int = 0
    truth: dict[str, Any] = field(default_factory=dict)  # fake backend only


class Backend(Protocol):
    async def chat(self, messages: list[dict], *, max_tokens: int, seed: int, schema: dict | None,
                   ctx: CallContext) -> Reply: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- #
# vLLM
# --------------------------------------------------------------------------- #


class VLLMBackend:
    def __init__(self, cfg: Config, transport=None):
        import httpx2

        self._httpx = httpx2
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.backend.concurrency)
        self.client = httpx2.AsyncClient(base_url=cfg.backend.base_url, timeout=cfg.backend.timeout_s,
                                         trust_env=False, transport=transport,
                                         limits=httpx2.Limits(max_connections=cfg.backend.concurrency + 4))

    def request_body(self, messages: list[dict], max_tokens: int, seed: int, schema: dict | None) -> dict:
        s = self.cfg.sampling
        body = {
            "model": self.cfg.backend.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "seed": seed,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "min_p": s.min_p,
            "presence_penalty": s.presence_penalty,
            "repetition_penalty": s.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": s.enable_thinking},
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        return body

    async def _post(self, path: str, body: dict) -> dict:
        h = self._httpx
        for attempt in range(self.cfg.backend.max_http_retries + 1):
            try:
                r = await self.client.post(path, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise h.HTTPStatusError(f"retryable status {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except (h.TransportError, h.HTTPStatusError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = isinstance(e, h.TransportError) or status in (429, 500, 502, 503, 504)
                if not retryable or attempt == self.cfg.backend.max_http_retries:
                    raise
                await asyncio.sleep(min(60.0, 2.0**attempt))
        raise AssertionError("unreachable")

    async def chat(self, messages, *, max_tokens, seed, schema, ctx) -> Reply:
        body = self.request_body(messages, max_tokens, seed, schema)
        async with self.sem:
            t0 = time.monotonic()
            data = await self._post("/v1/chat/completions", body)
            latency = time.monotonic() - t0
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return Reply(
            text=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached_tokens=details.get("cached_tokens"),
            latency=latency,
            reasoning=choice["message"].get("reasoning_content") or choice["message"].get("reasoning"),
        )

    async def tokenize_chat(self, messages: list[dict]) -> int:
        body = {"model": self.cfg.backend.model, "messages": messages, "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": self.cfg.sampling.enable_thinking}}
        data = await self._post("/tokenize", body)
        return int(data["count"])

    async def metrics(self) -> dict[str, float]:
        """Prometheus counters relevant to the pilot (prefix cache, preemptions, tokens)."""
        r = await self.client.get("/metrics")
        r.raise_for_status()
        return parse_metrics(r.text)

    async def aclose(self) -> None:
        await self.client.aclose()


_METRIC_RE = re.compile(r"^(vllm:[a-z_]+?)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")
_WANTED = ("prefix_cache_queries", "prefix_cache_hits", "num_preemptions", "prompt_tokens",
           "generation_tokens", "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each wanted vllm:* metric over its label sets."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line.strip())
        if not m:
            continue
        name = m.group(1).removeprefix("vllm:")
        if name in _WANTED:
            out[name] = out.get(name, 0.0) + float(m.group(2))
    return out


# --------------------------------------------------------------------------- #
# fake
# --------------------------------------------------------------------------- #


class FakeBackend:
    """Scripted reviewers. Deterministic given the per-call seed.

    Policies (config.FakePolicy.kind):
    - honest: reject a buggy patch w.p. p_detect, a clean one w.p. p_false_reject;
    - colluder: from round `collude_from` on, approve every partner patch; honest on strangers;
    - reciprocator: approve the partner's patch iff its own last patch was approved; honest otherwise;
    - random: approve w.p. 0.5.
    """

    def __init__(self, cfg: Config, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.sem = asyncio.Semaphore(cfg.backend.concurrency)
        self._last_prompt: dict[tuple[str, str], str] = {}

    def policy(self, arm: str) -> FakePolicy:
        return getattr(self.cfg.fake, arm)

    def _count(self, messages: list[dict]) -> int:
        if self.tokenizer is not None:
            return self.tokenizer.count_chat(messages)
        return sum(len(m["content"]) for m in messages) // 4

    def _verdict(self, pol: FakePolicy, rng: random.Random, info: dict, rnd: int, last_received: str | None) -> str:
        honest = ("REJECT" if rng.random() < pol.p_detect else "APPROVE") if info["buggy"] else (
            "REJECT" if rng.random() < pol.p_false_reject else "APPROVE")
        if pol.kind == "honest":
            return honest
        if pol.kind == "random":
            return "APPROVE" if rng.random() < 0.5 else "REJECT"
        if info["partner"]:
            if pol.kind == "colluder" and rnd >= pol.collude_from:
                return "APPROVE"
            if pol.kind == "reciprocator" and last_received is not None:
                return "APPROVE" if last_received == "APPROVED" else honest
        return honest

    def _review(self, pol: FakePolicy, rng: random.Random, ctx: CallContext) -> str:
        out = {}
        for label, info in sorted(ctx.truth["labels"].items()):
            v = self._verdict(pol, rng, info, ctx.round, ctx.truth.get("last_received"))
            fix = info["clean_code"] if v == "REJECT" and rng.random() < pol.p_fix else ""
            reason = ("The code does not match the specification." if v == "REJECT"
                      else "The code matches the specification.")
            out[label] = {"reason": reason, "verdict": v, "fix": fix}
        return json.dumps(out)

    async def chat(self, messages, *, max_tokens, seed, schema, ctx) -> Reply:
        async with self.sem:
            await asyncio.sleep(0)
            pol = self.policy(ctx.arm)
            rng = random.Random(seed)
            if ctx.kind in ("review", "fix") and rng.random() < pol.p_malformed:
                text = '{"X": {"verdict": "APPRO'
            elif ctx.kind == "review":
                text = self._review(pol, rng, ctx)
            elif ctx.kind == "fix":
                text = json.dumps({lb: {"fix": ctx.truth["labels"][lb]["clean_code"] if rng.random() < pol.p_fix else ""}
                                   for lb in ctx.truth["missing"]})
            else:
                text = pol.note.format(round=ctx.round)
            prompt = "\n".join(m["content"] for m in messages)
            key = (ctx.session_id, ctx.agent)
            prev = self._last_prompt.get(key, "")
            common = 0
            for a, b in zip(prev, prompt):
                if a != b:
                    break
                common += 1
            self._last_prompt[key] = prompt
            ptoks = self._count(messages)
            return Reply(text=text, finish_reason="stop", prompt_tokens=ptoks,
                         completion_tokens=max(1, len(text) // 4), cached_tokens=min(ptoks, common // 4),
                         latency=0.0)

    async def aclose(self) -> None:
        return None


def make_backend(cfg: Config, tokenizer=None) -> Backend:
    if cfg.backend.kind == "vllm":
        return VLLMBackend(cfg)
    return FakeBackend(cfg, tokenizer)
