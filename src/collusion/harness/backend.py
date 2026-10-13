"""Model backends.

- `VLLMBackend`: vLLM's OpenAI-compatible server (chat completions with JSON-schema
  structured output, thinking disabled through the chat template), plus its
  /tokenize and /metrics endpoints.
- `OpenRouterBackend`: OpenRouter's chat completions API, pinned to the
  configured provider(s) with no fallback, reasoning off through the `reasoning`
  parameter, and a spend cap from the per-call `usage.cost`.
- `FakeBackend`: scripted agents for tests, dry runs and the Step 4 noise
  simulator. It reads ground truth from the call context, which the real
  backends never see.

All return a `Reply`. A global semaphore caps in-flight requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import Config, FakePolicy

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Reply:
    text: str
    finish_reason: str  # "stop" | "length" | ...
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None  # prefix-cache hits, when the server reports them
    latency: float
    reasoning: str | None = None  # reasoning trace (thinking mode only); logged, never fed back
    cost: float | None = None  # USD charged for the call (OpenRouter)
    reasoning_tokens: int | None = None  # billed reasoning tokens (OpenRouter); 0 when reasoning is off
    provider: str | None = None  # provider that served the call (OpenRouter)


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


class OutOfCredit(RuntimeError):
    """OpenRouter returned 402: the account or key has insufficient credits."""


class BudgetExceeded(RuntimeError):
    """The run's logged spending reached backend.max_cost_usd."""


class ReplyError(RuntimeError):
    """An error reported inside a 200 response (retryable)."""


def _json_schema_format(schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}


# --------------------------------------------------------------------------- #
# OpenAI-compatible servers (vLLM, OpenRouter)
# --------------------------------------------------------------------------- #


class _OpenAICompatBackend:
    """HTTP client, concurrency cap and retry loop shared by vLLM and OpenRouter."""

    label = "server"
    RETRYABLE = frozenset({429, 500, 502, 503, 504})

    def __init__(self, cfg: Config, transport=None, *, base_url: str, trust_env: bool, headers: dict | None = None):
        import httpx2

        self._httpx = httpx2
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.backend.concurrency)
        self.client = httpx2.AsyncClient(base_url=base_url, timeout=cfg.backend.timeout_s, trust_env=trust_env,
                                         transport=transport, headers=headers or {},
                                         limits=httpx2.Limits(max_connections=cfg.backend.concurrency + 4))

    def _check_response(self, r) -> None:
        """Hook: raise a non-retryable error for a status that needs special handling."""

    def _check_body(self, data: dict) -> None:
        """Hook: raise ReplyError for an error reported inside a 200 response."""

    async def _post(self, path: str, body: dict) -> dict:
        h = self._httpx
        for attempt in range(self.cfg.backend.max_http_retries + 1):
            try:
                r = await self.client.post(path, json=body)
                if r.status_code in self.RETRYABLE:
                    raise h.HTTPStatusError(f"retryable status {r.status_code}", request=r.request, response=r)
                self._check_response(r)
                r.raise_for_status()
                data = r.json()
                self._check_body(data)
                return data
            except (h.TransportError, h.HTTPStatusError, ReplyError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = isinstance(e, (h.TransportError, ReplyError)) or status in self.RETRYABLE
                if not retryable or attempt == self.cfg.backend.max_http_retries:
                    raise
                delay = min(60.0, 2.0**attempt)
                from tqdm import tqdm

                tqdm.write(f"warning: {self.label} {path} failed ({type(e).__name__}: {str(e)[:120]}); "
                           f"retry {attempt + 1}/{self.cfg.backend.max_http_retries} in {delay:.0f} s", file=sys.stderr)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def request_body(self, messages: list[dict], max_tokens: int, seed: int, schema: dict | None) -> dict:
        raise NotImplementedError

    def _reply(self, data: dict, latency: float) -> Reply:
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

    async def chat(self, messages, *, max_tokens, seed, schema, ctx) -> Reply:
        body = self.request_body(messages, max_tokens, seed, schema)
        async with self.sem:
            t0 = time.monotonic()
            data = await self._post("/v1/chat/completions", body)
            latency = time.monotonic() - t0
        return self._reply(data, latency)

    async def aclose(self) -> None:
        await self.client.aclose()


# --------------------------------------------------------------------------- #
# vLLM
# --------------------------------------------------------------------------- #


class VLLMBackend(_OpenAICompatBackend):
    label = "vLLM"

    def __init__(self, cfg: Config, transport=None):
        super().__init__(cfg, transport, base_url=cfg.backend.base_url, trust_env=False)

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
            body["response_format"] = _json_schema_format(schema)
        return body

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


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #


def env_key(name: str) -> str | None:
    """An API key from the environment, else from a gitignored .env at the repo root."""
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == name:
                return v.strip().strip("'\"") or None
    return None


def _norm_provider(name: str) -> str:
    """'open-inference' (slug) and 'OpenInference' (display name) compare equal."""
    return re.sub(r"[^a-z0-9]", "", name.split("/")[0].lower())


class OpenRouterBackend(_OpenAICompatBackend):
    """OpenRouter chat completions, pinned to `backend.provider` with no fallback.

    Sampling fields at their neutral value (top_k <= 0, min_p == 0,
    presence_penalty == 0, repetition_penalty == 1) are left out, so the model
    runs on its own defaults for them and `require_parameters` does not filter
    providers on settings that are not used. Reasoning is switched off with the
    `reasoning` parameter. Every reply's `usage.cost` counts toward the spend cap.
    """

    label = "OpenRouter"
    RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})

    def __init__(self, cfg: Config, transport=None, spent: float = 0.0):
        key = env_key(cfg.backend.api_key_env)
        if not key:
            raise RuntimeError(f"no OpenRouter API key: set {cfg.backend.api_key_env} or add it to {ROOT / '.env'}")
        super().__init__(cfg, transport, base_url=cfg.backend.base_url, trust_env=True,
                         headers={"Authorization": f"Bearer {key}"})
        self.spent = spent
        self._pinned = {_norm_provider(p) for p in cfg.backend.provider}

    def request_body(self, messages: list[dict], max_tokens: int, seed: int, schema: dict | None) -> dict:
        s, b = self.cfg.sampling, self.cfg.backend
        body = {"model": b.model, "messages": messages, "max_tokens": max_tokens, "seed": seed,
                "temperature": s.temperature, "top_p": s.top_p}
        if s.top_k > 0:
            body["top_k"] = s.top_k
        if s.min_p > 0:
            body["min_p"] = s.min_p
        if s.presence_penalty != 0:
            body["presence_penalty"] = s.presence_penalty
        if s.repetition_penalty != 1.0:
            body["repetition_penalty"] = s.repetition_penalty
        body["reasoning"] = {"enabled": s.enable_thinking}
        if schema is not None:
            body["response_format"] = _json_schema_format(schema)
        provider = {"order": list(b.provider), "only": list(b.provider), "allow_fallbacks": False,
                    "require_parameters": True}
        if b.quantizations:
            provider["quantizations"] = list(b.quantizations)
        body["provider"] = provider
        return body

    def _check_response(self, r) -> None:
        if r.status_code == 402:
            raise OutOfCredit(f"OpenRouter 402 (insufficient credits): {r.text[:300]}")

    def _check_body(self, data: dict) -> None:
        cost = (data.get("usage") or {}).get("cost")
        err = data.get("error")
        choices = data.get("choices") or []
        if err or not choices or choices[0].get("finish_reason") == "error":
            self.spent += cost or 0.0  # a failed generation may still be billed
            msg = (err or {}).get("message") if isinstance(err, dict) else err
            raise ReplyError(f"error in 200 response: {msg or 'no choices / finish_reason=error'}"[:300])
        served = data.get("provider")
        if served and _norm_provider(served) not in self._pinned:
            raise RuntimeError(f"served by {served!r}, not the pinned provider(s) {list(self.cfg.backend.provider)}")

    def _reply(self, data: dict, latency: float) -> Reply:
        reply = super()._reply(data, latency)
        usage = data.get("usage") or {}
        reply.cost = usage.get("cost")
        reply.reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        reply.provider = data.get("provider")
        return reply

    async def chat(self, messages, *, max_tokens, seed, schema, ctx) -> Reply:
        cap = self.cfg.backend.max_cost_usd
        if cap > 0 and self.spent >= cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f} of the ${cap:.2f} cap (backend.max_cost_usd)")
        reply = await super().chat(messages, max_tokens=max_tokens, seed=seed, schema=schema, ctx=ctx)
        self.spent += reply.cost or 0.0
        return reply

    async def credits(self) -> dict:
        """Account credits: {"total_credits": ..., "total_usage": ...}."""
        r = await self.client.get("/v1/credits")
        self._check_response(r)
        r.raise_for_status()
        return r.json().get("data", {})


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


def make_backend(cfg: Config, tokenizer=None, spent: float = 0.0) -> Backend:
    """`spent`: USD already logged in this run (OpenRouter spend cap)."""
    if cfg.backend.kind == "vllm":
        return VLLMBackend(cfg)
    if cfg.backend.kind == "openrouter":
        return OpenRouterBackend(cfg, spent=spent)
    return FakeBackend(cfg, tokenizer)
