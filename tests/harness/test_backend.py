import asyncio
import json

import httpx2
import pytest

from collusion.harness import config as C
from collusion.harness.backend import CallContext, VLLMBackend, parse_metrics


def _cfg():
    return C.from_dict({"run_id": "t", "backend": {"kind": "vllm", "max_http_retries": 2}})


def test_request_body_and_reply_parsing(monkeypatch):
    seen = []
    flaky = {"n": 0}

    def handler(request: httpx2.Request):
        body = json.loads(request.content)
        seen.append(body)
        if flaky["n"] == 0:
            flaky["n"] += 1
            return httpx2.Response(503)
        return httpx2.Response(200, json={
            "choices": [{"message": {"content": '{"X": 1}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 96}}})

    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    be = VLLMBackend(_cfg(), transport=httpx2.MockTransport(handler))
    schema = {"type": "object"}

    async def go():
        try:
            return await be.chat([{"role": "user", "content": "hi"}], max_tokens=50, seed=7, schema=schema,
                                 ctx=CallContext("s", "arm1", 1, "A", "review"))
        finally:
            await be.aclose()

    r = asyncio.run(go())
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.cached_tokens) == ('{"X": 1}', 120, 7, 96)
    assert len(seen) == 2  # one retry after 503
    b = seen[-1]
    assert b["seed"] == 7 and b["max_tokens"] == 50 and b["temperature"] == 0.7 and b["top_k"] == 20
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["response_format"]["json_schema"]["schema"] == schema


def test_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(400, json={"error": "bad"})

    be = VLLMBackend(_cfg(), transport=httpx2.MockTransport(handler))

    async def go():
        try:
            await be.chat([{"role": "user", "content": "x"}], max_tokens=5, seed=1, schema=None,
                          ctx=CallContext("s", "arm1", 1, "A", "notes"))
        finally:
            await be.aclose()

    with pytest.raises(httpx2.HTTPStatusError):
        asyncio.run(go())
    assert len(calls) == 1


def test_parse_metrics():
    text = """
# HELP vllm:prefix_cache_hits_total x
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 120.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 200.0
vllm:num_preemptions_total{engine="0"} 3
vllm:kv_cache_usage_perc{engine="0"} 0.5
vllm:other_metric 9
"""
    m = parse_metrics(text)
    assert m == {"prefix_cache_hits": 120.0, "prefix_cache_queries": 200.0, "num_preemptions": 3.0,
                 "kv_cache_usage_perc": 0.5}


def test_retry_prints_warning_then_succeeds(capsys):
    import httpx2

    from collusion.harness.backend import VLLMBackend
    from collusion.harness.config import Backend, Config

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx2.Response(503, json={"error": "busy"})
        return httpx2.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    cfg = Config(run_id="t", backend=Backend(kind="vllm", max_http_retries=2))
    be = VLLMBackend(cfg, transport=httpx2.MockTransport(handler))

    async def go():
        import collusion.harness.backend as B

        real_sleep = B.asyncio.sleep
        B.asyncio.sleep = lambda s: real_sleep(0)  # skip the backoff wait
        try:
            return await be._post("/v1/chat/completions", {})
        finally:
            B.asyncio.sleep = real_sleep
            await be.aclose()

    import asyncio

    out = asyncio.run(go())
    assert calls["n"] == 2 and out["choices"][0]["finish_reason"] == "stop"
    assert "retry 1/2" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #


def _or_cfg(**backend):
    return C.from_dict({"run_id": "t", "backend": {
        "kind": "openrouter", "base_url": "https://openrouter.ai/api", "model": "deepseek/deepseek-v4-flash-0731",
        "provider": ["open-inference"], "quantizations": ["fp4"], "tokenizer": "deepseek-v4",
        "max_http_retries": 2, **backend},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "enable_thinking": False}})


def _ok(content='{"X": 1}', cost=0.001, provider="OpenInference", **usage):
    return {"provider": provider, "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 9, "cost": cost,
                      "completion_tokens_details": {"reasoning_tokens": 0}, **usage}}


def _or_call(cfg, handler, n=1, spent=0.0, monkeypatch=None):
    from collusion.harness.backend import OpenRouterBackend

    async def no_sleep(_):
        return None

    if monkeypatch is not None:
        monkeypatch.setattr(asyncio, "sleep", no_sleep)
    be = OpenRouterBackend(cfg, transport=httpx2.MockTransport(handler), spent=spent)

    async def go():
        try:
            out = []
            for _ in range(n):
                out.append(await be.chat([{"role": "user", "content": "hi"}], max_tokens=50, seed=7,
                                         schema={"type": "object"}, ctx=CallContext("s", "arm1", 1, "A", "review")))
            return out, be.spent
        finally:
            await be.aclose()

    return asyncio.run(go())


@pytest.fixture
def or_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")


def test_openrouter_request_body_and_reply(or_key):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(200, json=_ok())

    (r,), spent = _or_call(_or_cfg(), handler)
    req = seen[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    b = json.loads(req.content)
    assert b["model"] == "deepseek/deepseek-v4-flash-0731" and b["seed"] == 7 and b["max_tokens"] == 50
    assert (b["temperature"], b["top_p"]) == (1.0, 1.0)
    for neutral in ("top_k", "min_p", "presence_penalty", "repetition_penalty", "chat_template_kwargs"):
        assert neutral not in b, neutral
    assert b["reasoning"] == {"enabled": False}
    assert b["provider"] == {"order": ["open-inference"], "only": ["open-inference"], "allow_fallbacks": False,
                             "require_parameters": True, "quantizations": ["fp4"]}
    assert b["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert (r.text, r.prompt_tokens, r.cost, r.reasoning_tokens, r.provider) == ('{"X": 1}', 100, 0.001, 0, "OpenInference")
    assert spent == pytest.approx(0.001)


def test_openrouter_non_neutral_sampling_is_sent(or_key):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_ok())

    cfg = _or_cfg().replace(sampling=C.Sampling(temperature=0.6, top_p=0.9, top_k=40, presence_penalty=0.5))
    _or_call(cfg, handler)
    assert (seen[0]["top_k"], seen[0]["presence_penalty"]) == (40, 0.5) and "min_p" not in seen[0]


def test_openrouter_402_is_not_retried(or_key):
    from collusion.harness.backend import OutOfCredit

    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}})

    with pytest.raises(OutOfCredit):
        _or_call(_or_cfg(), handler)
    assert len(calls) == 1


def test_openrouter_error_in_200_and_503_are_retried(or_key, monkeypatch):
    replies = [httpx2.Response(503, json={"error": {"code": 503, "message": "no provider"}}),
               httpx2.Response(200, json={"error": {"code": 502, "message": "upstream died"}}),
               httpx2.Response(200, json=_ok())]

    def handler(request):
        return replies.pop(0)

    (r,), _ = _or_call(_or_cfg(), handler, monkeypatch=monkeypatch)
    assert r.text == '{"X": 1}' and not replies


def test_openrouter_finish_reason_error_is_retried(or_key, monkeypatch):
    bad = _ok()
    bad["choices"][0]["finish_reason"] = "error"
    replies = [httpx2.Response(200, json=bad), httpx2.Response(200, json=_ok())]
    (r,), spent = _or_call(_or_cfg(), lambda req: replies.pop(0), monkeypatch=monkeypatch)
    assert r.finish_reason == "stop" and spent == pytest.approx(0.002)  # the failed generation is billed too


def test_openrouter_rejects_other_provider(or_key):
    with pytest.raises(RuntimeError, match="pinned provider"):
        _or_call(_or_cfg(), lambda req: httpx2.Response(200, json=_ok(provider="DeepInfra")))


def test_openrouter_spend_cap(or_key):
    from collusion.harness.backend import BudgetExceeded

    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(200, json=_ok(cost=0.01))

    cfg = _or_cfg(max_cost_usd=0.05)
    out, spent = _or_call(cfg, handler, n=2, spent=0.03)  # resumed run: $0.03 already logged
    assert spent == pytest.approx(0.05) and len(out) == 2
    with pytest.raises(BudgetExceeded):
        _or_call(cfg, handler, n=1, spent=spent)
    assert len(calls) == 2  # the over-cap request was never sent


def test_openrouter_needs_api_key(monkeypatch, tmp_path):
    import collusion.harness.backend as B

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(B, "ROOT", tmp_path)  # no .env there
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        B.OpenRouterBackend(_or_cfg())
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=x\nOPENROUTER_API_KEY='sk-from-env-file'\n")
    assert B.env_key("OPENROUTER_API_KEY") == "sk-from-env-file"
