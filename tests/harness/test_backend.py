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
