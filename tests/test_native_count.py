"""Developer REST countTokens coverage for the shared Gemini request plan."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolSpec,
)

from amplifier_module_provider_gemini import GeminiProvider


def _provider(*, use_streaming: bool = False) -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        config={
            "default_model": "gemini-3.7-flash",
            "max_retries": 0,
            "use_streaming": use_streaming,
            "extra_request_params": {
                "cached_content": "cachedContents/unit-test",
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    }
                ],
                "tool_config": {"function_calling_config": {"mode": "ANY"}},
                "top_p": 0.5,
            },
        },
    )


def _request() -> ChatRequest:
    signature = base64.b64encode(b"thought-signature").decode("ascii")
    return ChatRequest(
        messages=[
            Message(role="system", content="System instructions"),
            Message(role="developer", content="Developer instructions"),
            Message(
                role="user",
                content=[
                    TextBlock(text="Use the attachment."),
                    ImageBlock(
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(
                                b"not-a-real-image"
                            ).decode("ascii"),
                        }
                    ),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="reasoning", signature=signature),
                    ToolCallBlock(
                        id="call-1",
                        name="lookup",
                        input={"query": "native count"},
                    ),
                ],
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"query": "native count"},
                        "signature": signature,
                    }
                ],
            ),
            Message(
                role="tool",
                name="lookup",
                tool_call_id="call-1",
                content="tool result",
            ),
        ],
        tools=[
            ToolSpec(
                name="lookup",
                description="Look up a value",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        max_output_tokens=321,
        reasoning_effort="low",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["nonstream", "stream"])
async def test_count_projection_matches_the_real_sdk_generation_body(
    streaming: bool,
) -> None:
    """The count body is built from the same public plan the SDK sends."""
    from google import genai
    from google.genai import types

    sent: list[dict[str, Any]] = []

    def receive(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "OK"}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            },
        )

    provider = _provider(use_streaming=streaming)
    request = _request()
    plan = provider._build_request_plan(request)
    expected = provider._count_request_payload(plan)["generateContentRequest"]
    client = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(
            base_url="https://unit.test/",
            async_client_args={"transport": httpx.MockTransport(receive)},
        ),
    )
    provider._client = client
    try:
        await provider.complete(request)
        assert sent == [expected]
    finally:
        await client.aio.aclose()
        client.close()


def test_direct_keywords_override_released_request_options() -> None:
    provider = _provider()

    options = provider._merge_request_options(
        {"model": "gemini-2.5-flash", "max_tokens": 100},
        {"max_tokens": 200},
    )

    assert options == {"model": "gemini-2.5-flash", "max_tokens": 200}


@pytest.mark.asyncio
async def test_request_budget_posts_full_request_and_keeps_input_limit_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[httpx.Request] = []
    transport = httpx.MockTransport(
        lambda request: sent.append(request)
        or httpx.Response(200, json={"totalTokens": 100})
    )
    real_client = httpx.AsyncClient

    def client_factory(*, timeout: float) -> httpx.AsyncClient:
        return real_client(timeout=timeout, transport=transport)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    provider = _provider()

    decision = await provider.request_budget(_request(), context_estimate=400)

    assert decision == {
        "estimated_input_tokens": 100,
        "input_limit_tokens": 1_048_576,
        "context_token_budget": 400,
        "max_output_tokens": 321,
        "measurement": {
            "kind": "provider_count",
            "source": "gemini.developer.countTokens",
            "input_tokens": 100,
        },
    }
    assert len(sent) == 1
    assert sent[0].url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.7-flash:countTokens"
    )
    assert sent[0].headers["x-goog-api-key"] == "test-key"
    assert json.loads(sent[0].content)["generateContentRequest"]["cachedContent"] == (
        "cachedContents/unit-test"
    )


@pytest.mark.asyncio
async def test_request_budget_returns_unavailable_for_malformed_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.AsyncClient

    def client_factory(*, timeout: float) -> httpx.AsyncClient:
        return real_client(
            timeout=timeout,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    assert await _provider().request_budget(_request(), context_estimate=400) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_request_params",
    [
        {"response_schema": {"type": "object"}},
        {"cached_content": "short-cache-id"},
        {"system_instruction": "overridden instruction"},
        {"tools": []},
        {"thinking_config": {"thinking_budget": 1, "thinking_level": "low"}},
    ],
)
async def test_request_budget_does_not_silently_drop_unsupported_config(
    monkeypatch: pytest.MonkeyPatch,
    extra_request_params: dict[str, Any],
) -> None:
    provider = _provider()
    provider.extra_request_params = extra_request_params

    def no_count_client(*, timeout: float) -> httpx.AsyncClient:
        raise AssertionError("unsupported request must not reach countTokens")

    monkeypatch.setattr(httpx, "AsyncClient", no_count_client)

    assert await provider.request_budget(_request(), context_estimate=400) is None


@pytest.mark.asyncio
async def test_unknown_effective_model_returns_no_native_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_count_client(*, timeout: float) -> httpx.AsyncClient:
        raise AssertionError("unknown model must not reach countTokens")

    monkeypatch.setattr(httpx, "AsyncClient", no_count_client)
    provider = _provider()

    assert (
        await provider.request_budget(
            _request(),
            context_estimate=400,
            model="gemini-9.9-unverified",
        )
        is None
    )


@pytest.mark.asyncio
async def test_request_budget_repairs_only_its_private_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.AsyncClient

    def client_factory(*, timeout: float) -> httpx.AsyncClient:
        return real_client(
            timeout=timeout,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"totalTokens": 100})
            ),
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    provider = _provider()
    request = ChatRequest(
        messages=[
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="missing", name="lookup", input={"query": "x"})
                ],
                tool_calls=[
                    {
                        "id": "missing",
                        "name": "lookup",
                        "arguments": {"query": "x"},
                    }
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )

    assert await provider.request_budget(request, context_estimate=400)
    assert len(request.messages) == 2
    assert "missing" not in provider._repaired_tool_ids


@pytest.mark.asyncio
async def test_request_budget_cancellation_does_not_dispatch_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    real_client = httpx.AsyncClient

    async def receive(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200, json={"totalTokens": 100})

    def client_factory(*, timeout: float) -> httpx.AsyncClient:
        return real_client(timeout=timeout, transport=httpx.MockTransport(receive))

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    provider = _provider()
    generation_client = object()
    provider._client = generation_client
    task = asyncio.create_task(
        provider.request_budget(_request(), context_estimate=400)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider._client is generation_client


def test_only_developer_api_instances_advertise_native_counting() -> None:
    capabilities = _provider().get_info().capabilities
    assert "request_budget" in capabilities
    assert "request_budget:provider_count" in capabilities

    no_key = GeminiProvider(api_key=None).get_info().capabilities
    unknown = GeminiProvider(
        api_key="test-key",
        config={"default_model": "gemini-9.9-unverified"},
    ).get_info().capabilities
    assert "request_budget" not in no_key
    assert "request_budget:provider_count" not in no_key
    assert "request_budget" not in unknown
    assert "request_budget:provider_count" not in unknown