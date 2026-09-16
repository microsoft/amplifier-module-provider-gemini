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

from amplifier_module_provider_gemini import (
    GeminiProvider,
    _GeminiRequestPlan,
    _count_system_instruction_content,
)


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


@pytest.mark.asyncio
async def test_request_budget_normalizes_plain_system_and_developer_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain system/developer text reaches countTokens as valid content."""
    sent: list[dict[str, Any]] = []
    real_client = httpx.AsyncClient

    def client_factory(*, timeout: float) -> httpx.AsyncClient:
        return real_client(
            timeout=timeout,
            transport=httpx.MockTransport(
                lambda request: sent.append(json.loads(request.content))
                or httpx.Response(200, json={"totalTokens": 100})
            ),
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    assert await _provider().request_budget(
        ChatRequest(
            messages=[
                Message(role="system", content="System instructions"),
                Message(role="developer", content="Developer instructions"),
                Message(role="user", content="User request"),
            ],
            max_output_tokens=321,
        ),
        context_estimate=400,
    )
    payload = sent[0]["generateContentRequest"]
    assert payload["systemInstruction"] == {
        "parts": [{"text": "System instructions"}],
        "role": "user",
    }
    assert payload["contents"][0] == {
        "role": "user",
        "parts": [{"text": "<context_file>\nDeveloper instructions\n</context_file>"}],
    }


def test_system_instruction_normalization_accepts_public_part_content_and_list() -> None:
    """The projection supports the public instruction representations."""
    from google.genai import types

    part = types.Part(text="Part instruction")
    content = types.Content(role="user", parts=[part])

    assert _count_system_instruction_content(part, types).parts == [part]
    assert _count_system_instruction_content(part, types).role == "user"
    assert _count_system_instruction_content(content, types) is content
    roleless_content = types.Content(parts=[part])
    assert _count_system_instruction_content(roleless_content, types) is roleless_content
    assert roleless_content.role is None
    assert _count_system_instruction_content(
        ["First instruction", part], types
    ).model_dump(exclude_none=True) == {
        "parts": [{"text": "First instruction"}, {"text": "Part instruction"}],
        "role": "user",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("instruction", [object(), [object()]])
async def test_request_budget_returns_unavailable_for_unsupported_system_instruction(
    monkeypatch: pytest.MonkeyPatch, instruction: object
) -> None:
    """Unsupported public forms must not raise or reach countTokens."""
    provider = _provider()
    plan = provider._build_request_plan(
        ChatRequest(messages=[Message(role="user", content="User request")])
    )
    plan.config.system_instruction = instruction
    monkeypatch.setattr(
        provider,
        "_build_request_plan",
        lambda *_args, **_kwargs: _GeminiRequestPlan(
            model=plan.model,
            contents=plan.contents,
            config=plan.config,
            count_unsupported_reason=plan.count_unsupported_reason,
        ),
    )

    def no_count_client(*, timeout: float) -> httpx.AsyncClient:
        raise AssertionError("unsupported request must not reach countTokens")

    monkeypatch.setattr(httpx, "AsyncClient", no_count_client)

    assert (
        await provider.request_budget(
            ChatRequest(messages=[Message(role="user", content="User request")]),
            context_estimate=400,
        )
        is None
    )


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
    task = asyncio.create_task(
        provider.request_budget(_request(), context_estimate=400)
    )
    started_wait = asyncio.create_task(started.wait())
    done, _ = await asyncio.wait(
        {task, started_wait}, timeout=1, return_when=asyncio.FIRST_COMPLETED
    )
    if task in done:
        started_wait.cancel()
        await asyncio.gather(started_wait, return_exceptions=True)
        await task
    if not started_wait.done():
        started_wait.cancel()
        task.cancel()
        await asyncio.gather(started_wait, task, return_exceptions=True)
        pytest.fail("countTokens did not dispatch within one second")
    assert started.is_set(), "countTokens did not dispatch within one second"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider._client is None


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment", "counting_available"),
    [
        ({}, True),
        ({"GOOGLE_GEMINI_BASE_URL": "https://alternate.example/"}, False),
        ({"GOOGLE_GEMINI_BASE_URL": ""}, True),
        ({"GOOGLE_GENAI_USE_VERTEXAI": "TRUE"}, False),
        ({"GOOGLE_GENAI_USE_VERTEXAI": "TrUe"}, False),
        ({"GOOGLE_GENAI_USE_VERTEXAI": "1"}, False),
        ({"GOOGLE_GENAI_USE_VERTEXAI": "false"}, True),
        ({"GOOGLE_GENAI_USE_VERTEXAI": ""}, True),
        ({"GOOGLE_GENAI_USE_VERTEXAI": "yes"}, True),
        (
            {
                "GOOGLE_GENAI_USE_ENTERPRISE": "false",
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
            },
            True,
        ),
        (
            {
                "GOOGLE_GENAI_USE_ENTERPRISE": "TRUE",
                "GOOGLE_GENAI_USE_VERTEXAI": "false",
            },
            False,
        ),
    ],
    ids=[
        "canonical-developer",
        "custom-base-url",
        "empty-base-url",
        "vertex-upper-true",
        "vertex-mixed-true",
        "vertex-one",
        "vertex-false",
        "vertex-empty",
        "vertex-other-false",
        "enterprise-false-wins",
        "enterprise-true-wins",
    ],
)
async def test_native_count_advertisement_and_dispatch_follow_sdk_route_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    counting_available: bool,
) -> None:
    """Neither advertisement nor HTTP can claim a route the SDK would not use."""
    for name in (
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    sent: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def client_factory(*, timeout: float) -> httpx.AsyncClient:
        return real_client(
            timeout=timeout,
            transport=httpx.MockTransport(
                lambda request: sent.append(request)
                or httpx.Response(200, json={"totalTokens": 100})
            ),
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    provider = _provider()

    assert ("request_budget" in provider.get_info().capabilities) is counting_available
    assert (
        await provider.request_budget(_request(), context_estimate=400) is not None
    ) is counting_available
    assert len(sent) == int(counting_available)
    assert provider._client is None


@pytest.mark.asyncio
async def test_initialized_noncanonical_client_stays_uncountable_after_environment_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider records its own client's route instead of rereading later env."""
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    provider = _provider()
    client = provider.client
    assert client.vertexai is True
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    def no_count_client(*, timeout: float) -> httpx.AsyncClient:
        raise AssertionError("noncanonical generation client must not reach countTokens")

    monkeypatch.setattr(httpx, "AsyncClient", no_count_client)
    try:
        assert "request_budget" not in provider.get_info().capabilities
        assert await provider.request_budget(_request(), context_estimate=400) is None
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
async def test_injected_client_with_unknown_route_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No public SDK base_url means an injected client cannot prove canonical."""
    provider = _provider()
    provider._client = object()

    def no_count_client(*, timeout: float) -> httpx.AsyncClient:
        raise AssertionError("unknown injected client must not reach countTokens")

    monkeypatch.setattr(httpx, "AsyncClient", no_count_client)

    assert "request_budget" not in provider.get_info().capabilities
    assert await provider.request_budget(_request(), context_estimate=400) is None


@pytest.mark.asyncio
async def test_replaced_client_cannot_inherit_a_canonical_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route snapshot belongs to its client, not every later injected client."""
    for name in (
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        monkeypatch.delenv(name, raising=False)
    provider = _provider()
    client = provider.client
    assert client.vertexai is False
    assert "request_budget:provider_count" in provider.get_info().capabilities
    provider._client = object()

    def no_count_client(*, timeout: float) -> httpx.AsyncClient:
        raise AssertionError("a replacement client must not inherit count eligibility")

    monkeypatch.setattr(httpx, "AsyncClient", no_count_client)
    try:
        assert "request_budget" not in provider.get_info().capabilities
        assert await provider.request_budget(_request(), context_estimate=400) is None
    finally:
        provider._client = client
        await provider.close()

    # Reinjecting the old closed client must not restore its discarded stamp.
    provider._client = client
    assert "request_budget" not in provider.get_info().capabilities