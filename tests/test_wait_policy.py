"""Default model waits survive virtual elapsed time; cancellation/errors remain live."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.llm_errors import LLMError, LLMTimeoutError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_gemini import GeminiProvider
from tests.test_streaming import FakeCoordinator, _chunk, _make_usage, _part


def setup_call(streaming, config=None, failure=None):
    provider = GeminiProvider(
        api_key="test-key",
        config={
            "use_streaming": streaming,
            "max_retries": 0,
            **(config or {}),
        },
    )
    provider.coordinator = FakeCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = []

    async def wait():
        entered.set()
        await release.wait()
        if failure:
            raise failure

    async def stream():
        try:
            await wait()
            yield _chunk([_part(text="done")], _make_usage())
        finally:
            closed.append(True)

    async def create(**kwargs):
        await wait()
        return _chunk([_part(text="done")], _make_usage())

    client = MagicMock()
    provider._client = client
    if streaming:
        call = client.aio.models.generate_content_stream = AsyncMock(
            return_value=stream()
        )
    else:
        call = client.aio.models.generate_content = AsyncMock(side_effect=create)

    request = ChatRequest(messages=[Message(role="user", content="hello")])
    return provider, request, entered, release, closed, call


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_default_wait_survives_an_hour_then_completes(monkeypatch, streaming):
    provider, request, entered, release, closed, _call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    # No real sleeping: move beyond every former model-work deadline while
    # the mock provider remains healthy but silent.
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 3600)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
    release.set()
    await task
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_cancellation_propagates_without_retry(streaming):
    provider, request, entered, _release, closed, call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_deadline_still_stops_model_work(monkeypatch, streaming):
    provider, request, entered, _release, closed, _call = setup_call(
        streaming, {"timeout": 10}
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 11)
        with pytest.raises(LLMTimeoutError):
            await task
    assert provider.timeout == 10
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_transport_failure_is_still_reported(streaming):
    provider, request, entered, release, closed, call = setup_call(
        streaming, failure=ConnectionError("transport disconnected")
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    release.set()
    with pytest.raises((LLMError, ConnectionError)):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.parametrize("timeout", [None, 30])
def test_sdk_transport_has_no_hidden_deadline(timeout):
    provider = GeminiProvider(api_key="test-key", config={"timeout": timeout})
    # Inspect SDK request policy without issuing an HTTP request.
    options = provider.client._api_client._http_options
    assert options.timeout == (None if timeout is None else timeout * 1000)
    assert provider.get_info().defaults["timeout"] is None


@pytest.mark.parametrize("timeout, expected", [(None, 600.0), (30, 30)])
def test_token_count_probe_keeps_its_existing_transport_bound(timeout, expected):
    provider = GeminiProvider(api_key="fixture", config={"timeout": timeout})
    assert provider._count_timeout == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("timeout", [None, 30])
async def test_real_sdk_http_request_carries_effective_timeout(
    monkeypatch, streaming, timeout
):
    import json

    import httpx
    from google import genai

    seen = []
    body = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "done"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    }

    async def handle(request):
        seen.append(request)
        if streaming:
            return httpx.Response(
                200,
                text="data: " + json.dumps(body) + "\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=body)

    real_client = genai.Client

    def client_with_in_memory_transport(**kwargs):
        # Keep the provider's HttpOptions unchanged except for a network-free
        # transport whose own short default exposes accidental SDK inheritance.
        kwargs["http_options"].async_client_args = {
            "transport": httpx.MockTransport(handle),
            "timeout": 0.001,
        }
        return real_client(**kwargs)

    monkeypatch.setattr(genai, "Client", client_with_in_memory_transport)
    provider = GeminiProvider(
        api_key="fixture",
        config={"timeout": timeout, "use_streaming": streaming, "max_retries": 0},
    )
    provider.coordinator = FakeCoordinator()
    try:
        result = await provider.complete(
            ChatRequest(messages=[Message(role="user", content="hello")])
        )
        assert result is not None
        assert len(seen) == 1
        assert seen[0].extensions["timeout"]["read"] == timeout
        assert "http_options" not in json.loads(seen[0].content)
    finally:
        await provider.close()
