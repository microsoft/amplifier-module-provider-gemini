"""Strict discovery at the real google-genai HTTP boundary, entirely offline.

Only the SDK transport is replaced. The provider, SDK request construction,
auth headers, JSON parser, pagination and kernel error mapping all run normally.
No live credentials, config files, SDK installation or inference are needed.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from google import genai
from google.genai import types

from amplifier_core.llm_errors import (
    AccessDeniedError,
    AuthenticationError,
    InvalidRequestError,
    LLMError,
    NotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from amplifier_module_provider_gemini import GeminiProvider


@pytest.fixture
def isolated_sdk_environment(monkeypatch):
    """Keep both SDK routes independent of host defaults and credentials."""
    monkeypatch.setattr("os.environ", {})
    adc = Mock(side_effect=AssertionError("ADC must not be consulted"))
    monkeypatch.setattr("google.auth.default", adc)
    yield
    adc.assert_not_called()


@asynccontextmanager
async def provider_at(
    handler, *, config=None, key="fixture-blue", host="blue.invalid", vertex=False,
):
    """Use a real SDK client, explicit fake account/endpoint and mock HTTP only."""
    sent = []
    api_version = "v1" if vertex else "v1beta"
    path = "publishers/google/models" if vertex else "models"

    def respond(request):
        sent.append(request)
        # Any hidden inference request fails, even if the provider catches it.
        assert request.method == "GET"
        assert request.url.host == host
        assert request.url.path == f"/{api_version}/{path}"
        assert request.headers["x-goog-api-key"] == key
        assert "authorization" not in request.headers
        return handler(request)

    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as http:
        with httpx.Client(transport=transport, trust_env=False) as sync_http:
            client = genai.Client(
                api_key=key,
                vertexai=vertex,
                http_options=types.HttpOptions(
                    base_url=f"https://{host}",
                    api_version=api_version,
                    httpx_client=sync_http,
                    httpx_async_client=http,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            provider = GeminiProvider(
                api_key=key,
                config={"max_retries": 0, "retry_jitter": False, **(config or {})},
            )
            provider._client = client
            try:
                yield provider, sent
            finally:
                await client.aio.aclose()
                client.close()
                assert all(request.method == "GET" for request in sent)


def model(name, **extra):
    return {"name": name, **extra}


def envelope(ids):
    return {
        "version": 1,
        "source": "provider-api",
        "models": [{"id": name} for name in ids],
    }


@pytest.mark.asyncio
async def test_exact_api_ids_without_family_age_or_capability_inference():
    names = [
        "models/gemini-old-001",
        "models/gemini-old-002",
        "models/gemini-future-exp",
        "models/future-unrecognized-family",
        "models/text-embedding-any",
        "tunedModels/customer-custom",
        "already-selectable",
    ]
    records = [model(name) for name in names]
    # This is API metadata, not a selection directive. Do not invent family
    # eligibility, chat capabilities or defaults for any of these records.
    records[4]["supportedGenerationMethods"] = ["embedContent"]
    async with provider_at(lambda _: httpx.Response(200, json={"models": records})) as (
        provider,
        sent,
    ):
        result = await provider.discover_models()
        assert result == envelope([name.removeprefix("models/") for name in names])
        assert json.loads(json.dumps(result)) == result
        assert len(sent) == 1
        assert not sent[0].url.params


@pytest.mark.asyncio
async def test_preserves_only_response_metadata():
    record = model(
        "models/novel-model",
        displayName="Account model",
        inputTokenLimit=12345,
        outputTokenLimit=678,
        thinking=True,
    )
    async with provider_at(lambda _: httpx.Response(200, json={"models": [record]})) as (
        provider,
        _,
    ):
        result = await provider.discover_models()
    assert result == {
        "version": 1,
        "source": "provider-api",
        "models": [{
            "id": "novel-model",
            "display_name": "Account model",
            "context_window": 12345,
            "max_output_tokens": 678,
        }],
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_sdk_environment")
@pytest.mark.parametrize("vertex", [False, True], ids=["developer", "vertex"])
@pytest.mark.parametrize("include_collection", [False, True], ids=["omitted", "array"])
@pytest.mark.parametrize("later_page", [False, True])
async def test_genuine_empty_page_is_success_without_defaults(
    vertex, include_collection, later_page,
):
    collection = "publisherModels" if vertex else "models"
    name = "publishers/google/models/first" if vertex else "models/first"
    terminal = {collection: []} if include_collection else {}

    def respond(request):
        if later_page and "pageToken" not in request.url.params:
            return httpx.Response(200, json={
                collection: [model(name)], "nextPageToken": "last",
            })
        return httpx.Response(200, json=terminal)

    async with provider_at(respond, vertex=vertex) as (provider, sent):
        result = await provider.discover_models()
        expected_ids = [name if vertex else "first"] if later_page else []
        assert result == envelope(expected_ids)
        assert json.loads(json.dumps(result)) == result
        assert [request.url.params.get("pageToken") for request in sent] == (
            [None, "last"] if later_page else [None]
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_sdk_environment")
@pytest.mark.parametrize("vertex", [False, True], ids=["developer", "vertex"])
@pytest.mark.parametrize("include_collection", [False, True], ids=["omitted", "array"])
async def test_pagination_including_empty_intermediate_pages_and_opaque_token(
    vertex, include_collection,
):
    token = "next+/=&?"
    collection = "publisherModels" if vertex else "models"
    prefix = "publishers/google/models/" if vertex else "models/"
    empty_page = {collection: []} if include_collection else {}

    def respond(request):
        page = request.url.params.get("pageToken")
        if page is None:
            return httpx.Response(200, json={
                collection: [model(prefix + "first")], "nextPageToken": token,
            })
        if page == token:
            return httpx.Response(200, json={**empty_page, "nextPageToken": "last"})
        assert page == "last"
        return httpx.Response(200, json={collection: [model(prefix + "last")]})

    async with provider_at(respond, vertex=vertex) as (provider, sent):
        result = await provider.discover_models()
        expected_ids = [prefix + name if vertex else name for name in ["first", "last"]]
        assert result == envelope(expected_ids)
        assert json.loads(json.dumps(result)) == result
        assert [request.url.params.get("pageToken") for request in sent] == [
            None, token, "last",
        ]


@pytest.mark.asyncio
async def test_every_call_refetches_and_returns_fresh_results():
    replies = iter([
        {"models": [model("models/first")]},
        {"models": [model("models/second")]},
        {"models": []},
    ])
    async with provider_at(lambda _: httpx.Response(200, json=next(replies))) as (
        provider,
        sent,
    ):
        first = await provider.discover_models()
        first["models"].append({"id": "caller-injected"})
        assert await provider.discover_models() == envelope(["second"])
        assert await provider.discover_models() == envelope([])
        assert len(sent) == 3


@pytest.mark.asyncio
async def test_model_effort_priority_and_legacy_listing_do_not_select_inventory():
    async with provider_at(
        lambda _: httpx.Response(200, json={"models": [model("models/api-only")]}),
    ) as (provider, sent):
        # Strict discovery must not call the legacy filtered surface.
        provider.list_models = AsyncMock(side_effect=AssertionError("legacy listing"))
        before = await provider.discover_models()
        provider.config.update({
            "model": "not-in-inventory", "default_model": "another-selection",
            "reasoning_effort": "high", "priority": -100,
        })
        provider.default_model = "another-selection"
        provider.priority = -100
        after = await provider.discover_models()
        assert before == after == envelope(["api-only"])
        provider.list_models.assert_not_awaited()
        assert len(sent) == 2
        assert sent[0].url == sent[1].url
        assert sent[0].content == sent[1].content


@pytest.mark.asyncio
async def test_each_account_and_endpoint_uses_its_own_inventory():
    async with provider_at(
        lambda _: httpx.Response(200, json={"models": [model("models/blue-only")]}),
    ) as (blue, blue_sent):
        async with provider_at(
            lambda _: httpx.Response(200, json={"models": [model("models/red-only")]}),
            key="fixture-red", host="red.invalid",
        ) as (red, red_sent):
            assert await blue.discover_models() == envelope(["blue-only"])
            assert await red.discover_models() == envelope(["red-only"])
            assert len(blue_sent) == len(red_sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("account", ["blue", "red"])
async def test_vertex_publisher_discovery_preserves_route_auth_and_pagination(
    monkeypatch, account,
):
    # Do not let SDK defaults inspect the runner's real account/configuration.
    # Explicit Vertex API-key mode also must never attempt ADC.
    monkeypatch.setattr("os.environ", {})
    adc = Mock(side_effect=AssertionError("ADC must not be consulted"))
    monkeypatch.setattr("google.auth.default", adc)
    key = f"fixture-vertex-{account}"
    host = f"vertex-{account}.invalid"
    names = [
        f"publishers/google/models/{account}-first",
        f"publishers/google/models/{account}-second",
    ]
    token = "vertex-next+/=&?"
    sent = []

    def respond(request):
        sent.append(request)
        assert request.method == "GET"
        assert request.url.scheme == "https"
        assert request.url.host == host
        assert request.url.path == "/v1/publishers/google/models"
        assert request.headers["x-goog-api-key"] == key
        assert "authorization" not in request.headers
        assert not request.content
        if len(sent) == 1:
            assert not request.url.params
            return httpx.Response(200, json={
                "publisherModels": [model(names[0])],
                "nextPageToken": token,
            })
        assert len(sent) == 2
        assert dict(request.url.params) == {"pageToken": token}
        return httpx.Response(200, json={"publisherModels": [model(names[1])]})

    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as http:
        with httpx.Client(transport=transport, trust_env=False) as sync_http:
            client = genai.Client(
                vertexai=True,
                api_key=key,
                http_options=types.HttpOptions(
                    base_url=f"https://{host}",
                    api_version="v1",
                    httpx_client=sync_http,
                    httpx_async_client=http,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            # The owning SDK connection is authoritative, not the provider's
            # unused fallback key or default chat-model selection.
            provider = GeminiProvider(
                api_key="unused-provider-key", config={"max_retries": 0},
            )
            provider._client = client
            try:
                assert client.vertexai is True
                assert await provider.discover_models() == envelope(names)
                assert len(sent) == 2
                adc.assert_not_called()
            finally:
                await client.aio.aclose()
                client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error", [
    (400, InvalidRequestError),
    (401, AuthenticationError),
    (403, AccessDeniedError),
    (404, NotFoundError),
    (429, RateLimitError),
    (500, ProviderUnavailableError),
    (503, ProviderUnavailableError),
])
@pytest.mark.parametrize("later_page", [False, True])
async def test_api_errors_never_become_empty_or_partial_success(status, error, later_page):
    def respond(request):
        if later_page and "pageToken" not in request.url.params:
            return httpx.Response(200, json={
                "models": [model("models/partial")], "nextPageToken": "next",
            })
        return httpx.Response(status, json={
            "error": {"code": status, "message": "fixture API error", "status": "ERROR"},
        })

    async with provider_at(respond) as (provider, sent):
        with pytest.raises(error) as caught:
            await provider.discover_models()
        assert caught.value.provider == "gemini"
        assert caught.value.status_code == status
        assert len(sent) == (2 if later_page else 1)


@pytest.mark.asyncio
async def test_later_transient_failure_retries_entire_inventory(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    first_pages = 0

    def respond(request):
        nonlocal first_pages
        if "pageToken" not in request.url.params:
            first_pages += 1
            return httpx.Response(200, json={
                "models": [model(f"models/attempt-{first_pages}")],
                "nextPageToken": "next",
            })
        if first_pages == 1:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(200, json={"models": [model("models/final")]})

    async with provider_at(respond, config={"max_retries": 1}) as (provider, sent):
        assert await provider.discover_models() == envelope(["attempt-2", "final"])
        assert len(sent) == 4
        assert [request.url.params.get("pageToken") for request in sent] == [
            None, "next", None, "next",
        ]


@pytest.mark.asyncio
async def test_transient_failure_exhausts_existing_retry_policy(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    async with provider_at(
        lambda _: httpx.Response(503, json={"error": {"message": "unavailable"}}),
        config={"max_retries": 2},
    ) as (provider, sent):
        with pytest.raises(ProviderUnavailableError):
            await provider.discover_models()
        assert len(sent) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failure_is_kernel_error(exception):
    def respond(request):
        raise exception("offline fixture transport failure", request=request)

    async with provider_at(respond) as (provider, sent):
        with pytest.raises(LLMError) as caught:
            await provider.discover_models()
        assert caught.value.provider == "gemini"
        assert caught.value.retryable is True
        assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_sdk_environment")
@pytest.mark.parametrize("vertex", [False, True], ids=["developer", "vertex"])
@pytest.mark.parametrize("payload", [
    None, [], {"unrelated": []}, {"models": None}, {"models": {}},
    {"models": "bad"}, {"models": False}, {"models": 0},
    {"models": [None]}, {"models": [{}]},
    {"models": [model(None)]}, {"models": [model(123)]},
    {"models": [model("")]}, {"models": [model(" models/padded")]},
    {"models": [model("models/")]}, {"models": [model("models/has space")]},
    {"models": [model("models/ok", inputTokenLimit="123")]},
    {"models": [model("models/ok", outputTokenLimit=True)]},
    {"models": [model("models/ok", inputTokenLimit=-1)]},
    {"models": [model("models/ok", displayName=None)]},
    {"models": [], "error": {"message": "failure disguised as HTTP 200"}},
    {"models": [], "nextPageToken": 42},
    {"models": [], "nextPageToken": False},
    {"models": [], "nextPageToken": []},
    {"models": [], "nextPageToken": {}},
    {"error": {"message": "failure disguised as HTTP 200"}},
    {"nextPageToken": "next", "error": {"message": "failure disguised as HTTP 200"}},
    {"nextPageToken": "next", "unrelated": []},
    {"nextPageToken": 42}, {"nextPageToken": False},
    {"nextPageToken": []}, {"nextPageToken": {}},
])
@pytest.mark.parametrize("later_page", [False, True])
async def test_malformed_response_is_nonretryable_kernel_error(payload, later_page, vertex):
    collection = "publisherModels" if vertex else "models"
    if vertex and isinstance(payload, dict) and "models" in payload:
        payload = {**payload}
        payload[collection] = payload.pop("models")

    def respond(request):
        if later_page and "pageToken" not in request.url.params:
            return httpx.Response(200, json={
                collection: [model("models/partial")], "nextPageToken": "next",
            })
        return httpx.Response(200, content=json.dumps(payload))

    async with provider_at(respond, config={"max_retries": 2}, vertex=vertex) as (
        provider, sent,
    ):
        with pytest.raises(LLMError, match="Malformed Gemini") as caught:
            await provider.discover_models()
        assert caught.value.retryable is False
        assert len(sent) == (2 if later_page else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "<html>proxy failure</html>", "{invalid"])
async def test_invalid_json_is_not_an_empty_inventory(content):
    async with provider_at(lambda _: httpx.Response(200, content=content)) as (
        provider,
        sent,
    ):
        with pytest.raises(LLMError, match="Malformed Gemini"):
            await provider.discover_models()
        assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_sdk_environment")
@pytest.mark.parametrize("vertex", [False, True], ids=["developer", "vertex"])
@pytest.mark.parametrize("include_collection", [False, True], ids=["omitted", "array"])
async def test_repeated_pagination_token_fails_without_partial_result(
    vertex, include_collection,
):
    collection = "publisherModels" if vertex else "models"
    page = {collection: [model("models/partial")]} if include_collection else {}
    async with provider_at(lambda _: httpx.Response(200, json={
        **page, "nextPageToken": "cycle",
    }), vertex=vertex) as (provider, sent):
        with pytest.raises(LLMError, match="Malformed Gemini") as caught:
            await provider.discover_models()
        assert caught.value.retryable is False
        assert len(sent) == 2


@pytest.mark.asyncio
async def test_missing_key_fails_before_sdk_construction(monkeypatch):
    constructor = AsyncMock(side_effect=AssertionError("must not construct SDK client"))
    monkeypatch.setattr(genai, "Client", constructor)
    provider = GeminiProvider(api_key=None, config={"max_retries": 0})
    with pytest.raises(AuthenticationError):
        await provider.discover_models()
    constructor.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_propagates_without_success_or_retry():
    def respond(_):
        raise asyncio.CancelledError()

    async with provider_at(respond) as (provider, sent):
        with pytest.raises(asyncio.CancelledError):
            await provider.discover_models()
        assert len(sent) == 1


@pytest.mark.asyncio
async def test_legacy_projection_remains_unchanged_and_independent():
    records = [
        model(name, displayName=name, inputTokenLimit=1000, outputTokenLimit=200)
        for name in [
            "models/gemini-2.5-flash", "models/gemini-1.5-pro-001",
            "models/gemini-future-exp", "models/not-gem-family",
        ]
    ]
    async with provider_at(lambda _: httpx.Response(200, json={"models": records})) as (
        provider,
        sent,
    ):
        legacy_before = await provider.list_models()
        discovered = await provider.discover_models()
        legacy_after = await provider.list_models()
        assert legacy_before == legacy_after
        assert len(legacy_before) == 1
        legacy = legacy_before[0]
        assert legacy.id == "gemini-2.5-flash"
        assert legacy.display_name == "models/gemini-2.5-flash"
        assert legacy.context_window == 1000
        assert legacy.max_output_tokens == 200
        assert legacy.capabilities == [
            "streaming", "json_mode", "thinking", "tools", "fast", "vision",
        ]
        assert legacy.defaults == {"temperature": 0.7, "max_tokens": 200}
        assert [item["id"] for item in discovered["models"]] == [
            record["name"].removeprefix("models/") for record in records
        ]
        assert len(sent) == 3
