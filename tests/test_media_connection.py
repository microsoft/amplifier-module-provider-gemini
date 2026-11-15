"""Offline effective SDK/HTTP auth-route regression boundary (no paid calls)."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from google import genai
from google.genai import types

from amplifier_core.llm_errors import InvalidRequestError, ProviderUnavailableError
from amplifier_core.message_models import ChatRequest, ImageBlock, Message
from amplifier_module_provider_gemini import GeminiProvider


class Capability:
    version = 1

    def validate(self, request, *, purpose, request_options):
        return {"valid": True}


def provider(cap, streaming=False):
    return GeminiProvider(
        api_key="offline",
        coordinator=SimpleNamespace(
            get_capability=lambda key: cap, hooks=SimpleNamespace(emit=AsyncMock()),
        ),
        config={"default_model": "gemini-3.1-pro-preview", "use_streaming": streaming,
                "max_retries": 2, "min_retry_delay": 0.001, "raw": True},
    )


def request():
    return ChatRequest(messages=[Message(role="user", content=[
        ImageBlock(source={"type": "base64", "media_type": "image/png", "data": "cGl4ZWxz"}),
    ], metadata={"experimental.media.v1": {"refs": []}})])


def response():
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}]}


async def close(client):
    await client.aio.aclose()
    client.close()


def selected(p, monkeypatch, receive):
    """Exercise the real provider factory with a real SDK, only its socket mocked."""
    client = genai.Client(api_key="offline", http_options=types.HttpOptions(
        async_client_args={"transport": httpx.MockTransport(receive)},
    ))
    with monkeypatch.context() as patch:
        patch.setattr(genai, "Client", lambda **kwargs: client)
        assert p.client is client
    return client


async def count(p, req, monkeypatch):
    real = httpx.AsyncClient
    counted = []

    def factory(*, timeout):
        def receive(wire):
            counted.append((str(wire.url), wire.headers.get("x-goog-api-key")))
            return httpx.Response(200, json={"totalTokens": 42})
        return real(timeout=timeout, transport=httpx.MockTransport(receive))

    with monkeypatch.context() as patch:
        patch.setattr(httpx, "AsyncClient", factory)
        budget = await p.request_budget(req, context_estimate=100)
    return budget, counted


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_effective_sdk_key_header_after_count_cannot_change_account(monkeypatch, streaming, record_property):
    p = provider(Capability(), streaming)
    sent = []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        budget, counted = await count(p, req, monkeypatch)
        assert budget["measurement"]["input_tokens"] == 42
        assert counted[0][1] == "offline"
        client._api_client._http_options.headers["x-goog-api-key"] = "other-account"
        with pytest.raises((InvalidRequestError, ProviderUnavailableError)):
            await p.complete(req)
        assert not sent
    finally:
        record_property("media.auth.boundary", json.dumps({
            "scenario": "header-after-count", "streaming": streaming,
            "generation_sends": len(sent),
            "sent_other_account": any(w.headers["x-goog-api-key"] != "offline" for w in sent),
        }))
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", [
    "header", "base_url", "api_key", "http_authorization", "http_cookie", "http_params",
])
async def test_prefreeze_sdk_mismatch_cannot_bless_observed_connection(monkeypatch, field):
    p = provider(Capability())
    sent = []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    api = client._api_client
    if field == "header":
        api._http_options.headers["x-goog-api-key"] = "other-account"
    elif field == "base_url":
        api._http_options.base_url = "https://different.invalid/"
    elif field == "api_key":
        api.api_key = "other-account"
    elif field == "http_authorization":
        api._async_httpx_client.headers["authorization"] = "Bearer other-account"
    elif field == "http_cookie":
        api._async_httpx_client.cookies.set("session", "other-account")
    else:
        api._async_httpx_client.params = {"key": "other-account"}
    try:
        with pytest.raises((InvalidRequestError, ProviderUnavailableError)):
            await p.complete(request())
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["header", "base_url", "api_key", "http_authorization"])
async def test_prefreeze_mismatch_blocks_actual_counter_as_well(monkeypatch, field):
    p = provider(Capability())
    sent = []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    api = client._api_client
    if field == "header":
        api._http_options.headers["x-goog-api-key"] = "other-account"
    elif field == "base_url":
        api._http_options.base_url = "https://different.invalid/"
    elif field == "api_key":
        api.api_key = "other-account"
    else:
        api._async_httpx_client.headers["authorization"] = "Bearer other-account"
    counted = []
    real = httpx.AsyncClient
    def factory(*, timeout):
        def receive(wire):
            counted.append(wire)
            return httpx.Response(200, json={"totalTokens": 42})
        return real(timeout=timeout, transport=httpx.MockTransport(receive))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(httpx, "AsyncClient", factory)
            with pytest.raises(InvalidRequestError):
                await p.request_budget(request(), context_estimate=100)
        assert not sent and not counted
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_received_retry_cannot_follow_changed_effective_key(monkeypatch, record_property):
    p = provider(Capability())
    sent = []

    def receive(wire):
        sent.append(wire.headers.get("x-goog-api-key"))
        client._api_client._http_options.headers["x-goog-api-key"] = "other-account"
        return httpx.Response(503, json={"error": {"code": 503, "message": "received"}})

    client = selected(p, monkeypatch, receive)
    try:
        with pytest.raises((InvalidRequestError, ProviderUnavailableError)):
            await p.complete(request())
        assert sent == ["offline"]
    finally:
        record_property("media.auth.boundary", json.dumps({
            "scenario": "header-after-received-error", "generation_sends": len(sent),
            "sent_other_account": any(key != "offline" for key in sent),
        }))
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["auth", "request_hook"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_owner_http_auth_or_hook_added_after_borrow_refuses_late(monkeypatch, change, streaming):
    p = provider(Capability(), streaming)
    sent, invoked = [], []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        plan = p._request_plan(req)
        models = p._media_models(plan)
        original = client._api_client._async_httpx_client
        if change == "auth":
            original.auth = httpx.BasicAuth("unexpected", "credential")
        else:
            async def hook(wire):
                invoked.append(True)
            original.event_hooks["request"].append(hook)
        with pytest.raises((ValueError, InvalidRequestError)):
            if streaming:
                iterator = await models.generate_content_stream(model=plan.model, contents=deepcopy(plan.contents), config=plan.config)
                async for _ in iterator:
                    pass
            else:
                await models.generate_content(model=plan.model, contents=deepcopy(plan.contents), config=plan.config)
        assert not sent and not invoked
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    "auth", "hook", "response_hook", "header", "http", "api",
    "transport", "mount", "mtls", "userinfo", "query", "host", "cookie",
])
async def test_capability_callback_cannot_change_connection_after_validation(monkeypatch, change):
    cap = Capability()
    p = provider(cap)
    sent = []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        plan = p._request_plan(req)
        models = p._media_models(plan)
        api, http = client._api_client, client._api_client._async_httpx_client
        original_validate = cap.validate
        async def hook(wire):
            pytest.fail("Unqualified hook ran")
        def mutate(value, **kwargs):
            result = original_validate(value, **kwargs)
            if change == "auth":
                http.auth = httpx.BasicAuth("other", "account")
            elif change == "hook":
                http.event_hooks["request"].append(hook)
            elif change == "response_hook":
                http.event_hooks["response"].append(hook)
            elif change == "header":
                api._http_options.headers["x-goog-api-key"] = "other-account"
            elif change == "http":
                api._async_httpx_client = object()
            elif change == "api":
                client._api_client = object()
            elif change == "transport":
                http._transport = httpx.MockTransport(lambda wire: pytest.fail("Replaced transport"))
            elif change == "mount":
                from httpx._utils import URLPattern
                http._mounts[URLPattern("https://")] = httpx.MockTransport(
                    lambda wire: pytest.fail("Replaced mounted transport"))
            elif change == "mtls":
                api._async_httpx_use_mtls_endpoint = True
            elif change == "userinfo":
                api._http_options.base_url = "https://user:secret@generativelanguage.googleapis.com/"
            elif change == "query":
                http.params = {"key": "other-account"}
            elif change == "host":
                http.headers["host"] = "other.invalid"
            elif change == "cookie":
                http.cookies.set("session", "other-account")
            return result
        cap.validate = mutate
        with pytest.raises((ValueError, InvalidRequestError)):
            await models.generate_content(model=plan.model, contents=deepcopy(plan.contents), config=plan.config)
        assert not sent
    finally:
        client._api_client = api
        api._async_httpx_client = http
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["key", "authorization", "cookie", "query", "userinfo", "host"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_concrete_sdk_http_request_is_checked_before_transport(monkeypatch, change, streaming):
    p = provider(Capability(), streaming)
    sent = []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        plan = p._request_plan(req)
        models = p._media_models(plan)
        api = models._api_client
        build = api._build_request
        def alter(*args, **kwargs):
            wire = build(*args, **kwargs)
            if change == "key":
                wire.headers["x-goog-api-key"] = "other-account"
            elif change == "authorization":
                wire.headers["authorization"] = "Bearer other-account"
            elif change == "cookie":
                wire.headers["cookie"] = "session=other-account"
            elif change == "query":
                wire.url = str(httpx.URL(wire.url).copy_add_param("key", "other-account"))
            elif change == "userinfo":
                wire.url = str(httpx.URL(wire.url).copy_with(username="user", password="secret"))
            else:
                wire.headers["host"] = "other.invalid"
            return wire
        api._build_request = alter
        with pytest.raises(ValueError, match="effective HTTP request"):
            if streaming:
                iterator = await models.generate_content_stream(model=plan.model, contents=deepcopy(plan.contents), config=plan.config)
                async for _ in iterator:
                    pass
            else:
                await models.generate_content(model=plan.model, contents=deepcopy(plan.contents), config=plan.config)
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_private_counter_actual_auth_callback_is_also_guarded(monkeypatch):
    cap = Capability()
    p = provider(cap)
    sent, counted, counters = [], [], []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    p._request_plan(req)
    real = httpx.AsyncClient
    def factory(*, timeout):
        def receive(wire):
            counted.append(wire)
            return httpx.Response(200, json={"totalTokens": 42})
        counter = real(timeout=timeout, transport=httpx.MockTransport(receive))
        counters.append(counter)
        return counter
    def mutate(value, **kwargs):
        if counters:
            counters[0].auth = httpx.BasicAuth("other", "account")
        return {"valid": True}
    cap.validate = mutate
    try:
        with monkeypatch.context() as patch:
            patch.setattr(httpx, "AsyncClient", factory)
            # Refusal is represented as unavailable or a fixed safe error;
            # it must never produce a native measurement or a physical send.
            budget = await p.request_budget(req, context_estimate=100)
        assert budget is None
        assert not sent and not counted
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_selected_connection_positive_real_sdk_count_send_and_copy(monkeypatch, streaming, record_property):
    p = provider(Capability(), streaming)
    sent = []
    def receive(wire):
        sent.append(wire)
        if streaming:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps(response()) + "\n\n")
        return httpx.Response(200, json=response())
    client = selected(p, monkeypatch, receive)
    req = request()
    headers = deepcopy(client._api_client._http_options.headers)
    try:
        budget, counted = await count(p, req, monkeypatch)
        assert budget["measurement"]["input_tokens"] == 42
        plan = p._request_plan(req)
        models = p._media_models(plan)
        assert models._api_client._http_options is not client._api_client._http_options
        assert models._api_client._http_options.headers is not client._api_client._http_options.headers
        assert await p.complete(req)
        assert len(sent) == len(counted) == 1
        assert sent[0].headers["x-goog-api-key"] == counted[0][1] == "offline"
        assert sent[0].url.host == httpx.URL(counted[0][0]).host == "generativelanguage.googleapis.com"
        assert client._api_client._http_options.headers == headers
        assert not p._media_plans
        record_property("media.connection", json.dumps({
            "sdk": "2.30.1", "streaming": streaming, "count_sends": len(counted),
            "generation_sends": len(sent), "account_route_equal": True,
            "options_isolated": True,
        }))
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_selected_custom_generation_route_does_not_borrow_fixed_counter(monkeypatch):
    p = provider(Capability())
    sent = []
    with monkeypatch.context() as patch:
        patch.setenv("GOOGLE_GEMINI_BASE_URL", "https://offline.invalid/")
        client = selected(p, patch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    p._client_uses_canonical_developer_route = True  # A stale marker cannot bless this.
    req = request()
    try:
        budget, counted = await count(p, req, monkeypatch)
        assert budget is None and not counted
        assert await p.complete(req)
        assert len(sent) == 1 and sent[0].url.host == "offline.invalid"
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_owner_execution_override_cannot_shadow_borrowed_guards(monkeypatch, streaming):
    p = provider(Capability(), streaming)
    sent = []
    def receive(wire):
        sent.append(wire.headers.get("x-goog-api-key"))
        if streaming:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps(response()) + "\n\n")
        return httpx.Response(200, json=response())
    client = selected(p, monkeypatch, receive)
    req = request()
    try:
        budget, counted = await count(p, req, monkeypatch)
        assert budget["measurement"]["input_tokens"] == 42 and len(counted) == 1
        http = client._api_client._async_httpx_client
        if streaming:
            async def send(wire, **kwargs):
                wire.headers["x-goog-api-key"] = "other-account"
                return await httpx.AsyncClient.send(http, wire, **kwargs)
            http.send = send
        else:
            async def send_request(*args, **kwargs):
                kwargs["headers"]["x-goog-api-key"] = "other-account"
                return await httpx.AsyncClient.request(http, *args, **kwargs)
            http.request = send_request
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            await p.complete(req)
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", [
    "build_request", "_send_single_request", "_send_handling_auth",
    "_send_handling_redirects", "_transport_for_url", "_build_request_auth",
])
async def test_owner_other_execution_extensions_are_not_borrowed(monkeypatch, method):
    p = provider(Capability())
    sent = []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        p._request_plan(req)
        def forbidden(*args, **kwargs):
            pytest.fail("Owner execution extension was called")
        setattr(client._api_client._async_httpx_client, method, forbidden)
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            await p.complete(req)
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_request", "_async_request_once", "_build_request"])
async def test_owner_sdk_execution_overrides_are_not_borrowed(monkeypatch, method):
    p = provider(Capability())
    sent = []
    client = selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        p._request_plan(req)
        def forbidden(*args, **kwargs):
            pytest.fail("Owner SDK execution extension was called")
        setattr(client._api_client, method, forbidden)
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            await p.complete(req)
        assert not sent
    finally:
        await close(client)
