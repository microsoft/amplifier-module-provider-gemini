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


def finite_selected(p, monkeypatch, receive, *, transport=None, mounted=False):
    """Use provider config and the SDK's real constructor/default selection."""
    constructor = genai.Client
    def factory(**kwargs):
        args = {"transport": httpx.MockTransport(receive)}
        if transport is not None:
            if mounted:
                args["mounts"] = {"https://generativelanguage.googleapis.com": transport}
            else:
                args["transport"] = transport
        kwargs["http_options"] = kwargs["http_options"].model_copy(
            update={"async_client_args": args})
        return constructor(**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(genai, "Client", factory)
        return p.client


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("text_phase", ["before-count", "after-count"])
async def test_finite_timeout_text_then_media_preserves_selected_count_send(monkeypatch, streaming, text_phase):
    p = provider(Capability(), streaming)
    p.timeout = 7.25
    sent = []
    def receive(wire):
        sent.append(wire)
        if wire.url.path.endswith(":streamGenerateContent"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps(response()) + "\n\n")
        return httpx.Response(200, json=response())
    client = finite_selected(p, monkeypatch, receive)
    req = request()
    try:
        assert client._api_client._http_options.timeout == 7250
        if text_phase == "after-count":
            budget, counted = await count(p, req, monkeypatch)
            assert budget["measurement"]["input_tokens"] == 42 and len(counted) == 1
        # This is an ordinary SDK generation, not a media borrowed view. It
        # legitimately adds the derived timeout header to the owning options.
        await client.aio.models.generate_content(model=p.default_model, contents="ordinary text")
        assert client._api_client._http_options.headers["X-Server-Timeout"] == "8"
        if text_phase == "before-count":
            budget, counted = await count(p, req, monkeypatch)
            assert budget["measurement"]["input_tokens"] == 42 and len(counted) == 1
        assert await p.complete(req)
        assert len(sent) == 2
        assert all(w.headers["x-goog-api-key"] == "offline" for w in sent)
        assert sent[-1].headers["x-server-timeout"] == "8"
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("phase", ["initial", "after-count", "after-borrow"])
async def test_finite_production_transport_execution_wrap_is_refused(monkeypatch, mounted, phase):
    p = provider(Capability())
    invoked = []
    transport = httpx.AsyncHTTPTransport()
    async def wrapped(wire):
        invoked.append(wire)
        wire.headers["x-goog-api-key"] = "other-account"
        return httpx.Response(200, json=response())
    if phase == "initial":
        transport.handle_async_request = wrapped
    client = finite_selected(p, monkeypatch, lambda wire: httpx.Response(200, json=response()),
                             transport=transport, mounted=mounted)
    req = request()
    try:
        if phase == "after-count":
            budget, counted = await count(p, req, monkeypatch)
            assert budget["measurement"]["input_tokens"] == 42 and len(counted) == 1
        elif phase == "after-borrow":
            plan = p._request_plan(req)
            models = p._media_models(plan)
        if phase != "initial":
            transport.handle_async_request = wrapped
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            if phase == "after-borrow":
                await models.generate_content(model=plan.model, contents=plan.contents, config=plan.config)
            else:
                await p.complete(req)
        assert not invoked  # Wrapper must not see the private image request.
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before-count", "after-count", "after-borrow"])
async def test_finite_closed_owner_refuses_without_private_transport(monkeypatch, phase):
    p = provider(Capability())
    sent = []
    client = finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        if phase == "after-count":
            budget, counted = await count(p, req, monkeypatch)
            assert budget["measurement"]["input_tokens"] == 42 and len(counted) == 1
        elif phase == "after-borrow":
            plan = p._request_plan(req)
            models = p._media_models(plan)
        await client.aio.aclose()
        assert client._api_client._async_httpx_client.is_closed
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            if phase == "before-count":
                await count(p, req, monkeypatch)
            elif phase == "after-borrow":
                await models.generate_content(model=plan.model, contents=plan.contents, config=plan.config)
            else:
                await p.complete(req)
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", [None, "https://env-only.invalid/"])
async def test_finite_sdk_selected_default_base_precedes_env_and_stays_generation_only(monkeypatch, environment):
    from google.genai import _base_url
    p = provider(Capability())
    sent = []
    old = (_base_url._default_base_gemini_url, _base_url._default_base_vertex_url)
    try:
        if environment:
            monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", environment)
        else:
            monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
        _base_url.set_default_base_urls("https://sdk-selected.invalid/", None)
        client = finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
        try:
            assert client._api_client._http_options.base_url == "https://sdk-selected.invalid/"
            assert p._client_uses_canonical_developer_route is False
            budget, counted = await count(p, request(), monkeypatch)
            assert budget is None and not counted
            assert await p.complete(request())
            assert len(sent) == 1 and sent[0].url.host == "sdk-selected.invalid"
        finally:
            await close(client)
    finally:
        _base_url.set_default_base_urls(*old)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ImportError, KeyError])
async def test_finite_private_sdk_unavailable_does_not_break_ordinary_text(monkeypatch, error):
    from amplifier_module_provider_gemini import _media
    p = provider(None)
    sent = []
    def unavailable(*args, **kwargs):
        raise error("private SDK adapter unavailable")
    monkeypatch.setattr(_media, "MediaConnection", unavailable)
    try:
        finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
        assert await p.complete(ChatRequest(messages=[Message(role="user", content="ordinary")]))
        assert len(sent) == 1
        p.coordinator.get_capability = lambda key: Capability()
        with pytest.raises(InvalidRequestError, match="connection refused"):
            await p.complete(request())
        assert len(sent) == 1
    finally:
        if p._client is not None:
            await close(p._client)


@pytest.mark.asyncio
async def test_finite_missing_httpx_data_field_safe_media_refusal(monkeypatch):
    p = provider(Capability())
    sent = []
    client = finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    http = client._api_client._async_httpx_client
    old = http.__dict__.pop("_default_encoding")
    try:
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            await p.complete(request())
        assert not sent
    finally:
        http._default_encoding = old
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_finite_normal_production_transport_pool_stays_owner_controlled(monkeypatch, mounted, streaming):
    import httpcore
    from amplifier_module_provider_gemini._media import AttemptHTTP
    sent, streams = [], []
    body = json.dumps(response()).encode()
    if streaming:
        body = b"data: " + body + b"\n\n"
    content_type = b"text/event-stream" if streaming else b"application/json"
    reply = (b"HTTP/1.1 200 OK\r\nContent-Type: " + content_type
             + b"\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    class Stream(httpcore.AsyncMockStream):
        async def write(self, buffer, timeout=None):
            sent.append(buffer)
    class Backend(httpcore.AsyncMockBackend):
        async def connect_tcp(self, *args, **kwargs):
            stream = Stream([reply])
            streams.append(stream)
            return stream
    transport = httpx.AsyncHTTPTransport()
    real_pool = transport._pool
    transport._pool = httpcore.AsyncConnectionPool(network_backend=Backend([]))
    p = provider(Capability(), streaming)
    client = finite_selected(p, monkeypatch, lambda wire: pytest.fail("Wrong transport"),
                             transport=transport, mounted=mounted)
    req = request()
    try:
        plan = p._request_plan(req)
        models = p._media_models(plan)
        view = models._api_client._async_httpx_client
        assert isinstance(view, AttemptHTTP)
        # Starting UNOPENED and becoming OPENED is normal, not an identity change.
        assert not client._api_client._async_httpx_client.is_closed
        assert await p.complete(req)
        assert len(streams) == 1
        assert b"x-goog-api-key: offline" in b"".join(sent)
        await view.aclose()
        await view.__aexit__(None, None, None)
        assert not streams[0]._closed
        await client.aio.aclose()
        assert streams[0]._closed
    finally:
        client.close()
        await real_pool.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [
    "timeout-value", "timeout-option", "timeout-duplicate", "key", "base",
])
async def test_finite_timeout_normalization_does_not_ignore_other_changes(monkeypatch, mutation):
    p = provider(Capability())
    p.timeout = 7.25
    sent = []
    client = finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        budget, _ = await count(p, req, monkeypatch)
        assert budget["measurement"]["input_tokens"] == 42
        opts = client._api_client._http_options
        if mutation == "timeout-value":
            opts.headers["X-Server-Timeout"] = "999"
        elif mutation == "timeout-option":
            opts.timeout = 9000
        elif mutation == "timeout-duplicate":
            opts.headers["X-Server-Timeout"] = "8"
            opts.headers["x-server-timeout"] = "999"
        elif mutation == "key":
            opts.headers["X-Server-Timeout"] = "8"
            opts.headers["x-goog-api-key"] = "other-account"
        else:
            opts.headers["X-Server-Timeout"] = "8"
            opts.base_url = "https://changed.invalid/"
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            await p.complete(req)
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_finite_missing_new_private_httpx_data_safe_text_and_media(monkeypatch):
    from amplifier_module_provider_gemini import _media
    p = provider(None)
    sent = []
    monkeypatch.setattr(_media, "_HTTP_DATA", _media._HTTP_DATA | {"_future_missing"})
    client = finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    try:
        assert await p.complete(ChatRequest(messages=[Message(role="user", content="ordinary")]))
        assert len(sent) == 1
        p.coordinator.get_capability = lambda key: Capability()
        with pytest.raises(InvalidRequestError, match="connection refused"):
            await p.complete(request())
        assert len(sent) == 1
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_finite_explicit_timeout_header_remains_immutable(monkeypatch):
    p = provider(Capability())
    p.timeout = 7.25
    sent = []
    constructor = genai.Client
    def factory(**kwargs):
        kwargs["http_options"] = kwargs["http_options"].model_copy(update={
            "headers": {"X-Server-Timeout": "8"},
            "async_client_args": {"transport": httpx.MockTransport(
                lambda wire: sent.append(wire) or httpx.Response(200, json=response()))},
        })
        return constructor(**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(genai, "Client", factory)
        client = p.client
    req = request()
    try:
        budget, _ = await count(p, req, monkeypatch)
        assert budget["measurement"]["input_tokens"] == 42
        client._api_client._http_options.headers.pop("X-Server-Timeout")
        with pytest.raises((InvalidRequestError, ProviderUnavailableError)):
            await p.complete(req)
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_finite_sdk_canonical_default_overrides_custom_environment(monkeypatch):
    from google.genai import _base_url
    old = (_base_url._default_base_gemini_url, _base_url._default_base_vertex_url)
    p = provider(Capability())
    sent = []
    try:
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://env-only.invalid/")
        _base_url.set_default_base_urls("https://generativelanguage.googleapis.com/", None)
        assert p._native_counting_available(p.default_model)
        client = finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
        try:
            assert p._client_uses_canonical_developer_route is True
            req = request()
            budget, counted = await count(p, req, monkeypatch)
            assert budget["measurement"]["input_tokens"] == 42 and len(counted) == 1
            assert await p.complete(req)
            assert len(sent) == 1 and sent[0].url.host == "generativelanguage.googleapis.com"
        finally:
            await close(client)
    finally:
        _base_url.set_default_base_urls(*old)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mock-instance", "mock-subclass"])
async def test_finite_mock_transport_is_not_an_execution_override_exemption(monkeypatch, kind):
    p = provider(Capability())
    invoked = []
    async def unsafe(wire):
        invoked.append(wire)
        return httpx.Response(200, json=response())
    if kind == "mock-instance":
        transport = httpx.MockTransport(lambda wire: pytest.fail("Not reached"))
        transport.handle_async_request = unsafe
    else:
        class Subclass(httpx.MockTransport):
            async def handle_async_request(self, wire):
                return await unsafe(wire)
        transport = Subclass(lambda wire: pytest.fail("Not reached"))
    client = finite_selected(p, monkeypatch, lambda wire: pytest.fail("Not reached"), transport=transport)
    try:
        with pytest.raises((InvalidRequestError, ProviderUnavailableError)):
            await p.complete(request())
        assert not invoked
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ImportError, KeyError])
async def test_finite_private_borrow_failure_is_sanitized_refusal(monkeypatch, error):
    from amplifier_module_provider_gemini import image_operation
    p = provider(Capability())
    sent = []
    client = finite_selected(p, monkeypatch, lambda wire: sent.append(wire) or httpx.Response(200, json=response()))
    req = request()
    try:
        plan = p._request_plan(req)
        def unavailable(*args, **kwargs):
            raise error("private SDK sentinel must not escape")
        monkeypatch.setattr(image_operation, "_compatibility_client", unavailable)
        with pytest.raises(InvalidRequestError, match="SDK transport unavailable") as caught:
            p._media_models(plan)
        assert "sentinel" not in str(caught.value)
        assert not sent
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("phase", ["initial", "after-count", "after-borrow"])
async def test_finite_pool_execution_override_refused_before_private_wrapper(monkeypatch, mounted, streaming, phase):
    import httpcore
    p = provider(Capability(), streaming)
    transport = httpx.AsyncHTTPTransport()
    pool = transport._pool
    invoked = []
    async def wrapper(core_request):
        invoked.append(True)
        core_request.headers = [(k, b"other-account" if k == b"x-goog-api-key" else v)
                                for k, v in core_request.headers]
        body = json.dumps(response()).encode()
        if streaming:
            body = b"data: " + body + b"\n\n"
        async def chunks():
            yield body
        headers = [(b"content-type", b"text/event-stream")] if streaming else []
        return httpcore.Response(200, headers=headers, content=chunks())
    if phase == "initial":
        pool.handle_async_request = wrapper
    client = finite_selected(p, monkeypatch, lambda wire: pytest.fail("Wrong transport"),
                             transport=transport, mounted=mounted)
    req = request()
    try:
        if phase == "after-count":
            budget, counted = await count(p, req, monkeypatch)
            assert budget["measurement"]["input_tokens"] == 42 and len(counted) == 1
        elif phase == "after-borrow":
            plan = p._request_plan(req)
            models = p._media_models(plan)
        pool.handle_async_request = wrapper
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            if phase in ("initial", "after-count"):
                await p.complete(req)
            elif streaming:
                iterator = await models.generate_content_stream(model=plan.model, contents=plan.contents, config=plan.config)
                async for _ in iterator:
                    pass
            else:
                await models.generate_content(model=plan.model, contents=plan.contents, config=plan.config)
        assert not invoked
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
async def test_finite_pool_identity_cannot_be_replaced_after_count(monkeypatch, mounted):
    import httpcore
    p = provider(Capability())
    transport = httpx.AsyncHTTPTransport()
    old_pool = transport._pool
    client = finite_selected(p, monkeypatch, lambda wire: pytest.fail("Wrong transport"),
                             transport=transport, mounted=mounted)
    req = request()
    try:
        budget, _ = await count(p, req, monkeypatch)
        assert budget["measurement"]["input_tokens"] == 42
        transport._pool = httpcore.AsyncConnectionPool(network_backend=httpcore.AsyncMockBackend([]))
        with pytest.raises((InvalidRequestError, ProviderUnavailableError, ValueError)):
            await p.complete(req)
    finally:
        await close(client)
        await old_pool.aclose()
