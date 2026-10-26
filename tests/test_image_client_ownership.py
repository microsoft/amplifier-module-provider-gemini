"""Offline ownership regressions for real SDK image helper views.

Run unchanged against google-genai 2.30.1 and 2.33.0. SDK construction,
resource views, and destructors are real; HTTP traffic uses MockTransport.
The optional aiohttp cleanup registry receives a counting session double,
not a real network session. No SDK ownership flags or destructors are patched.
"""

import asyncio
import gc
import weakref

import httpx
import pytest
from google import genai
from google.genai import types

from amplifier_module_provider_gemini.image_operation import (
    _BoundHTTP,
    _compatibility_client,
    _interaction_client,
)


class _CountingTransport(httpx.MockTransport):
    def __init__(self):
        super().__init__(self._respond)
        self.requests = 0
        self.closes = 0

    def _respond(self, request):
        self.requests += 1
        return httpx.Response(200, text="owner remains usable")

    def close(self):
        self.closes += 1
        super().close()

    async def aclose(self):
        self.closes += 1
        await super().aclose()


class _CountingSession:
    """Only the aiohttp close protocol is exercised, with no aiohttp dependency."""

    def __init__(self):
        self.closes = 0
        self.closed = False

    async def close(self):
        self.closes += 1
        self.closed = True


@pytest.fixture(autouse=True)
def _offline_sdk_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "false")
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)


def _seed_owned_session(api, session):
    # Emulate an already-open chat session. The older SDK has one slot;
    # newer SDKs keep per-loop slots. Neither is a custom-client ownership flag.
    if hasattr(api, "_aiohttp_sessions"):
        sessions = api._aiohttp_sessions
        assert isinstance(sessions, dict) and not sessions
        sessions[asyncio.get_running_loop()] = session
    else:
        assert hasattr(api, "_aiohttp_session"), "SDK session layout needs qualification"
        assert api._aiohttp_session is None
        api._aiohttp_session = session


async def _collect_and_drain(baseline):
    """Collect view cycles and observe only tasks created during collection."""
    errors = []

    def observe(task):
        if task.cancelled():
            errors.append("unexpected cleanup cancellation")
        else:
            error = task.exception()
            if error is not None:
                errors.append(repr(error))

    # Bounded rounds also collect copies kept alive by their destructor task.
    for _ in range(3):
        gc.collect()
        tasks = asyncio.all_tasks() - baseline
        for task in tasks:
            task.add_done_callback(observe)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=1.0)
            for task in pending:
                task.cancel()
            assert not pending, "SDK view cleanup did not drain within one second"
        await asyncio.sleep(0)  # Run scheduled close callbacks, not a timed delay.
    assert not errors, f"SDK view cleanup raised: {errors}"


async def _discard_view(owner, helper, transport):
    if helper == "compatibility":
        view = _compatibility_client(owner, transport)
    else:
        view = _interaction_client(owner, 1.0, transport)
    reference = weakref.ref(view)
    # A non-cyclic view can schedule cleanup immediately on del, before gc().
    baseline = asyncio.all_tasks()
    del view
    await _collect_and_drain(baseline)
    assert reference() is None, "test retained the image resource view"


async def _discard_image_view(owner, helper, binding):
    if binding == "bound":
        async with _BoundHTTP(owner._api_client._async_httpx_client) as transport:
            await _discard_view(owner, helper, transport)
    else:
        await _discard_view(owner, helper, None)


async def _close_owner(owner):
    # Both transports are mock-backed. Keep even fixture cleanup time-bounded.
    await asyncio.wait_for(asyncio.to_thread(owner.close), timeout=1.0)
    await asyncio.wait_for(owner.aio.aclose(), timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["compatibility", "interactions"])
@pytest.mark.parametrize("binding", ["direct", "bound"])
async def test_image_view_gc_preserves_sdk_owned_clients_and_sessions(helper, binding):
    sync_transport = _CountingTransport()
    async_transport = _CountingTransport()
    # Transport arguments do NOT supply custom clients: the SDK constructs and
    # owns both HTTPX clients, while every possible HTTP send stays in memory.
    owner = genai.Client(
        api_key="offline-ownership-placeholder",
        vertexai=False,
        http_options=types.HttpOptions(
            client_args={"transport": sync_transport, "trust_env": False},
            async_client_args={"transport": async_transport, "trust_env": False},
        ),
    )
    session = _CountingSession()
    try:
        api = owner._api_client
        sync_http = api._httpx_client
        async_http = api._async_httpx_client
        _seed_owned_session(api, session)
        assert not sync_http.is_closed and not async_http.is_closed
        assert (sync_transport.closes, async_transport.closes, session.closes) == (0, 0, 0)

        await _discard_image_view(owner, helper, binding)

        assert (sync_transport.requests, async_transport.requests) == (0, 0)
        observed = (sync_transport.closes, async_transport.closes, session.closes)
        assert observed == (0, 0, 0), (
            "discarding an image view closed owner resources "
            f"(sync HTTPX, async HTTPX, aiohttp session): {observed}"
        )
        assert not sync_http.is_closed and not async_http.is_closed
        assert not session.closed
        assert sync_http.get("https://fixture.invalid/owner").status_code == 200
        response = await asyncio.wait_for(
            async_http.get("https://fixture.invalid/owner"), timeout=1.0
        )
        assert response.status_code == 200
        assert (sync_transport.requests, async_transport.requests) == (1, 1)

        await _close_owner(owner)

        assert sync_http.is_closed and async_http.is_closed and session.closed
        assert (
            sync_transport.closes,
            async_transport.closes,
            session.closes,
        ) == (1, 1, 1)
    finally:
        await _close_owner(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["compatibility", "interactions"])
@pytest.mark.parametrize("binding", ["direct", "bound"])
async def test_image_view_and_owner_leave_custom_httpx_clients_to_caller(helper, binding):
    sync_transport = _CountingTransport()
    async_transport = _CountingTransport()
    sync_http = httpx.Client(transport=sync_transport, trust_env=False)
    async_http = httpx.AsyncClient(transport=async_transport, trust_env=False)
    owner = None
    try:
        owner = genai.Client(
            api_key="offline-ownership-placeholder",
            vertexai=False,
            http_options=types.HttpOptions(
                httpx_client=sync_http,
                httpx_async_client=async_http,
            ),
        )
        assert owner._api_client._httpx_client is sync_http
        assert owner._api_client._async_httpx_client is async_http

        await _discard_image_view(owner, helper, binding)

        assert (sync_transport.requests, async_transport.requests) == (0, 0)
        assert (sync_transport.closes, async_transport.closes) == (0, 0)
        assert not sync_http.is_closed and not async_http.is_closed

        await _close_owner(owner)

        assert (sync_transport.closes, async_transport.closes) == (0, 0)
        assert not sync_http.is_closed and not async_http.is_closed
        assert sync_http.get("https://fixture.invalid/caller").status_code == 200
        response = await asyncio.wait_for(
            async_http.get("https://fixture.invalid/caller"), timeout=1.0
        )
        assert response.status_code == 200
        assert (sync_transport.requests, async_transport.requests) == (1, 1)

        sync_http.close()
        await asyncio.wait_for(async_http.aclose(), timeout=1.0)
        assert sync_http.is_closed and async_http.is_closed
        assert (sync_transport.closes, async_transport.closes) == (1, 1)
    finally:
        if owner is not None:
            await _close_owner(owner)
        sync_http.close()
        await asyncio.wait_for(async_http.aclose(), timeout=1.0)


@pytest.mark.asyncio
async def test_borrowed_api_explicit_cleanup_leaves_resources_for_owner():
    from amplifier_module_provider_gemini.image_operation import _borrow_api_client

    sync_transport = _CountingTransport()
    async_transport = _CountingTransport()
    owner = genai.Client(
        api_key="offline-ownership-fixture",
        vertexai=False,
        http_options=types.HttpOptions(
            client_args={"transport": sync_transport, "trust_env": False},
            async_client_args={"transport": async_transport, "trust_env": False},
        ),
    )
    session = _CountingSession()
    try:
        api = owner._api_client
        _seed_owned_session(api, session)
        view = _borrow_api_client(owner)
        assert view.__dict__ is not api.__dict__
        assert view._borrowed_owner is owner
        assert view._httpx_client is api._httpx_client
        assert view._async_httpx_client is api._async_httpx_client
        assert view.request.__self__ is view
        assert view.async_request.__self__ is view
        view._http_options = view._http_options.model_copy(update={"timeout": 1250})
        assert api._http_options.timeout != 1250

        assert view.close() is None
        assert await view.aclose() is None
        baseline = asyncio.all_tasks()
        view.__del__()
        assert asyncio.all_tasks() == baseline
        assert (sync_transport.closes, async_transport.closes, session.closes) == (0, 0, 0)
        assert not api._httpx_client.is_closed and not api._async_httpx_client.is_closed

        await _close_owner(owner)

        assert (sync_transport.closes, async_transport.closes, session.closes) == (1, 1, 1)
        assert api._httpx_client.is_closed and api._async_httpx_client.is_closed
        assert session.closed
    finally:
        await _close_owner(owner)


@pytest.mark.asyncio
async def test_borrowed_api_retains_top_level_owner_until_view_is_discarded():
    from amplifier_module_provider_gemini.image_operation import _borrow_api_client

    owner = genai.Client(
        api_key="offline-ownership-fixture",
        vertexai=False,
        http_options=types.HttpOptions(
            client_args={"transport": _CountingTransport(), "trust_env": False},
            async_client_args={"transport": _CountingTransport(), "trust_env": False},
        ),
    )
    reference = weakref.ref(owner)
    view = _borrow_api_client(owner)
    try:
        del owner
        await _collect_and_drain(asyncio.all_tasks())
        assert reference() is not None
        assert view._borrowed_owner is reference()
        assert not view._httpx_client.is_closed
        assert not view._async_httpx_client.is_closed
        # Close through the retained owner, not the borrowed lease.
        await _close_owner(reference())
    finally:
        if reference() is not None:
            await _close_owner(reference())
        del view
        await _collect_and_drain(asyncio.all_tasks())
    assert reference() is None


def test_borrowed_api_rejects_unqualified_client_state():
    from types import SimpleNamespace

    from amplifier_module_provider_gemini.image_operation import _borrow_api_client

    with pytest.raises(TypeError, match="Unsupported SDK API client state"):
        _borrow_api_client(SimpleNamespace(_api_client=SimpleNamespace()))


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["interactions", "generate_content"])
@pytest.mark.parametrize("state", ["wrong_base", "missing_state"])
@pytest.mark.parametrize("construction", ["prebuilt", "lazy"])
async def test_unqualified_api_state_refuses_preparation_without_sending(
    transport, state, construction, monkeypatch
):
    from types import SimpleNamespace

    from amplifier_module_provider_gemini import GeminiProvider
    from amplifier_module_provider_gemini.image_operation import PROTOCOL

    sync_transport = _CountingTransport()
    async_transport = _CountingTransport()
    key = "offline-ownership-fixture"
    owner = genai.Client(
        api_key=key,
        vertexai=False,
        http_options=types.HttpOptions(
            client_args={"transport": sync_transport, "trust_env": False},
            async_client_args={"transport": async_transport, "trust_env": False},
        ),
    )
    api = owner._api_client
    original_state = api.__dict__.copy()
    provider = GeminiProvider(api_key=key)
    intent = {
        "protocol": PROTOCOL,
        "request_id": "unsupported-state",
        "operation": "generate",
        "prompt": "A blue square",
        "inputs": [],
        "selection": {
            "model": "gemini-3.8-flash",
            "config": {
                "image_generation": {
                    "model": "gemini-nano-banana-2.1",
                    "transport": transport,
                }
            },
        },
        "output": {
            "count": 1,
            "mime_type": "image/png",
            "size": "auto",
            "quality": "auto",
            "background": "auto",
        },
    }
    try:
        if state == "wrong_base":
            owner._api_client = SimpleNamespace(**original_state)
        else:
            del api.__dict__["_httpx_client"]
        if construction == "prebuilt":
            provider._client = provider._count_route_client = owner
            provider._client_uses_canonical_developer_route = True
        else:
            monkeypatch.setattr(genai, "Client", lambda **kwargs: owner)

        preparation = await provider.prepare_image_generation(intent)

        assert preparation["supported"] is False
        assert preparation["reason_code"] == "unsupported_sdk"
        assert "plan" not in preparation
        assert not getattr(provider, "_prepared_image_operations", {})
        descriptor = provider.get_image_generation_capabilities()
        assert descriptor["supported"] is False
        assert descriptor["unavailable_transports"] == {
            "interactions": "unsupported_sdk",
            "generate_content": "unsupported_sdk",
        }
        assert (sync_transport.requests, async_transport.requests) == (0, 0)
    finally:
        owner._api_client = api
        api.__dict__.update(original_state)
        await _close_owner(owner)