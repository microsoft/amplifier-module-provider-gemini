"""Tests for GeminiProvider.close() method.

recipes-89e -- close() must ACTUALLY close the genai client's transports,
bounded, off the event loop.

Before this, close() only did `self._client = None`: the client's httpx
transports were never closed, merely dereferenced, and their sockets waited
on garbage collection. The trap the fix has to avoid is equally real --
`google.genai.Client.close()` is SYNCHRONOUS, so calling it inline from an
async close() would block the event loop for the duration of the transport
shutdown, trading a leak for a stall.

`google.genai.Client` holds TWO transports, both built eagerly by the
constructor and each with its own close (verified against google-genai
1.56.0 -- this module's dependency floor -- and 2.22.0):

  - async: `await client.aio.aclose()`   <-- the one this provider uses
  - sync:  `client.close()`              <-- synchronous; must be offloaded

Every API call in this module goes through `self.client.aio.*`, so closing
only the sync surface would leave the real sockets uncollected. The tests
below drive both, and drive the real bound rather than a mocked-out one.
"""

import asyncio
import logging
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_gemini import _DEFAULT_CLOSE_TIMEOUT
from amplifier_module_provider_gemini import GeminiProvider


def _fake_client(aclose=None, close=None) -> MagicMock:
    """A stand-in with the same close surface as google.genai.Client."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.aclose = aclose if aclose is not None else AsyncMock()
    client.close = close if close is not None else MagicMock()
    return client


# ---------------------------------------------------------------------------
# The leak itself: a real client's transports must actually be closed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_actually_closes_a_real_genai_clients_transports():
    """The defect, driven against a REAL google.genai.Client.

    No network: both httpx transports are constructed eagerly by
    `Client(...)`, so `is_closed` is observable without ever making a call.
    Asserting on `is_closed` (not on a mock call) is what distinguishes
    "we called something named close" from "the transport is shut".
    """
    from google import genai

    provider = GeminiProvider(api_key="k")
    client = genai.Client(api_key="k")
    provider._client = client

    api = client._api_client
    sync_transport = api._httpx_client
    async_transport = api._async_httpx_client
    # Precondition: both open, i.e. the leak is real and observable here.
    assert sync_transport.is_closed is False
    assert async_transport.is_closed is False

    await provider.close()

    assert async_transport.is_closed is True, (
        "the ASYNC transport -- the one every self.client.aio.* call in this "
        "module uses -- was left open"
    )
    assert sync_transport.is_closed is True, "the sync transport was left open"
    assert provider._client is None


@pytest.mark.asyncio
async def test_close_calls_both_close_surfaces():
    """Both the async aclose and the sync close are invoked, exactly once."""
    provider = GeminiProvider(api_key="k")
    client = _fake_client()
    provider._client = client

    await provider.close()

    client.aio.aclose.assert_awaited_once()
    client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Never block the event loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_close_runs_off_the_event_loop():
    """The SDK's synchronous close() must not stall the loop while it runs.

    A sync close that occupies its thread for ~0.3s is the whole hazard:
    called inline it freezes every other task in the process. A ticker task
    running concurrently proves the loop stayed live -- inline execution
    would let through at most one tick.
    """
    release = threading.Event()
    entered = threading.Event()

    def _blocking_close():
        entered.set()
        release.wait(10)  # bounded so a failure can't wedge the suite

    provider = GeminiProvider(api_key="k", config={"close_timeout": 30.0})
    provider._client = _fake_client(close=MagicMock(side_effect=_blocking_close))

    ticks = 0

    async def _ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(_ticker())
    try:
        close_task = asyncio.create_task(provider.close())
        # Let the worker thread get into the blocking call, then let the
        # loop run while it is still blocked.
        await asyncio.sleep(0.3)
        assert entered.is_set(), "the sync close never started"
        assert ticks >= 5, (
            f"event loop only ticked {ticks} times while the synchronous "
            "close was running -- it is being called inline, not offloaded"
        )
        release.set()
        await asyncio.wait_for(close_task, timeout=5.0)
    finally:
        release.set()
        ticker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ticker


# ---------------------------------------------------------------------------
# The hard bound.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_returns_within_bound_when_async_close_hangs(caplog):
    """An aclose() that never returns must not hold the caller past the bound."""

    async def _hang():
        await asyncio.Event().wait()  # never set

    provider = GeminiProvider(api_key="k", config={"close_timeout": 0.25})
    provider._client = _fake_client(aclose=MagicMock(side_effect=lambda: _hang()))

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(provider.close(), timeout=5.0)
    elapsed = time.monotonic() - started

    assert 0.25 <= elapsed < 2.0, f"close() took {elapsed:.2f}s, expected ~0.25s"

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    abandoned = [m for m in warnings if "abandoning the genai client" in m]
    assert len(abandoned) == 1, f"expected one abandonment warning, got: {warnings}"
    # The warning must name the instance so a hang in one of several mounted
    # providers is attributable.
    assert f"id=0x{id(provider):x}" in abandoned[0]
    assert "gemini" in abandoned[0]
    # The slot is cleared even though the close never finished.
    assert provider._client is None


@pytest.mark.asyncio
async def test_close_returns_within_bound_when_sync_close_hangs(caplog):
    """A wedged SYNCHRONOUS close is bounded too -- the thread is abandoned."""
    release = threading.Event()

    def _blocking_close():
        release.wait(10)

    provider = GeminiProvider(api_key="k", config={"close_timeout": 0.25})
    provider._client = _fake_client(close=MagicMock(side_effect=_blocking_close))

    started = time.monotonic()
    try:
        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(provider.close(), timeout=5.0)
        elapsed = time.monotonic() - started

        assert 0.25 <= elapsed < 2.0, f"close() took {elapsed:.2f}s, expected ~0.25s"
        assert any(
            "abandoning the genai client" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )
        assert provider._client is None
    finally:
        release.set()  # let the abandoned worker thread exit


@pytest.mark.asyncio
async def test_a_hung_async_close_does_not_starve_the_sync_close():
    """The two closes run concurrently, not in sequence.

    Sequenced behind a wedged aclose(), the sync transport would never be
    closed at all -- the bound would expire first.
    """
    called = threading.Event()

    async def _hang():
        await asyncio.Event().wait()

    provider = GeminiProvider(api_key="k", config={"close_timeout": 0.25})
    provider._client = _fake_client(
        aclose=MagicMock(side_effect=lambda: _hang()),
        close=MagicMock(side_effect=lambda: called.set()),
    )

    await asyncio.wait_for(provider.close(), timeout=5.0)

    assert called.is_set(), (
        "the sync close never ran -- it is sequenced behind the hung async "
        "close instead of running concurrently with it"
    )


@pytest.mark.asyncio
async def test_close_is_prompt_and_silent_for_a_healthy_client(caplog):
    """The normal path must not pay the bound, and must not warn."""
    provider = GeminiProvider(api_key="k")
    client = _fake_client()
    provider._client = client

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await provider.close()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"healthy close took {elapsed:.2f}s"
    assert not [r for r in caplog.records if "abandoning" in r.getMessage()]


@pytest.mark.asyncio
async def test_close_default_bound_is_five_seconds():
    """The documented default (README `close_timeout`) is what ships."""
    assert _DEFAULT_CLOSE_TIMEOUT == 5.0
    provider = GeminiProvider(api_key="k")
    assert provider._close_timeout == 5.0


@pytest.mark.asyncio
async def test_close_timeout_config_key_is_recognized(caplog):
    """`close_timeout` must be in the consumed-key allowlist -- otherwise
    setting it earns a spurious 'unknown config key' warning."""
    with caplog.at_level(logging.WARNING):
        provider = GeminiProvider(api_key="k", config={"close_timeout": "1.5"})
    assert provider._close_timeout == 1.5  # settings.yaml strings coerce
    assert not any(
        "unknown config key" in r.getMessage().lower() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Never-built, idempotency, and the CancelledError path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_is_a_no_op_when_client_was_never_built():
    """Lazy init means the client may never exist -- close() must no-op."""
    provider = GeminiProvider(api_key="k")
    assert provider._client is None

    await provider.close()  # must not raise

    assert provider._client is None


@pytest.mark.asyncio
async def test_close_can_be_called_twice():
    """close() is idempotent; the second call closes nothing again."""
    provider = GeminiProvider(api_key="k")
    client = _fake_client()
    provider._client = client

    await provider.close()
    await provider.close()

    client.aio.aclose.assert_awaited_once()
    client.close.assert_called_once()
    assert provider._client is None


@pytest.mark.asyncio
async def test_close_resets_client_so_the_provider_can_be_reused():
    """A provider used after teardown must lazily rebuild, not reuse a
    closed transport."""
    provider = GeminiProvider(api_key="k")
    client = _fake_client()
    provider._client = client

    await provider.close()

    assert provider._client is None
    rebuilt = provider.client
    assert rebuilt is not client


@pytest.mark.asyncio
async def test_close_swallows_errors_raised_by_the_client(caplog):
    """A close that raises is logged, never propagated into teardown."""
    provider = GeminiProvider(api_key="k")
    provider._client = _fake_client(
        aclose=AsyncMock(side_effect=RuntimeError("transport already gone")),
        close=MagicMock(side_effect=OSError("bad fd")),
    )

    with caplog.at_level(logging.WARNING):
        await provider.close()  # must not raise

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("transport already gone" in m for m in messages)
    assert any("bad fd" in m for m in messages)
    assert provider._client is None


@pytest.mark.asyncio
async def test_close_swallows_cancelled_error_from_the_client():
    """CancelledError raised BY the client's own close is not propagated."""
    provider = GeminiProvider(api_key="k")
    provider._client = _fake_client(
        aclose=AsyncMock(side_effect=asyncio.CancelledError)
    )

    await provider.close()  # must not raise

    assert provider._client is None


@pytest.mark.asyncio
async def test_cancelling_the_caller_leaves_the_shielded_close_running():
    """Cancelling the CALLER must not cancel a close already in flight, and
    close() swallows the resulting CancelledError."""
    provider = GeminiProvider(api_key="k", config={"close_timeout": 30.0})

    started = asyncio.Event()
    finished = asyncio.Event()

    async def _slow_aclose():
        started.set()
        await asyncio.sleep(0.3)
        finished.set()

    provider._client = _fake_client(
        aclose=MagicMock(side_effect=lambda: _slow_aclose())
    )

    task = asyncio.create_task(provider.close())
    await asyncio.wait_for(started.wait(), timeout=5.0)
    task.cancel()
    # The caller's cancellation is absorbed by close() itself.
    await asyncio.gather(task, return_exceptions=True)

    # The shielded close keeps going and completes on its own.
    await asyncio.wait_for(finished.wait(), timeout=5.0)
    assert provider._client is None
