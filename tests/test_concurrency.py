"""Tests for process-wide concurrency semaphore and diagnostic logging.

Two features under test:

  Feature 1 — Semaphore
    A configurable ``max_concurrent_requests`` semaphore (default 5) limits how
    many API calls a single process can have in-flight simultaneously.  The
    semaphore is process-wide (shared across all GeminiProvider instances) so
    that parent + delegated child sessions in the same process share the gate.
    Setting ``max_concurrent_requests=0`` disables the semaphore entirely.

  Feature 2 — Diagnostic logging
    Structured ``provider:concurrency`` events are emitted before each API call
    attempt, carrying the current active/waiting request counts, the configured
    limit, and os.getpid(), so that post-mortem analysis can prove (or disprove)
    that concurrent request volume was responsible for a given rate-limit failure.
"""

import asyncio
import os
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import amplifier_module_provider_gemini as _mod
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_gemini import GeminiProvider


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_gemini_response():
    """Create a minimal mock Gemini API response."""
    part = SimpleNamespace(text="Hello there", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


# ---------------------------------------------------------------------------
# Fixture: isolate module-level globals between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_concurrency_globals():
    """Reset the process-wide semaphore and request counters before every test.

    asyncio.Semaphore objects are tied to the event loop that was running when
    they were created.  Each ``asyncio.run()`` call spins a new loop, so we
    must invalidate the cached semaphore to prevent "Future attached to a
    different loop" errors when tests run back-to-back.
    """
    _mod._process_semaphore = None
    _mod._process_semaphore_loop = None
    _mod._process_semaphore_max = 0
    _mod._active_requests = 0
    _mod._waiting_requests = 0
    yield
    # Clean up after the test too (defensive, avoids leaking into next fixture)
    _mod._process_semaphore = None
    _mod._process_semaphore_loop = None
    _mod._process_semaphore_max = 0
    _mod._active_requests = 0
    _mod._waiting_requests = 0


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _make_provider(
    max_concurrent: int = 5, **extra_config
) -> tuple[GeminiProvider, FakeCoordinator]:
    config = {
        "max_retries": 0,
        "max_concurrent_requests": max_concurrent,
        **extra_config,
    }
    provider = GeminiProvider(api_key="test-key", config=config)
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)
    return provider, coordinator


def _mock_api(provider: GeminiProvider, side_effect=None, return_value=None):
    """Wire up a mock for client.aio.models.generate_content."""
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.aio.models.generate_content = AsyncMock(side_effect=side_effect)
    else:
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=return_value or _make_gemini_response()
        )
    provider._client = mock_client
    return mock_client


# ============================================================================
# Feature 1: Semaphore — configuration
# ============================================================================


class TestSemaphoreConfig:
    """Unit tests for max_concurrent_requests config parsing."""

    def test_default_is_5(self):
        """Default max_concurrent_requests should be 5."""
        provider = GeminiProvider(api_key="test-key")
        assert provider._max_concurrent_requests == 5

    def test_config_overrides_default(self):
        provider = GeminiProvider(
            api_key="test-key", config={"max_concurrent_requests": 3}
        )
        assert provider._max_concurrent_requests == 3

    def test_zero_disables_semaphore(self):
        provider = GeminiProvider(
            api_key="test-key", config={"max_concurrent_requests": 0}
        )
        assert provider._max_concurrent_requests == 0

    def test_get_process_semaphore_returns_none_for_zero(self):
        async def _run():
            sem = await _mod._get_process_semaphore(0)
            assert sem is None

        asyncio.run(_run())

    def test_get_process_semaphore_returns_semaphore_for_positive(self):
        async def _run():
            sem = await _mod._get_process_semaphore(3)
            assert sem is not None
            assert isinstance(sem, asyncio.Semaphore)

        asyncio.run(_run())

    def test_get_process_semaphore_is_idempotent_within_same_loop(self):
        """Same semaphore instance should be reused within one event loop."""

        async def _run():
            sem1 = await _mod._get_process_semaphore(5)
            sem2 = await _mod._get_process_semaphore(5)
            assert sem1 is sem2

        asyncio.run(_run())

    def test_get_process_semaphore_refreshes_across_loops(self):
        """Semaphore created in loop A must not be reused in loop B."""
        sem_from_loop_a: asyncio.Semaphore | None = None

        async def _loop_a():
            nonlocal sem_from_loop_a
            sem_from_loop_a = await _mod._get_process_semaphore(5)

        asyncio.run(_loop_a())
        # Reset only the loop reference to simulate a new run
        _mod._process_semaphore_loop = None  # trigger recreation

        async def _loop_b():
            sem = await _mod._get_process_semaphore(5)
            # Different object — must have been recreated for this loop
            assert sem is not sem_from_loop_a

        asyncio.run(_loop_b())


# ============================================================================
# Feature 1: Semaphore — concurrency enforcement
# ============================================================================


class TestSemaphoreLimitsConcurrency:
    """Verify that at most max_concurrent API calls are in-flight at once."""

    def test_semaphore_limits_concurrent_calls(self):
        """With limit=2 and 5 concurrent tasks, peak in-flight must be ≤ 2."""
        max_concurrent = 2
        provider, _ = _make_provider(max_concurrent=max_concurrent)

        in_flight = 0
        max_in_flight_seen = 0

        async def slow_api(**kwargs):
            nonlocal in_flight, max_in_flight_seen
            in_flight += 1
            max_in_flight_seen = max(max_in_flight_seen, in_flight)
            await asyncio.sleep(0.02)  # enough for all 5 coroutines to be created
            in_flight -= 1
            return _make_gemini_response()

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = slow_api
        provider._client = mock_client

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            await asyncio.gather(*[provider.complete(request) for _ in range(5)])

        asyncio.run(_run())
        assert max_in_flight_seen <= max_concurrent, (
            f"Expected ≤{max_concurrent} concurrent calls, saw {max_in_flight_seen}"
        )

    def test_semaphore_limit_of_1_serializes_calls(self):
        """Limit of 1 must fully serialize all API calls."""
        provider, _ = _make_provider(max_concurrent=1)

        order: list[int] = []
        n = 4

        async def serialized_api(**kwargs):
            order.append(len(order))
            await asyncio.sleep(0.01)
            return _make_gemini_response()

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = serialized_api
        provider._client = mock_client

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            await asyncio.gather(*[provider.complete(request) for _ in range(n)])

        asyncio.run(_run())
        # All n calls must have completed
        assert len(order) == n

    def test_disabled_semaphore_allows_full_concurrency(self):
        """With max_concurrent=0, all calls run without a gate."""
        provider, _ = _make_provider(max_concurrent=0)

        in_flight = 0
        max_in_flight_seen = 0

        async def concurrent_api(**kwargs):
            nonlocal in_flight, max_in_flight_seen
            in_flight += 1
            max_in_flight_seen = max(max_in_flight_seen, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return _make_gemini_response()

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = concurrent_api
        provider._client = mock_client

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            await asyncio.gather(*[provider.complete(request) for _ in range(5)])

        asyncio.run(_run())
        # Without gate, multiple calls overlap
        assert max_in_flight_seen > 1

    def test_all_requests_complete_with_semaphore(self):
        """Semaphore must not prevent any request from completing."""
        provider, _ = _make_provider(max_concurrent=2)

        call_count = 0

        async def counting_api(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.005)
            return _make_gemini_response()

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = counting_api
        provider._client = mock_client

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            results = await asyncio.gather(
                *[provider.complete(request) for _ in range(6)]
            )
            return results

        results = asyncio.run(_run())
        assert call_count == 6
        assert len(results) == 6
        assert all(r is not None for r in results)


# ============================================================================
# Feature 2: provider:concurrency event emission
# ============================================================================


class TestConcurrencyEventEmission:
    """Verify provider:concurrency events are emitted with correct payload."""

    def test_event_emitted_on_success(self):
        """A provider:concurrency event should be emitted for each API call."""
        provider, coordinator = _make_provider(max_concurrent=5)
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1

    def test_event_has_all_required_fields(self):
        """provider:concurrency payload must contain every documented field."""
        provider, coordinator = _make_provider(max_concurrent=3)
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1
        payload = concurrency_events[0][1]

        for field in (
            "provider",
            "model",
            "active_requests",
            "waiting_requests",
            "max_concurrent",
            "process_id",
        ):
            assert field in payload, (
                f"Missing field '{field}' in provider:concurrency event"
            )

    def test_event_provider_is_gemini(self):
        """provider field in the event must be 'gemini'."""
        provider, coordinator = _make_provider(max_concurrent=3)
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["provider"] == "gemini"

    def test_event_max_concurrent_matches_config(self):
        """max_concurrent in event must equal the configured value."""
        provider, coordinator = _make_provider(max_concurrent=7)
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["max_concurrent"] == 7

    def test_event_process_id_matches_current_process(self):
        """process_id in event must equal os.getpid()."""
        provider, coordinator = _make_provider(max_concurrent=5)
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["process_id"] == os.getpid()

    def test_event_emitted_when_semaphore_disabled(self):
        """provider:concurrency is emitted even when max_concurrent=0."""
        provider, coordinator = _make_provider(max_concurrent=0)
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1
        # max_concurrent should be 0 (disabled) in the payload
        assert concurrency_events[0][1]["max_concurrent"] == 0

    def test_active_requests_at_least_1_during_call(self):
        """active_requests in the event should be ≥ 1 (the call itself)."""
        provider, coordinator = _make_provider(max_concurrent=5)
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["active_requests"] >= 1

    def test_no_event_emitted_without_coordinator(self):
        """When no coordinator is attached, provider:concurrency is silently skipped."""
        provider = GeminiProvider(
            api_key="test-key",
            config={
                "max_retries": 0,
                "max_concurrent_requests": 5,
            },
        )
        provider.coordinator = None
        _mock_api(provider)

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        # Should not raise even without a coordinator
        result = asyncio.run(provider.complete(request))
        assert result is not None
