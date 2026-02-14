"""Tests for Gemini retry pattern (Phase 2).

Verifies exponential backoff, retry-after handling, and
provider:retry event emission using shared retry_with_backoff.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    AuthenticationError,
    ProviderUnavailableError,
    RateLimitError,
)
from amplifier_core.message_models import ChatRequest, Message
from google.genai import errors as genai_errors

from amplifier_module_provider_gemini import GeminiProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


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


def _make_server_error(
    code: int = 503, message: str = "error"
) -> genai_errors.ServerError:
    return genai_errors.ServerError(
        code, {"error": {"message": message, "status": "ERROR"}}
    )


def _make_client_error(code: int, message: str = "error") -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code, {"error": {"message": message, "status": "ERROR"}}
    )


def test_retries_on_retryable_error_then_succeeds():
    """Retry loop should retry on retryable errors and succeed when API recovers."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 3, "min_retry_delay": 0.01, "max_retry_delay": 0.1},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Fail twice with ServerError (503), then succeed
    exc = _make_server_error(503, "Temporarily down")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[exc, exc, _make_gemini_response()]
    )
    provider._client = mock_client

    result = asyncio.run(provider.complete(_simple_request()))

    # Should have called API 3 times total
    assert mock_client.aio.models.generate_content.await_count == 3

    # Should have emitted 2 provider:retry events
    retry_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 2
    assert retry_events[0][1]["provider"] == "gemini"
    assert retry_events[0][1]["attempt"] == 1
    assert retry_events[1][1]["attempt"] == 2
    assert retry_events[0][1]["error_type"] == "ProviderUnavailableError"

    # Response should be valid
    assert result is not None


def test_no_retry_on_non_retryable_error():
    """Non-retryable errors (e.g. AuthenticationError) should raise immediately."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 5, "min_retry_delay": 0.01},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    exc = _make_client_error(401, "Bad key")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except AuthenticationError:
        pass

    # Should have called API exactly once (no retry)
    assert mock_client.aio.models.generate_content.await_count == 1

    # No retry events
    retry_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 0


def test_retry_after_honored_by_shared_utility():
    """RateLimitError with retry_after should be honored by shared retry utility."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 2, "max_retry_delay": 60.0, "min_retry_delay": 0.01},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # RateLimitError with moderate retry_after (within max_delay)
    exc = RateLimitError("Rate limited", retry_after=5.0, provider="gemini")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[exc, _make_gemini_response()]
    )
    provider._client = mock_client

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = asyncio.run(provider.complete(_simple_request()))

    assert result is not None
    assert mock_client.aio.models.generate_content.await_count == 2

    # The sleep delay should respect retry_after (at least 5.0)
    sleep_delay = mock_sleep.call_args[0][0]
    assert sleep_delay >= 4.0  # retry_after=5.0 minus jitter


def test_exhausts_all_retries_then_raises():
    """After max_retries attempts, the error should propagate."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 2, "min_retry_delay": 0.01, "max_retry_delay": 0.05},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    exc = _make_server_error(503, "Still down")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ProviderUnavailableError:
        pass

    # 1 initial + 2 retries = 3 total calls
    assert mock_client.aio.models.generate_content.await_count == 3

    # 2 retry events
    retry_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 2


def test_timeout_error_retried():
    """asyncio.TimeoutError should be translated and retried."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 2, "min_retry_delay": 0.01, "max_retry_delay": 0.05},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Fail once with timeout, then succeed
    timeout_exc = asyncio.TimeoutError()
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[timeout_exc, _make_gemini_response()]
    )
    provider._client = mock_client

    result = asyncio.run(provider.complete(_simple_request()))

    assert mock_client.aio.models.generate_content.await_count == 2
    assert result is not None

    retry_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 1
    assert retry_events[0][1]["error_type"] == "LLMTimeoutError"
