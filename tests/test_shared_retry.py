"""Tests for shared retry_with_backoff integration (Gemini).

Verifies that the Gemini provider uses the shared RetryConfig and
retry_with_backoff from amplifier-core instead of its own retry loop,
and adopts new error types (AccessDeniedError for 403).
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import AccessDeniedError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.utils.retry import RetryConfig
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


def _make_client_error(code: int, message: str = "error") -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code, {"error": {"message": message, "status": "ERROR"}}
    )


# --- Structural: uses shared RetryConfig ---


def test_provider_has_retry_config():
    """Provider should store a RetryConfig instance (not separate vars)."""
    provider = GeminiProvider(api_key="test-key")
    assert hasattr(provider, "_retry_config")
    assert isinstance(provider._retry_config, RetryConfig)


def test_retry_config_respects_config_values():
    """RetryConfig should be populated from provider config dict."""
    provider = GeminiProvider(
        api_key="test-key",
        config={
            "max_retries": 7,
            "min_retry_delay": 2.0,
            "max_retry_delay": 120.0,
            "retry_jitter": False,
        },
    )
    assert provider._retry_config.max_retries == 7
    assert provider._retry_config.min_delay == 2.0
    assert provider._retry_config.max_delay == 120.0
    assert provider._retry_config.jitter == 0.0  # False -> 0.0


def test_no_calculate_retry_delay_method():
    """_calculate_retry_delay should be removed (replaced by shared utility)."""
    assert not hasattr(GeminiProvider, "_calculate_retry_delay")


# --- Error type: 403 -> AccessDeniedError ---


def test_client_error_403_becomes_access_denied_error():
    """ClientError with code 403 -> AccessDeniedError (not AuthenticationError)."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())

    exc = _make_client_error(403, "Forbidden - insufficient permissions")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except AccessDeniedError as e:
        assert e.provider == "gemini"
        assert e.status_code == 403
        assert e.__cause__ is exc


# --- Retry behavior through shared utility ---


def test_retry_with_shared_utility_succeeds():
    """Shared retry_with_backoff should retry transient errors and return on success."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 3, "min_retry_delay": 0.01, "max_retry_delay": 0.1},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Fail twice with ServerError (503), then succeed
    exc = genai_errors.ServerError(
        503, {"error": {"message": "Temporarily down", "status": "ERROR"}}
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[exc, exc, _make_gemini_response()]
    )
    provider._client = mock_client

    result = asyncio.run(provider.complete(_simple_request()))

    # Should have called API 3 times total
    assert mock_client.aio.models.generate_content.await_count == 3
    assert result is not None

    # Should have emitted provider:retry events
    retry_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 2
    assert retry_events[0][1]["provider"] == "gemini"


def test_jitter_backward_compat_bool_true():
    """retry_jitter=True (old bool format) should map to jitter=0.2."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"retry_jitter": True},
    )
    assert provider._retry_config.jitter == 0.2


def test_jitter_backward_compat_bool_false():
    """retry_jitter=False (old bool format) should map to jitter=0.0."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"retry_jitter": False},
    )
    assert provider._retry_config.jitter == 0.0
