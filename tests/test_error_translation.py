"""Tests for Gemini error translation (Phase 2).

Verifies that native Google GenAI SDK exceptions are translated to kernel error types
with correct provider attribution and cause chain preservation.

The google-genai SDK raises:
- google.genai.errors.ClientError (4xx) with .code for HTTP status
- google.genai.errors.ServerError (5xx) with .code for HTTP status
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
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


def _make_provider(**config_overrides) -> GeminiProvider:
    """Create a provider with retry disabled for error translation tests."""
    config = {"max_retries": 0, **config_overrides}
    provider = GeminiProvider(api_key="test-key", config=config)
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_client_error(code: int, message: str = "error") -> genai_errors.ClientError:
    """Create a ClientError with a given HTTP status code."""
    return genai_errors.ClientError(
        code, {"error": {"message": message, "status": "ERROR"}}
    )


def _make_server_error(code: int, message: str = "error") -> genai_errors.ServerError:
    """Create a ServerError with a given HTTP status code."""
    return genai_errors.ServerError(
        code, {"error": {"message": message, "status": "ERROR"}}
    )


def test_client_error_429_becomes_rate_limit_error():
    """ClientError with code 429 -> RateLimitError."""
    provider = _make_provider()

    exc = _make_client_error(429, "Quota exceeded")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except RateLimitError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.__cause__ is exc
        assert e.status_code == 429


def test_client_error_401_becomes_authentication_error():
    """ClientError with code 401 -> AuthenticationError."""
    provider = _make_provider()

    exc = _make_client_error(401, "Invalid key")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except AuthenticationError as e:
        assert e.provider == "gemini"
        assert e.retryable is False
        assert e.__cause__ is exc
        assert e.status_code == 401


def test_client_error_403_becomes_authentication_error():
    """ClientError with code 403 -> AuthenticationError."""
    provider = _make_provider()

    exc = _make_client_error(403, "Forbidden")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except AuthenticationError as e:
        assert e.provider == "gemini"
        assert e.retryable is False
        assert e.__cause__ is exc
        assert e.status_code == 403


def test_client_error_400_becomes_invalid_request_error():
    """ClientError with code 400 -> InvalidRequestError."""
    provider = _make_provider()

    exc = _make_client_error(400, "Bad param")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except InvalidRequestError as e:
        assert e.provider == "gemini"
        assert e.retryable is False
        assert e.__cause__ is exc
        assert e.status_code == 400


def test_client_error_token_limit_becomes_context_length_error():
    """ClientError with 'token limit' in message -> ContextLengthError."""
    provider = _make_provider()

    exc = _make_client_error(400, "token limit exceeded for this model")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ContextLengthError as e:
        assert e.provider == "gemini"
        assert e.__cause__ is exc
        assert e.status_code == 400


def test_client_error_safety_becomes_content_filter_error():
    """ClientError with 'safety' in message -> ContentFilterError."""
    provider = _make_provider()

    exc = _make_client_error(400, "Response blocked due to safety settings")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ContentFilterError as e:
        assert e.provider == "gemini"
        assert e.__cause__ is exc
        assert e.status_code == 400


def test_server_error_503_becomes_provider_unavailable_error():
    """ServerError with code 503 -> ProviderUnavailableError."""
    provider = _make_provider()

    exc = _make_server_error(503, "Service down")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ProviderUnavailableError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.__cause__ is exc
        assert e.status_code == 503


def test_server_error_500_becomes_provider_unavailable_error():
    """ServerError with code 500 -> ProviderUnavailableError."""
    provider = _make_provider()

    exc = _make_server_error(500, "Internal error")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ProviderUnavailableError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.__cause__ is exc
        assert e.status_code == 500


def test_asyncio_timeout_becomes_llm_timeout_error():
    """asyncio.TimeoutError -> LLMTimeoutError."""
    provider = _make_provider()

    exc = asyncio.TimeoutError()
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except LLMTimeoutError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.__cause__ is exc


def test_unknown_exception_becomes_retryable_llm_error():
    """Generic Exception -> LLMError(retryable=True)."""
    provider = _make_provider()

    exc = RuntimeError("Something unexpected")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except LLMError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.__cause__ is exc
        assert "Something unexpected" in str(e)


def test_error_event_emitted_on_failure():
    """llm:response error event should be emitted when API call fails."""
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    exc = RuntimeError("Boom")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
    except LLMError:
        pass

    error_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "llm:response" and e[1].get("status") == "error"
    ]
    assert len(error_events) >= 1


def test_cause_chain_preserved():
    """__cause__ should point to the original native exception."""
    provider = _make_provider()

    native_exc = _make_client_error(429, "Rate limited")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=native_exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except RateLimitError as e:
        # The cause chain should be preserved for debugging
        assert e.__cause__ is native_exc
        assert isinstance(e.__cause__, genai_errors.ClientError)


# ---------------------------------------------------------------------------
# Fallback google.api_core.exceptions.ResourceExhausted tests
# ---------------------------------------------------------------------------

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None  # type: ignore[assignment]


def test_fallback_resource_exhausted_has_retryable_true():
    """Fallback ResourceExhausted -> RateLimitError with retryable=True."""
    if google_exceptions is None:
        import pytest

        pytest.skip("google.api_core not installed")

    provider = _make_provider()

    exc = google_exceptions.ResourceExhausted("Quota exceeded")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except RateLimitError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.status_code == 429
        assert e.__cause__ is exc


def test_fallback_resource_exhausted_extracts_retry_after():
    """Fallback ResourceExhausted with Retry-After header -> retry_after set."""
    if google_exceptions is None:
        import pytest

        pytest.skip("google.api_core not installed")

    provider = _make_provider()

    fake_response = MagicMock()
    fake_response.headers = {"Retry-After": "30"}
    exc = google_exceptions.ResourceExhausted(
        "Quota exceeded", errors=[], response=fake_response
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except RateLimitError as e:
        assert e.retry_after == 30.0
        assert e.retryable is True


def test_fallback_resource_exhausted_fail_fast_when_retry_after_exceeds_max():
    """Fallback ResourceExhausted with retry_after > max_delay -> retryable=False."""
    if google_exceptions is None:
        import pytest

        pytest.skip("google.api_core not installed")

    # max_retry_delay defaults to 60
    provider = _make_provider(max_retry_delay=60)

    fake_response = MagicMock()
    fake_response.headers = {"Retry-After": "120"}
    exc = google_exceptions.ResourceExhausted(
        "Quota exceeded", errors=[], response=fake_response
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except RateLimitError as e:
        assert e.retry_after == 120.0
        assert e.retryable is False


# ---------------------------------------------------------------------------
# ServerError Retry-After extraction tests
# ---------------------------------------------------------------------------


def test_server_error_extracts_retry_after_from_headers():
    """ServerError with Retry-After header -> ProviderUnavailableError with retry_after."""
    provider = _make_provider()

    exc = _make_server_error(503, "Service overloaded")
    fake_response = MagicMock()
    fake_response.headers = {"Retry-After": "10"}
    exc.response = fake_response

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ProviderUnavailableError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.status_code == 503
        assert e.retry_after == 10.0
        assert e.__cause__ is exc


def test_server_error_without_retry_after_preserves_existing_behavior():
    """ServerError without Retry-After header -> ProviderUnavailableError with retry_after=None."""
    provider = _make_provider()

    exc = _make_server_error(500, "Internal error")
    # No .response attribute set — _extract_retry_after should return None

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ProviderUnavailableError as e:
        assert e.provider == "gemini"
        assert e.retryable is True
        assert e.status_code == 500
        assert e.retry_after is None
        assert e.__cause__ is exc
