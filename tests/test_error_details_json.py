"""Tests for json.dumps(e.details) error message pattern.

Verifies that GenAI SDK error handlers pass json.dumps(e.details) as the error
message instead of str(e), while google.api_core fallback and keyword matching
remain unchanged.
"""

import asyncio
import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    AccessDeniedError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
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


# --- ClientError: error message should be json.dumps(e.details) ---


def test_client_error_429_uses_json_details():
    """ClientError 429 -> RateLimitError with json.dumps(details) message."""
    provider = _make_provider()
    exc = _make_client_error(429, "Quota exceeded")
    expected_msg = json.dumps(exc.details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except RateLimitError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


def test_client_error_401_uses_json_details():
    """ClientError 401 -> AuthenticationError with json.dumps(details) message."""
    provider = _make_provider()
    exc = _make_client_error(401, "Invalid key")
    expected_msg = json.dumps(exc.details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except AuthenticationError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


def test_client_error_403_uses_json_details():
    """ClientError 403 -> AccessDeniedError with json.dumps(details) message."""
    provider = _make_provider()
    exc = _make_client_error(403, "Forbidden")
    expected_msg = json.dumps(exc.details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except AccessDeniedError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


def test_client_error_context_length_uses_json_details():
    """ClientError with token limit keyword -> ContextLengthError with json.dumps(details)."""
    provider = _make_provider()
    exc = _make_client_error(400, "token limit exceeded for this model")
    expected_msg = json.dumps(exc.details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ContextLengthError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


def test_client_error_safety_uses_json_details():
    """ClientError with safety keyword -> ContentFilterError with json.dumps(details)."""
    provider = _make_provider()
    exc = _make_client_error(400, "Response blocked due to safety settings")
    expected_msg = json.dumps(exc.details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ContentFilterError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


def test_client_error_generic_400_uses_json_details():
    """ClientError 400 generic -> InvalidRequestError with json.dumps(details)."""
    provider = _make_provider()
    exc = _make_client_error(400, "Bad param")
    expected_msg = json.dumps(exc.details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except InvalidRequestError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


# --- ServerError: error message should be json.dumps(e.details) ---


def test_server_error_uses_json_details():
    """ServerError -> ProviderUnavailableError with json.dumps(details) message."""
    provider = _make_provider()
    exc = _make_server_error(503, "Service down")
    expected_msg = json.dumps(exc.details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ProviderUnavailableError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


# --- Ultimate fallback: uses details pattern too ---


def test_unknown_error_with_details_uses_json_details():
    """Unknown exception with .details attr -> LLMError with json.dumps(details)."""
    provider = _make_provider()

    # Create a RuntimeError with a .details attribute
    exc = RuntimeError("Something unexpected")
    details = {"custom": "structured data"}
    setattr(exc, "details", details)
    expected_msg = json.dumps(details)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except LLMError as e:
        assert str(e) == expected_msg, f"Expected {expected_msg!r}, got {str(e)!r}"


def test_unknown_error_without_details_uses_str():
    """Unknown exception without .details -> LLMError with str(e) as before."""
    provider = _make_provider()

    exc = RuntimeError("Something unexpected")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except LLMError as e:
        assert "Something unexpected" in str(e)


# --- google.api_core fallback: should still use str(e) ---


def test_google_api_core_fallback_uses_str():
    """google.api_core exceptions should still use str(e), not json.dumps."""
    try:
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        import pytest

        pytest.skip("google.api_core not installed")

    provider = _make_provider()

    exc = google_exceptions.InvalidArgument("Bad argument value")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except InvalidRequestError as e:
        # Should contain str(e), not json.dumps
        assert "Bad argument" in str(e)


# --- Keyword matching still works with str(e).lower() ---


def test_keyword_matching_still_detects_context_length():
    """str(e).lower() keyword matching for context length should still work."""
    provider = _make_provider()
    exc = _make_client_error(400, "Request exceeds context length limit")

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ContextLengthError:
        pass  # Correct classification


def test_keyword_matching_still_detects_content_filter():
    """str(e).lower() keyword matching for content filter should still work."""
    provider = _make_provider()
    exc = _make_client_error(400, "Content blocked by content filter")

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)
    provider._client = mock_client

    try:
        asyncio.run(provider.complete(_simple_request()))
        assert False, "Should have raised"
    except ContentFilterError:
        pass  # Correct classification
