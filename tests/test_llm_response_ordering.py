"""Tests for llm:response ordering fix + canonical input_tokens key.

Verifies:
- llm:response event is emitted AFTER _convert_to_chat_response()
- llm:response event uses canonical keys: input_tokens, output_tokens, cache_read_tokens
- "input" and "output" legacy keys are NOT present in usage
"""

import asyncio
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import google
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_gemini import GeminiProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_gemini_response(
    prompt_token_count: int = 100,
    candidates_token_count: int = 50,
    total_token_count: int = 150,
    cached_content_token_count: int | None = None,
):
    """Create a minimal mock Gemini API response."""
    part = SimpleNamespace(text="Hello", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage_kwargs = {
        "prompt_token_count": prompt_token_count,
        "candidates_token_count": candidates_token_count,
        "total_token_count": total_token_count,
    }
    if cached_content_token_count is not None:
        usage_kwargs["cached_content_token_count"] = cached_content_token_count
    usage = SimpleNamespace(**usage_kwargs)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _make_provider(config: dict | None = None) -> tuple[GeminiProvider, FakeCoordinator]:
    """Create a GeminiProvider with a fake coordinator and mocked client."""
    cfg = {"max_retries": 0, "use_streaming": False, **(config or {})}
    provider = GeminiProvider(api_key="test-key", config=cfg)
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    provider._client = mock_client

    return provider, coordinator


def _run_complete(
    provider: GeminiProvider,
    mock_response=None,
) -> None:
    """Run provider.complete() with google.genai mocked out."""
    mock_genai = MagicMock()
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    if mock_response is not None:
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )
    with (
        patch.dict(sys.modules, {"google.genai": mock_genai}),
        patch.object(google, "genai", mock_genai, create=True),
    ):
        asyncio.run(provider.complete(request))


# ── Canonical key tests ─────────────────────────────────────────────────────────


def test_llm_response_uses_canonical_input_tokens_key():
    """llm:response usage should use 'input_tokens', not 'input'."""
    provider, coordinator = _make_provider()
    _run_complete(provider)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"
    usage = response_events[0].get("usage", {})
    assert "input_tokens" in usage, (
        "llm:response usage should use canonical key 'input_tokens', not 'input'"
    )
    assert "input" not in usage, (
        "llm:response usage should NOT use legacy key 'input'"
    )


def test_llm_response_uses_canonical_output_tokens_key():
    """llm:response usage should use 'output_tokens', not 'output'."""
    provider, coordinator = _make_provider()
    _run_complete(provider)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"
    usage = response_events[0].get("usage", {})
    assert "output_tokens" in usage, (
        "llm:response usage should use canonical key 'output_tokens', not 'output'"
    )
    assert "output" not in usage, (
        "llm:response usage should NOT use legacy key 'output'"
    )


def test_llm_response_includes_cache_read_tokens_when_present():
    """llm:response usage should include 'cache_read_tokens' when cache data is present."""
    provider, coordinator = _make_provider()
    cached_response = _make_gemini_response(
        prompt_token_count=100,
        candidates_token_count=50,
        total_token_count=180,
        cached_content_token_count=30,
    )
    provider._client.aio.models.generate_content = AsyncMock(return_value=cached_response)
    _run_complete(provider, mock_response=cached_response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"
    usage = response_events[0].get("usage", {})
    assert "cache_read_tokens" in usage, (
        "llm:response usage should include 'cache_read_tokens' when cache data present"
    )
    assert usage["cache_read_tokens"] == 30


def test_llm_response_usage_values_match_converted_response():
    """llm:response usage values should reflect the converted ChatResponse usage."""
    provider, coordinator = _make_provider()
    mock_response = _make_gemini_response(
        prompt_token_count=100,
        candidates_token_count=50,
        total_token_count=150,
        cached_content_token_count=30,
    )
    _run_complete(provider, mock_response=mock_response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1
    usage = response_events[0].get("usage", {})
    assert usage.get("input_tokens") == 100
    assert usage.get("output_tokens") == 50
    assert usage.get("cache_read_tokens") == 30
