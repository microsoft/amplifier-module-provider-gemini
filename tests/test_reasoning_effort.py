"""Tests for Gemini reasoning_effort support (Phase 2).

Verifies that request.reasoning_effort maps to thinking_budget values,
and that kwargs["thinking_budget"] overrides reasoning_effort.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

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


def _make_gemini_response():
    """Create a minimal mock Gemini API response."""
    part = SimpleNamespace(text="Hello", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _make_provider() -> GeminiProvider:
    provider = GeminiProvider(api_key="test-key", config={"max_retries": 0, "use_streaming": False})
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _capture_config(provider: GeminiProvider):
    """Set up mock and return a function to extract the config passed to generate_content."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    provider._client = mock_client
    return mock_client


def test_reasoning_effort_low_sets_budget_4096():
    """reasoning_effort='low' -> thinking_budget=4096."""
    provider = _make_provider()
    mock_client = _capture_config(provider)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="low",
    )
    asyncio.run(provider.complete(request))

    # Extract the config argument passed to generate_content
    call_kwargs = mock_client.aio.models.generate_content.await_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.thinking_config.thinking_budget == 4096


def test_reasoning_effort_medium_sets_dynamic():
    """reasoning_effort='medium' -> thinking_budget=-1 (dynamic)."""
    provider = _make_provider()
    mock_client = _capture_config(provider)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="medium",
    )
    asyncio.run(provider.complete(request))

    call_kwargs = mock_client.aio.models.generate_content.await_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.thinking_config.thinking_budget == -1


def test_reasoning_effort_high_sets_dynamic():
    """reasoning_effort='high' -> thinking_budget=-1 (dynamic)."""
    provider = _make_provider()
    mock_client = _capture_config(provider)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="high",
    )
    asyncio.run(provider.complete(request))

    call_kwargs = mock_client.aio.models.generate_content.await_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.thinking_config.thinking_budget == -1


def test_reasoning_effort_none_preserves_default():
    """reasoning_effort=None -> no explicit budget/level; model default thinking.

    Historically this asserted thinking_budget == -1 ("explicit dynamic").
    Post-retarget, no directive at all means neither thinking_budget nor
    thinking_level is sent -- verified live that Gemini still thinks by
    default (and, with include_thoughts=True, still returns thought
    summaries) with a bare ThinkingConfig(include_thoughts=True). Sending
    -1 explicitly was functionally equivalent but foreclosed the
    thinking_level path for models that only accept levels.
    """
    provider = _make_provider()
    mock_client = _capture_config(provider)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=None,
    )
    asyncio.run(provider.complete(request))

    call_kwargs = mock_client.aio.models.generate_content.await_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level is None
    assert config.thinking_config.include_thoughts is True


def test_kwargs_thinking_budget_overrides_reasoning_effort():
    """kwargs['thinking_budget'] takes absolute precedence over reasoning_effort."""
    provider = _make_provider()
    mock_client = _capture_config(provider)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="low",  # Would set 4096
    )
    # But kwargs override to 8192
    asyncio.run(provider.complete(request, thinking_budget=8192))

    call_kwargs = mock_client.aio.models.generate_content.await_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.thinking_config.thinking_budget == 8192


def test_metadata_thinking_budget_with_no_reasoning_effort():
    """request.metadata['thinking_budget'] should work when reasoning_effort is None."""
    provider = _make_provider()
    mock_client = _capture_config(provider)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        metadata={"thinking_budget": 16384},
    )
    asyncio.run(provider.complete(request))

    call_kwargs = mock_client.aio.models.generate_content.await_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.thinking_config.thinking_budget == 16384


def test_thinking_disabled_with_budget_zero():
    """thinking_budget=0 in kwargs should disable thinking entirely.

    Bug fix: the pre-retarget implementation OMITTED thinking_config
    entirely when thinking_budget resolved to 0, and this test asserted
    exactly that ("thinking_config is None"). Verified live against the
    real API that omitting the field is NOT the same as sending an explicit
    zero: gemini-2.5-flash with no thinking_config at all still reports a
    populated thoughts_token_count (still thinking, budget=0 never actually
    reached the API), while an EXPLICIT thinking_budget=0 correctly reports
    thoughts_token_count=None (thinking genuinely disabled). The fix always
    sends the explicit value through instead of omitting it.
    """
    provider = _make_provider()
    mock_client = _capture_config(provider)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="high",  # Would enable thinking
    )
    asyncio.run(provider.complete(request, thinking_budget=0))

    call_kwargs = mock_client.aio.models.generate_content.await_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0
    assert config.thinking_config.thinking_level is None
