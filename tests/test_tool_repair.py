"""Tests for Gemini _repaired_tool_ids bug fix (Phase 2).

Verifies that repaired tool IDs are tracked to prevent infinite
detection loops across LLM iterations.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock

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
    provider = GeminiProvider(api_key="test-key", config={"max_retries": 0})
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    provider._client = mock_client
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    return provider


def test_missing_tool_result_is_repaired_and_event_emitted():
    """Missing tool results should be repaired with synthetic results and emit event."""
    provider = _make_provider()
    fake_coordinator = provider.coordinator  # type: ignore[assignment]

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Should emit repair event
    repair_events = [
        e
        for e in fake_coordinator.hooks.events  # type: ignore[attr-defined]
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "gemini"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_repaired_tool_ids_are_not_detected_again():
    """Repaired tool IDs should be tracked and not trigger infinite detection loops.

    This test verifies the fix for the infinite loop bug where:
    1. Missing tool results are detected and synthetic results are injected
    2. Synthetic results are NOT persisted to message store
    3. On next iteration, same missing tool results are detected again
    4. This creates an infinite loop of detection -> injection -> detection

    The fix tracks repaired tool IDs to skip re-detection.
    """
    provider = _make_provider()
    fake_coordinator = provider.coordinator  # type: ignore[assignment]

    # Create a request with missing tool result
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    # First call - should detect and repair
    asyncio.run(provider.complete(request))

    # Verify repair happened
    assert "call_abc123" in provider._repaired_tool_ids
    repair_events_1 = [
        e
        for e in fake_coordinator.hooks.events  # type: ignore[attr-defined]
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_1) == 1

    # Clear events for second call
    fake_coordinator.hooks.events.clear()  # type: ignore[attr-defined]

    # Second call with SAME messages (simulating message store not persisting synthetic results)
    messages_2 = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2)

    asyncio.run(provider.complete(request_2))

    # Should NOT emit another repair event for the same tool ID
    repair_events_2 = [
        e
        for e in fake_coordinator.hooks.events  # type: ignore[attr-defined]
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked to prevent infinite loops."""
    provider = _make_provider()
    fake_coordinator = provider.coordinator  # type: ignore[assignment]

    # Create request with 3 parallel tool calls, none with results
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="grep", input={"pattern": "a"}),
                ToolCallBlock(id="call_2", name="grep", input={"pattern": "b"}),
                ToolCallBlock(id="call_3", name="grep", input={"pattern": "c"}),
            ],
        ),
        Message(role="user", content="No tool results"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # All 3 should be tracked
    assert provider._repaired_tool_ids == {"call_1", "call_2", "call_3"}

    # Verify repair event has all 3
    repair_events = [
        e
        for e in fake_coordinator.hooks.events  # type: ignore[attr-defined]
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3


def test_repaired_tool_ids_initialized_empty():
    """_repaired_tool_ids should be initialized as an empty set in __init__."""
    provider = GeminiProvider(api_key="test-key")
    assert hasattr(provider, "_repaired_tool_ids")
    assert provider._repaired_tool_ids == set()
    assert isinstance(provider._repaired_tool_ids, set)
