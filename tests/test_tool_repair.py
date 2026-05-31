"""Tests for Gemini JIT tool result repair logic.

Verifies that:
- Repaired tool IDs are tracked to prevent infinite detection loops (Phase 2)
- Synthetic tool results are inserted at the correct position, not appended (Phase 3)
- A synthetic assistant bridge is inserted when a real user message immediately
  follows the injected tool results (FM3)
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
    provider = GeminiProvider(api_key="test-key", config={"max_retries": 0, "use_streaming": False})
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


def test_synthetic_result_inserted_before_subsequent_user_message():
    """Synthetic tool result must be inserted immediately after its source assistant
    message — not appended to the end — so that it appears before any subsequent
    user messages.

    Conversation layout before repair:
        [0] assistant  (tool call: call_1)
        [1] user       ("follow-up question")

    Expected layout after repair (tool result at index 1, user message pushed down):
        [0] assistant  (tool call: call_1)
        [1] tool       (synthetic result for call_1)
        [2] assistant  (FM3 bridge — closes incomplete turn)
        [3] user       ("follow-up question")
    """
    provider = _make_provider()

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="search", input={"q": "test"})],
        ),
        Message(role="user", content="follow-up question"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # The user message should still be present and LAST
    assert request.messages[-1].role == "user"
    assert request.messages[-1].content == "follow-up question"

    # The synthetic tool result must be at index 1 (right after the assistant msg)
    assert request.messages[1].role == "tool"
    assert getattr(request.messages[1], "tool_call_id", None) == "call_1"

    # Synthetic result must NOT be appended at the end
    assert request.messages[0].role == "assistant"


def test_synthetic_result_inserted_correctly_with_multiple_groups():
    """When tool calls appear in two separate assistant messages, synthetics for
    each are inserted immediately after their respective source message.

    Conversation layout before repair:
        [0] assistant  (tool call: call_A)
        [1] user       ("mid-conversation message")
        [2] assistant  (tool call: call_B)
        [3] user       ("final question")

    Expected layout after repair (processing in reverse order so indices stay valid):
        [0] assistant  (tool call: call_A)
        [1] tool       (synthetic result for call_A)
        [2] assistant  (FM3 bridge for call_A group)
        [3] user       ("mid-conversation message")
        [4] assistant  (tool call: call_B)
        [5] tool       (synthetic result for call_B)
        [6] assistant  (FM3 bridge for call_B group)
        [7] user       ("final question")
    """
    provider = _make_provider()

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_A", name="read_file", input={"path": "a.py"})
            ],
        ),
        Message(role="user", content="mid-conversation message"),
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_B", name="write_file", input={"path": "b.py"})
            ],
        ),
        Message(role="user", content="final question"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Both call IDs should be tracked as repaired
    assert "call_A" in provider._repaired_tool_ids
    assert "call_B" in provider._repaired_tool_ids

    # Collect roles sequence for easy assertion
    roles = [m.role for m in request.messages]

    # Both user messages must still be present
    user_msgs = [m for m in request.messages if m.role == "user"]
    assert len(user_msgs) == 2

    # Synthetic tool result for call_A must appear before the first user message
    call_a_idx = next(
        i
        for i, m in enumerate(request.messages)
        if getattr(m, "tool_call_id", None) == "call_A"
    )
    first_user_idx = next(i for i, m in enumerate(request.messages) if m.role == "user")
    assert call_a_idx < first_user_idx, (
        f"Synthetic for call_A (idx {call_a_idx}) must precede first user msg "
        f"(idx {first_user_idx}). Roles: {roles}"
    )

    # Synthetic tool result for call_B must appear before the second user message
    call_b_idx = next(
        i
        for i, m in enumerate(request.messages)
        if getattr(m, "tool_call_id", None) == "call_B"
    )
    second_user_idx = next(
        i
        for i, m in enumerate(request.messages)
        if m.role == "user" and i > first_user_idx
    )
    assert call_b_idx < second_user_idx, (
        f"Synthetic for call_B (idx {call_b_idx}) must precede second user msg "
        f"(idx {second_user_idx}). Roles: {roles}"
    )


def test_fm3_assistant_bridge_inserted_before_real_user_message():
    """FM3: After injecting synthetic tool results, if a real user message follows
    immediately, an assistant bridge message should be inserted to close the
    incomplete assistant turn.

    A 'real user message' is role=='user' with no tool_call_id and content
    not starting with '<system-reminder>'.
    """
    provider = _make_provider()

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_x", name="bash", input={"cmd": "ls"})],
        ),
        Message(role="user", content="What did you find?"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Expected sequence:
    #   [0] assistant (original tool call)
    #   [1] tool      (synthetic result)
    #   [2] assistant (FM3 bridge)
    #   [3] user      ("What did you find?")
    assert len(request.messages) == 4, (
        f"Expected 4 messages after FM3 repair, got {len(request.messages)}: "
        f"{[m.role for m in request.messages]}"
    )
    assert request.messages[0].role == "assistant"
    assert request.messages[1].role == "tool"
    assert getattr(request.messages[1], "tool_call_id", None) == "call_x"
    assert request.messages[2].role == "assistant", (
        "FM3 bridge should be assistant role"
    )
    assert request.messages[3].role == "user"
    assert request.messages[3].content == "What did you find?"


def test_fm3_not_triggered_when_tool_result_follows():
    """FM3 should NOT insert an assistant bridge when the next message after
    the synthetic results is itself a tool result (role=='tool'), because the
    turn is not yet complete — more tool results are expected.
    """
    provider = _make_provider()

    # call_a already has a result; call_b does not.
    # After repair, call_b's synthetic is inserted right before the existing
    # tool result for call_a. The message following the synthetic is a tool msg,
    # not a user msg, so FM3 should not fire.
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_a", name="grep", input={"pattern": "foo"}),
                ToolCallBlock(id="call_b", name="grep", input={"pattern": "bar"}),
            ],
        ),
        Message(
            role="tool",
            content="result for call_a",
            tool_call_id="call_a",
            name="grep",
        ),
        Message(role="user", content="Continue"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Only call_b is missing; call_a has a result already.
    assert "call_b" in provider._repaired_tool_ids
    assert "call_a" not in provider._repaired_tool_ids

    # Find the synthetic for call_b
    call_b_msg = next(
        (m for m in request.messages if getattr(m, "tool_call_id", None) == "call_b"),
        None,
    )
    assert call_b_msg is not None, "Synthetic for call_b should have been inserted"

    call_b_idx = request.messages.index(call_b_msg)
    # The message right after call_b's synthetic is the existing tool result for call_a
    assert request.messages[call_b_idx + 1].role == "tool"
    assert getattr(request.messages[call_b_idx + 1], "tool_call_id", None) == "call_a"


def test_fm3_not_triggered_for_system_reminder_content():
    """FM3 should NOT insert an assistant bridge when the next 'user' message
    is actually a system-reminder injection (content starts with '<system-reminder>').
    """
    provider = _make_provider()

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_z", name="ls", input={})],
        ),
        Message(
            role="user",
            content="<system-reminder>Do not forget context.</system-reminder>",
        ),
        Message(role="user", content="Real question here"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # The system-reminder message is NOT a real user message, so FM3 bridge
    # should not be inserted between the synthetic result and the reminder.
    call_z_idx = next(
        i
        for i, m in enumerate(request.messages)
        if getattr(m, "tool_call_id", None) == "call_z"
    )
    next_after_synthetic = request.messages[call_z_idx + 1]
    assert next_after_synthetic.role != "assistant" or next_after_synthetic.content == (
        "<system-reminder>Do not forget context.</system-reminder>"
    ), (
        "FM3 bridge must not be inserted before a system-reminder user message. "
        f"Got role={next_after_synthetic.role!r}, content={next_after_synthetic.content!r}"
    )
