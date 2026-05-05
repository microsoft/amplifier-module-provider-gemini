"""Tests for thought_signature roundtrip preservation (GitHub issue #41).

Gemini's API returns a ``thought_signature`` field on ``functionCall``
content blocks when thinking is enabled.  When those parts are included in
a subsequent request (as conversation history), the field MUST be present or
the API rejects the request with INVALID_ARGUMENT:
  "Function call is missing a thought_signature in functionCall parts."

Verification matrix
-------------------
1. Parse: thought_signature is extracted from a Gemini API response and
   stored on the resulting ToolCall / ToolCallBlock objects.
2. Serialize: thought_signature is re-attached to function_call parts when
   converting assistant messages back to Gemini request format.
3. Roundtrip: end-to-end — parse response → build assistant message →
   serialize to Gemini → the thought_signature appears in the output.
4. Backward-compat: responses WITHOUT thought_signature still work.
5. Multi-call: multiple tool calls, each with their own thought_signature,
   are all preserved correctly.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock

from amplifier_module_provider_gemini import GeminiProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_fc_part(
    name: str,
    args: dict[str, Any],
    thought_signature: str | None = None,
) -> SimpleNamespace:
    """Build a mock Gemini function_call response part."""
    fc = SimpleNamespace(name=name, args=args)
    if thought_signature is not None:
        fc.thought_signature = thought_signature
    # When thought_signature is absent (old API), the attribute is simply not
    # present — mirroring actual SDK behaviour.
    return SimpleNamespace(function_call=fc)


def _make_text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, thought=False)


def _make_gemini_response(parts: list) -> SimpleNamespace:
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=20,
        candidates_token_count=10,
        total_token_count=30,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _make_provider() -> GeminiProvider:
    provider = GeminiProvider(api_key="test-key", config={"max_retries": 0})
    mock_client = MagicMock()
    provider._client = mock_client
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)
    return provider


# ---------------------------------------------------------------------------
# 1. Parse: thought_signature preserved in ToolCall
# ---------------------------------------------------------------------------


def test_parse_preserves_thought_signature_in_tool_call():
    """_convert_to_chat_response must store thought_signature on ToolCall."""
    provider = _make_provider()
    sig = "abc123_base64_signature=="
    response = _make_gemini_response(
        [_make_fc_part("my_tool", {"key": "val"}, thought_signature=sig)]
    )

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.tool_calls, "Expected at least one tool call"
    tc = chat_response.tool_calls[0]
    assert tc.name == "my_tool"
    assert tc.arguments == {"key": "val"}
    # Extra field must be preserved
    assert hasattr(tc, "thought_signature"), "thought_signature missing from ToolCall"
    assert tc.thought_signature == sig


def test_parse_preserves_thought_signature_in_tool_call_block():
    """_convert_to_chat_response must store thought_signature on ToolCallBlock."""
    provider = _make_provider()
    sig = "xyz789_signature=="
    response = _make_gemini_response(
        [_make_fc_part("another_tool", {}, thought_signature=sig)]
    )

    chat_response = provider._convert_to_chat_response(response)

    # Find the ToolCallBlock in content
    blocks = [b for b in chat_response.content if hasattr(b, "type") and b.type == "tool_call"]
    assert blocks, "Expected ToolCallBlock in content"
    block = blocks[0]
    assert hasattr(block, "thought_signature"), "thought_signature missing from ToolCallBlock"
    assert block.thought_signature == sig


# ---------------------------------------------------------------------------
# 2. Serialize: thought_signature re-attached in function_call parts
# ---------------------------------------------------------------------------


def test_serialize_reattaches_thought_signature():
    """_convert_messages must include thought_signature in function_call when present."""
    provider = _make_provider()
    sig = "reattach_sig_base64=="

    # Simulate what the orchestrator stores: an assistant message with tool_calls
    # that carry thought_signature (extra field via extra="allow").
    messages = [
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {
                    "id": "gemini_call_aabbcc",
                    "name": "search",
                    "arguments": {"q": "hello"},
                    "thought_signature": sig,
                }
            ],
        }
    ]

    _, gemini_contents = provider._convert_messages(messages)

    assert len(gemini_contents) == 1
    model_msg = gemini_contents[0]
    assert model_msg["role"] == "model"
    fc_parts = [p for p in model_msg["parts"] if "function_call" in p]
    assert fc_parts, "Expected at least one function_call part"
    fc = fc_parts[0]["function_call"]
    assert "thought_signature" in fc, "thought_signature must be in serialized function_call"
    assert fc["thought_signature"] == sig


def test_serialize_without_thought_signature_still_works():
    """Backward-compat: tool calls without thought_signature serialize normally."""
    provider = _make_provider()

    messages = [
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {
                    "id": "gemini_call_112233",
                    "name": "grep",
                    "arguments": {"pattern": "foo"},
                    # No thought_signature field
                }
            ],
        }
    ]

    _, gemini_contents = provider._convert_messages(messages)

    fc_parts = [p for p in gemini_contents[0]["parts"] if "function_call" in p]
    assert fc_parts
    fc = fc_parts[0]["function_call"]
    assert fc["name"] == "grep"
    assert "thought_signature" not in fc, (
        "thought_signature must NOT appear when it was absent on the original ToolCall"
    )


# ---------------------------------------------------------------------------
# 3. End-to-end roundtrip
# ---------------------------------------------------------------------------


def test_thought_signature_full_roundtrip():
    """Parse a Gemini response → build assistant message → serialize → signature present.

    This is the exact failure path described in issue #41:
    - Turn 1: Gemini returns functionCall with thought_signature
    - Turn 2: provider serializes the previous turn back into Gemini format
              → thought_signature must be present or API returns INVALID_ARGUMENT
    """
    provider = _make_provider()
    sig = "turn1_thought_sig_base64=="

    # --- Turn 1: parse Gemini response ---
    response = _make_gemini_response(
        [_make_fc_part("read_file", {"path": "src/main.py"}, thought_signature=sig)]
    )
    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.tool_calls
    tc = chat_response.tool_calls[0]
    assert tc.thought_signature == sig  # preserved after parse

    # --- Simulate orchestrator building turn 2 request ---
    # The assistant message contains the tool_calls from turn 1.
    # The orchestrator stores ToolCall objects (which carry thought_signature via
    # extra="allow") and serializes them via model_dump().
    tc_dict = tc.model_dump()  # should include thought_signature
    assert tc_dict.get("thought_signature") == sig, (
        "thought_signature must survive model_dump() on ToolCall"
    )

    assistant_msg_dict = {
        "role": "assistant",
        "content": [],
        "tool_calls": [tc_dict],
    }

    # --- Turn 2 serialize back to Gemini request format ---
    _, gemini_contents = provider._convert_messages([assistant_msg_dict])

    assert len(gemini_contents) == 1
    model_msg = gemini_contents[0]
    fc_parts = [p for p in model_msg["parts"] if "function_call" in p]
    assert fc_parts
    fc = fc_parts[0]["function_call"]
    assert "thought_signature" in fc, (
        "thought_signature MUST be present in the serialized function_call for turn 2; "
        "its absence causes INVALID_ARGUMENT from Gemini API"
    )
    assert fc["thought_signature"] == sig
    assert fc["name"] == "read_file"
    assert fc["args"] == {"path": "src/main.py"}


# ---------------------------------------------------------------------------
# 4. Multiple tool calls — all signatures preserved
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_each_thought_signature_preserved():
    """Each tool call gets its own thought_signature preserved independently."""
    provider = _make_provider()
    sig_a = "sig_for_tool_a=="
    sig_b = "sig_for_tool_b=="

    response = _make_gemini_response([
        _make_fc_part("tool_a", {"x": 1}, thought_signature=sig_a),
        _make_fc_part("tool_b", {"y": 2}, thought_signature=sig_b),
    ])
    chat_response = provider._convert_to_chat_response(response)

    assert len(chat_response.tool_calls) == 2
    tc_by_name = {tc.name: tc for tc in chat_response.tool_calls}

    assert tc_by_name["tool_a"].thought_signature == sig_a
    assert tc_by_name["tool_b"].thought_signature == sig_b

    # Roundtrip
    tc_dicts = [tc.model_dump() for tc in chat_response.tool_calls]
    assistant_msg_dict = {
        "role": "assistant",
        "content": [],
        "tool_calls": tc_dicts,
    }
    _, gemini_contents = provider._convert_messages([assistant_msg_dict])
    fc_parts = [p for p in gemini_contents[0]["parts"] if "function_call" in p]
    assert len(fc_parts) == 2

    fc_by_name = {p["function_call"]["name"]: p["function_call"] for p in fc_parts}
    assert fc_by_name["tool_a"]["thought_signature"] == sig_a
    assert fc_by_name["tool_b"]["thought_signature"] == sig_b


# ---------------------------------------------------------------------------
# 5. Mixed: some calls have thought_signature, some don't
# ---------------------------------------------------------------------------


def test_mixed_tool_calls_only_signed_ones_get_signature():
    """If only some tool calls carry a signature, only those should re-attach it."""
    provider = _make_provider()
    sig = "only_one_sig=="

    # tool_a has signature; tool_b does not (old API or non-thinking model)
    fc_a = _make_fc_part("tool_a", {}, thought_signature=sig)
    fc_b = _make_fc_part("tool_b", {})  # no thought_signature attribute at all

    response = _make_gemini_response([fc_a, fc_b])
    chat_response = provider._convert_to_chat_response(response)

    tc_by_name = {tc.name: tc for tc in chat_response.tool_calls}
    assert tc_by_name["tool_a"].thought_signature == sig
    assert not getattr(tc_by_name["tool_b"], "thought_signature", None), (
        "tool_b must NOT have thought_signature"
    )

    # Serialize roundtrip
    tc_dicts = [tc.model_dump() for tc in chat_response.tool_calls]
    assistant_msg_dict = {"role": "assistant", "content": [], "tool_calls": tc_dicts}
    _, gemini_contents = provider._convert_messages([assistant_msg_dict])
    fc_parts = [p for p in gemini_contents[0]["parts"] if "function_call" in p]
    fc_by_name = {p["function_call"]["name"]: p["function_call"] for p in fc_parts}

    assert "thought_signature" in fc_by_name["tool_a"]
    assert fc_by_name["tool_a"]["thought_signature"] == sig
    assert "thought_signature" not in fc_by_name["tool_b"]


# ---------------------------------------------------------------------------
# 6. Integration: second complete() call does not raise
# ---------------------------------------------------------------------------


def test_second_tool_call_turn_does_not_raise_with_thought_signature():
    """Full complete() path: second turn with prior tool call (thought_signature present)
    should not fail.  Simulates the exact scenario from the bug report.
    """
    provider = _make_provider()
    sig = "integration_sig_base64=="

    # Simulate a ToolCallBlock from a previous turn (as the orchestrator would store it)
    prev_tool_block = ToolCallBlock(
        id="gemini_call_prev",
        name="bash",
        input={"command": "ls"},
        thought_signature=sig,  # extra field, preserved via extra="allow"
    )

    # Build the next request: previous assistant turn + tool result + new user message
    messages = [
        Message(
            role="assistant",
            content=[prev_tool_block],
            tool_calls=[  # type: ignore[call-arg]  # extra field
                {
                    "id": "gemini_call_prev",
                    "name": "bash",
                    "arguments": {"command": "ls"},
                    "thought_signature": sig,
                }
            ],
        ),
        Message(role="tool", content="file_list.txt", tool_call_id="gemini_call_prev", name="bash"),
        Message(role="user", content="What's in the file?"),
    ]

    # The second Gemini response (for the follow-up question)
    follow_up_response = _make_gemini_response(
        [_make_text_part("The file contains some data.")]
    )
    provider._client.aio.models.generate_content = AsyncMock(
        return_value=follow_up_response
    )

    request = ChatRequest(messages=messages)
    # Should NOT raise InvalidRequestError
    asyncio.run(provider.complete(request))

    # Verify the API was called with the correct thought_signature in the request
    call_args = provider._client.aio.models.generate_content.call_args
    contents = call_args.kwargs["contents"]

    # Find the model-role message that contains the function_call
    model_msgs = [c for c in contents if c.get("role") == "model"]
    assert model_msgs, "Expected a model message in the API call"
    fc_parts = [p for m in model_msgs for p in m.get("parts", []) if "function_call" in p]
    assert fc_parts, "Expected function_call parts in the model message"
    assert fc_parts[0]["function_call"].get("thought_signature") == sig, (
        "thought_signature must be present in the API request to avoid INVALID_ARGUMENT"
    )
