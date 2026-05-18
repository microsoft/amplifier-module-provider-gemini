"""Tests for Gemini thought_signature round-trip.

Verifies that:
- function_call parts with thought_signature are captured on inbound (ToolCallBlock + ToolCall)
- text parts with thought_signature are captured on inbound (TextBlock)
- thinking parts already capture thought_signature (regression)
- tool_calls with signature are echoed with thought_signature on outbound
- thinking blocks with signature are emitted (not dropped) on outbound
- no thought_signature key is emitted when signature is None/missing (older-model compat)
- full round-trip: inbound parse -> outbound build preserves signature
- multiple parallel function_calls where only first has signature
"""

import base64
from types import SimpleNamespace
from typing import cast

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ThinkingBlock

from amplifier_module_provider_gemini import GeminiProvider


# ============================================================
# Helpers / fixtures
# ============================================================


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider() -> GeminiProvider:
    provider = GeminiProvider(api_key="test-key", config={"max_retries": 0})
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_usage():
    """Minimal usage metadata SimpleNamespace."""
    return SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
        thoughts_token_count=None,
        cached_content_token_count=None,
    )


def _make_response(parts):
    """Wrap a list of parts into a mock Gemini API response."""
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate], usage_metadata=_make_usage())


# ============================================================
# Inbound tests  (_convert_to_chat_response)
# ============================================================


def test_inbound_function_call_signature_captured():
    """function_call part with thought_signature -> ToolCallBlock.signature and ToolCall.signature."""
    sig_bytes = b"\x01\x02\x03"
    fc = SimpleNamespace(name="todo", args={"content": "do something"})
    part = SimpleNamespace(thought=False, function_call=fc, thought_signature=sig_bytes)
    response = _make_response([part])

    provider = _make_provider()
    result = provider._convert_to_chat_response(response)

    # ToolCallBlock in content
    assert result.content, "Expected content blocks"
    tc_block = result.content[0]
    assert getattr(tc_block, "signature", None) == sig_bytes, (
        f"ToolCallBlock.signature should be {sig_bytes!r}, "
        f"got {getattr(tc_block, 'signature', None)!r}"
    )

    # ToolCall in tool_calls list
    assert result.tool_calls, "Expected tool_calls"
    tc = result.tool_calls[0]
    assert getattr(tc, "signature", None) == sig_bytes, (
        f"ToolCall.signature should be {sig_bytes!r}, "
        f"got {getattr(tc, 'signature', None)!r}"
    )


def test_inbound_text_signature_captured():
    """Non-thought text part with thought_signature -> TextBlock.signature."""
    sig_bytes = b"\x04\x05\x06"
    part = SimpleNamespace(text="final answer", thought=False, thought_signature=sig_bytes)
    response = _make_response([part])

    provider = _make_provider()
    result = provider._convert_to_chat_response(response)

    assert result.content, "Expected content blocks"
    tb = result.content[0]
    assert getattr(tb, "signature", None) == sig_bytes, (
        f"TextBlock.signature should be {sig_bytes!r}, "
        f"got {getattr(tb, 'signature', None)!r}"
    )


def test_inbound_thinking_signature_unchanged():
    """Regression: ThinkingBlock.signature capture still works (existing code path)."""
    sig_bytes = b"\xde\xad\xbe\xef"
    part = SimpleNamespace(text="my reasoning", thought=True, thought_signature=sig_bytes)
    response = _make_response([part])

    provider = _make_provider()
    result = provider._convert_to_chat_response(response)

    assert result.content, "Expected content blocks"
    tb = result.content[0]
    assert isinstance(tb, ThinkingBlock), f"Expected ThinkingBlock, got {type(tb)}"
    assert tb.signature == base64.b64encode(sig_bytes).decode("ascii"), (
        f"ThinkingBlock.signature should be base64 string, got {tb.signature!r}"
    )


def test_inbound_no_signature_no_field():
    """Parts without thought_signature leave signature unset (older model compat)."""
    fc = SimpleNamespace(name="grep", args={"pattern": "test"})
    # Deliberately no thought_signature attribute on the part
    part = SimpleNamespace(thought=False, function_call=fc)
    response = _make_response([part])

    provider = _make_provider()
    result = provider._convert_to_chat_response(response)

    assert result.content
    assert getattr(result.content[0], "signature", None) is None, (
        "signature should be absent/None for parts without thought_signature"
    )
    assert result.tool_calls
    assert getattr(result.tool_calls[0], "signature", None) is None, (
        "ToolCall.signature should be absent/None for parts without thought_signature"
    )


# ============================================================
# Outbound tests  (_convert_messages)
# ============================================================


def test_outbound_tool_call_signature_echoed():
    """tool_calls entry with signature -> function_call part carries thought_signature (base64)."""
    sig_bytes = b"\xca\xfe\xba\xbe"
    expected_b64 = base64.b64encode(sig_bytes).decode("ascii")

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "todo",
                    "arguments": {"content": "test"},
                    "signature": sig_bytes,
                }
            ],
        }
    ]

    provider = _make_provider()
    _, gemini_contents = provider._convert_messages(messages)

    assert len(gemini_contents) == 1
    parts = gemini_contents[0]["parts"]
    assert len(parts) == 1
    fc_part = parts[0]
    assert "function_call" in fc_part
    assert "thought_signature" in fc_part, (
        f"Expected thought_signature in fc_part, got keys: {list(fc_part.keys())}"
    )
    assert fc_part["thought_signature"] == expected_b64, (
        f"Expected {expected_b64!r}, got {fc_part['thought_signature']!r}"
    )


def test_outbound_thinking_block_not_dropped():
    """Thinking block with signature -> emitted as thought part with thought_signature."""
    sig_bytes = b"\x11\x22\x33"
    expected_b64 = base64.b64encode(sig_bytes).decode("ascii")

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "my reasoning here",
                    "signature": sig_bytes,
                    "visibility": "internal",
                }
            ],
            "tool_calls": [],
        }
    ]

    provider = _make_provider()
    _, gemini_contents = provider._convert_messages(messages)

    assert len(gemini_contents) == 1
    parts = gemini_contents[0]["parts"]

    thought_parts = [p for p in parts if p.get("thought") is True]
    assert thought_parts, f"Expected a thought part in {parts}"
    tp = thought_parts[0]
    assert tp.get("text") == "my reasoning here", (
        f"Expected thinking text, got: {tp.get('text')!r}"
    )
    assert tp.get("thought_signature") == expected_b64, (
        f"Expected {expected_b64!r}, got {tp.get('thought_signature')!r}"
    )


def test_outbound_no_signature_no_thought_signature_field():
    """tool_calls without signature -> no thought_signature key in function_call part (older-model compat)."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "grep",
                    "arguments": {"pattern": "test"},
                    # No signature key
                }
            ],
        }
    ]

    provider = _make_provider()
    _, gemini_contents = provider._convert_messages(messages)

    parts = gemini_contents[0]["parts"]
    fc_part = parts[0]
    assert "function_call" in fc_part
    assert "thought_signature" not in fc_part, (
        f"thought_signature should be absent for tool_calls without signature, got: {fc_part}"
    )


def test_outbound_thinking_block_without_signature_still_dropped():
    """Thinking block WITHOUT signature (old model) should still be dropped — backward compat."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "some thoughts",
                    "visibility": "internal",
                    # No signature
                }
            ],
            "tool_calls": [],
        }
    ]

    provider = _make_provider()
    _, gemini_contents = provider._convert_messages(messages)

    # Either no entry or an entry with no thought parts
    if gemini_contents:
        parts = gemini_contents[0]["parts"]
        thought_parts = [p for p in parts if p.get("thought") is True]
        assert not thought_parts, (
            f"Thinking blocks without signature should not be echoed, got: {thought_parts}"
        )


# ============================================================
# Round-trip integration tests
# ============================================================


def test_round_trip_function_call_signature():
    """Inbound parse -> serialize -> outbound build -> thought_signature survives."""
    sig_bytes = b"\xfe\xed\xfa\xce"
    expected_b64 = base64.b64encode(sig_bytes).decode("ascii")

    # Simulate Gemini response with function_call + thought_signature
    fc = SimpleNamespace(name="todo", args={"content": "track something"})
    part = SimpleNamespace(thought=False, function_call=fc, thought_signature=sig_bytes)
    response = _make_response([part])

    provider = _make_provider()
    chat_response = provider._convert_to_chat_response(response)

    # Replicate what complete() does: serialize via model_dump, feed to _convert_messages
    tool_calls_raw = [tc.model_dump() for tc in chat_response.tool_calls]
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls_raw,
    }

    _, gemini_contents = provider._convert_messages([assistant_msg])

    parts = gemini_contents[0]["parts"]
    assert len(parts) == 1
    fc_part = parts[0]
    assert "thought_signature" in fc_part, (
        f"Round-trip should preserve thought_signature, got: {fc_part}"
    )
    assert fc_part["thought_signature"] == expected_b64


def test_round_trip_multiple_parallel_calls_only_first_has_signature():
    """Multiple parallel function_calls: only [0] has signature; others must not get one."""
    sig_bytes = b"\xaa\xbb\xcc"
    expected_b64 = base64.b64encode(sig_bytes).decode("ascii")

    def _fc_part(name, sig=None):
        fc = SimpleNamespace(name=name, args={})
        ns = SimpleNamespace(thought=False, function_call=fc)
        if sig is not None:
            ns.thought_signature = sig
        return ns

    parts_in = [
        _fc_part("todo", sig=sig_bytes),
        _fc_part("grep"),   # no signature
        _fc_part("bash"),   # no signature
    ]
    response = _make_response(parts_in)

    provider = _make_provider()
    chat_response = provider._convert_to_chat_response(response)

    # Verify inbound: only first TC captured a signature
    assert len(chat_response.tool_calls) == 3
    assert getattr(chat_response.tool_calls[0], "signature", None) == sig_bytes
    assert getattr(chat_response.tool_calls[1], "signature", None) is None
    assert getattr(chat_response.tool_calls[2], "signature", None) is None

    # Round-trip through _convert_messages
    tool_calls_raw = [tc.model_dump() for tc in chat_response.tool_calls]
    assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls_raw}
    _, gemini_contents = provider._convert_messages([assistant_msg])

    parts_out = gemini_contents[0]["parts"]
    assert len(parts_out) == 3

    # First part carries thought_signature
    assert "thought_signature" in parts_out[0], (
        f"First function_call should have thought_signature, got: {parts_out[0]}"
    )
    assert parts_out[0]["thought_signature"] == expected_b64

    # Remaining two must NOT carry thought_signature
    assert "thought_signature" not in parts_out[1], (
        f"Second function_call should NOT have thought_signature, got: {parts_out[1]}"
    )
    assert "thought_signature" not in parts_out[2], (
        f"Third function_call should NOT have thought_signature, got: {parts_out[2]}"
    )
