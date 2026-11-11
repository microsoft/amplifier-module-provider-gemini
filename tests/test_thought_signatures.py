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
import copy
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
import httpx
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ThinkingBlock
from google import genai
from google.genai import types

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


def _legacy_loop_message(provider: GeminiProvider, parts: list) -> dict:
    """Capture a response through the legacy loop's lossy tool-call shape."""
    chat_response = provider._convert_to_chat_response(_make_response(parts))
    assistant_message = {
        "role": "assistant",
        "content": [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in chat_response.content
        ],
        "tool_calls": [
            {
                "id": tool_call.id,
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in chat_response.tool_calls
        ],
    }
    return json.loads(json.dumps(assistant_message))


def _function_call_parts(gemini_contents: list[dict]) -> list[dict]:
    """Return all emitted Gemini function-call parts in message order."""
    return [
        part
        for content in gemini_contents
        for part in content["parts"]
        if "function_call" in part
    ]


# ============================================================
# Inbound tests  (_convert_to_chat_response)
# ============================================================


def test_inbound_function_call_signature_captured():
    """function_call part with thought_signature -> ToolCallBlock.signature and ToolCall.signature.

    Captured as a base64 str, NOT the SDK's raw bytes (see
    test_inbound_signature_is_json_safe_not_raw_bytes for why: raw bytes
    break JSON serialization for non-UTF-8 signatures, which is the normal
    case for an opaque cryptographic signature)."""
    sig_bytes = b"\x01\x02\x03"
    expected_b64 = base64.b64encode(sig_bytes).decode("ascii")
    fc = SimpleNamespace(name="todo", args={"content": "do something"})
    part = SimpleNamespace(thought=False, function_call=fc, thought_signature=sig_bytes)
    response = _make_response([part])

    provider = _make_provider()
    result = provider._convert_to_chat_response(response)

    # ToolCallBlock in content
    assert result.content, "Expected content blocks"
    tc_block = result.content[0]
    assert getattr(tc_block, "signature", None) == expected_b64, (
        f"ToolCallBlock.signature should be base64 {expected_b64!r}, "
        f"got {getattr(tc_block, 'signature', None)!r}"
    )

    # ToolCall in tool_calls list
    assert result.tool_calls, "Expected tool_calls"
    tc = result.tool_calls[0]
    assert getattr(tc, "signature", None) == expected_b64, (
        f"ToolCall.signature should be base64 {expected_b64!r}, "
        f"got {getattr(tc, 'signature', None)!r}"
    )


def test_inbound_text_signature_captured():
    """Non-thought text part with thought_signature -> TextBlock.signature.

    Captured as a base64 str, NOT the SDK's raw bytes -- see
    test_inbound_signature_is_json_safe_not_raw_bytes."""
    sig_bytes = b"\x04\x05\x06"
    expected_b64 = base64.b64encode(sig_bytes).decode("ascii")
    part = SimpleNamespace(
        text="final answer", thought=False, thought_signature=sig_bytes
    )
    response = _make_response([part])

    provider = _make_provider()
    result = provider._convert_to_chat_response(response)

    assert result.content, "Expected content blocks"
    tb = result.content[0]
    assert getattr(tb, "signature", None) == expected_b64, (
        f"TextBlock.signature should be base64 {expected_b64!r}, "
        f"got {getattr(tb, 'signature', None)!r}"
    )


def test_inbound_thinking_signature_unchanged():
    """Regression: ThinkingBlock.signature capture still works (existing code path)."""
    sig_bytes = b"\xde\xad\xbe\xef"
    part = SimpleNamespace(
        text="my reasoning", thought=True, thought_signature=sig_bytes
    )
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
                    "thinking_provenance": NATIVE_PROVENANCE,
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
                    "thinking_provenance": NATIVE_PROVENANCE,
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
                    "thinking_provenance": NATIVE_PROVENANCE,
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


def test_outbound_untagged_thinking_is_visible_text_not_native_state():
    """Legacy untagged thinking keeps its summary, not native opaque semantics."""
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

    assert gemini_contents == [{"role": "model", "parts": [{"text": "some thoughts"}]}]


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
        _fc_part("grep"),  # no signature
        _fc_part("bash"),  # no signature
    ]
    response = _make_response(parts_in)

    provider = _make_provider()
    chat_response = provider._convert_to_chat_response(response)

    # Verify inbound: only first TC captured a signature (as base64 str)
    assert len(chat_response.tool_calls) == 3
    assert getattr(chat_response.tool_calls[0], "signature", None) == expected_b64
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


def test_legacy_loop_json_history_recovers_tool_call_signature_once():
    """A legacy loop's unsigned tool_calls recover matching block signatures."""
    raw_signature = bytes([0xFF, 0xFE, 0x80, 0x81, 0x00, 0x9D])
    expected_signature = base64.b64encode(raw_signature).decode("ascii")
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=raw_signature,
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    original_message = copy.deepcopy(legacy_message)

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert legacy_message == original_message
    assert len(function_parts) == 1
    assert function_parts[0]["thought_signature"] == expected_signature
    assert base64.b64decode(function_parts[0]["thought_signature"]) == raw_signature


def test_legacy_signature_recovery_accepts_explicit_none():
    """An explicitly None legacy signature has the same recovery path as absent."""
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    legacy_message["tool_calls"][0]["signature"] = None

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 1
    assert function_parts[0]["thought_signature"] == base64.b64encode(b"signed").decode(
        "ascii"
    )


def test_legacy_signature_recovery_does_not_change_empty_signatures():
    """An explicit empty signature remains unsigned instead of being recovered."""
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    legacy_message["tool_calls"][0]["signature"] = ""
    legacy_message["tool_calls"][0]["thinking_provenance"] = NATIVE_PROVENANCE

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 1
    assert "thought_signature" not in function_parts[0]


def test_legacy_signature_recovery_preserves_canonical_tool_call_signature():
    """An existing tool_calls signature wins over the matching content block."""
    content_signature = b"content-signature"
    canonical_signature = b"canonical-signature"
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=content_signature,
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    legacy_message["tool_calls"][0]["signature"] = base64.b64encode(
        canonical_signature
    ).decode("ascii")
    legacy_message["tool_calls"][0]["thinking_provenance"] = NATIVE_PROVENANCE

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 1
    assert function_parts[0]["thought_signature"] == base64.b64encode(
        canonical_signature
    ).decode("ascii")


def test_legacy_signature_recovery_requires_nonempty_matching_call_id():
    """A content block cannot recover a signature for an empty legacy ID."""
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    legacy_message["tool_calls"][0]["id"] = ""

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 1
    assert function_parts[0]["thought_signature"] == IMPORT_SENTINEL


def test_legacy_signature_recovery_requires_a_matching_content_block():
    """A non-matching ID cannot borrow a signature from another block."""
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    legacy_message["tool_calls"][0]["id"] = "unmatched-call-id"

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 1
    assert function_parts[0]["thought_signature"] == IMPORT_SENTINEL


def test_legacy_signature_recovery_rejects_mismatched_duplicate_call():
    """A duplicate ID with different call data cannot steal another signature."""
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    legacy_message["tool_calls"].append(
        {
            "id": legacy_message["tool_calls"][0]["id"],
            "tool": "grep",
            "arguments": {"pattern": "persist"},
        }
    )

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 2
    assert "thought_signature" in function_parts[0]
    assert "thought_signature" not in function_parts[1]


def test_legacy_signature_recovery_rejects_ambiguous_content_blocks():
    """Duplicate matching content blocks cannot establish one original signature."""
    part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [part])
    legacy_message["content"].append(dict(legacy_message["content"][0]))

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 1
    assert function_parts[0]["thought_signature"] == IMPORT_SENTINEL


def test_legacy_parallel_calls_recover_only_the_signed_sibling():
    """Unsigned parallel siblings do not receive the first call's signature."""
    signed_part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )
    unsigned_part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="grep", args={"pattern": "persist"}),
    )

    provider = _make_provider()
    legacy_message = _legacy_loop_message(provider, [signed_part, unsigned_part])

    _, gemini_contents = provider._convert_messages([legacy_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 2
    assert "thought_signature" in function_parts[0]
    assert "thought_signature" not in function_parts[1]


def test_legacy_sequential_calls_do_not_recover_across_assistant_messages():
    """A same-ID unsigned later call cannot recover an earlier message's signature."""
    signed_part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
        thought_signature=b"signed",
    )
    unsigned_part = SimpleNamespace(
        thought=False,
        function_call=SimpleNamespace(name="todo", args={"content": "persist"}),
    )

    provider = _make_provider()
    first_message = _legacy_loop_message(provider, [signed_part])
    second_message = _legacy_loop_message(provider, [unsigned_part])
    for message in (first_message, second_message):
        message["content"][0]["id"] = "same-call-id"
        message["tool_calls"][0]["id"] = "same-call-id"

    _, gemini_contents = provider._convert_messages([first_message, second_message])

    function_parts = _function_call_parts(gemini_contents)
    assert len(function_parts) == 2
    assert "thought_signature" in function_parts[0]
    assert "thought_signature" not in function_parts[1]


def test_tool_call_content_alone_does_not_reconstruct_a_function_call():
    """Content remains provenance-only; tool_calls still control emitted calls."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "todo",
                    "input": {"content": "persist"},
                    "signature": base64.b64encode(b"signed").decode("ascii"),
                }
            ],
            "tool_calls": [],
        }
    ]

    provider = _make_provider()
    _, gemini_contents = provider._convert_messages(messages)

    assert _function_call_parts(gemini_contents) == []


# ============================================================
# JSON-safety regression (the actual load-bearing bug this audit found)
# ============================================================


def test_inbound_signature_is_json_safe_not_raw_bytes():
    """Captured signatures must be JSON-safe (base64 str), never raw bytes.

    This module is stateless full-resend: the entire Message list gets
    rebuilt from the stored conversation history on every turn. That stored
    history commonly crosses a JSON boundary somewhere in the stack (session
    persistence, event logging, an orchestrator calling
    model_dump(mode="json")). A raw-bytes signature containing a byte
    sequence that isn't valid UTF-8 -- the NORMAL case for an opaque
    cryptographic signature -- crashes that serialization outright.

    Verified directly against amplifier_core's own models before this fix:
        ToolCallBlock(signature=bytes([0xff, 0xfe, 0x80])).model_dump(mode="json")
    raised UnicodeDecodeError. ThinkingBlock was never affected (its
    signature field is already typed str | None and was already encoded at
    capture time) -- only TextBlock and ToolCallBlock/ToolCall had the bug.
    """
    # A byte sequence that is NOT valid UTF-8 (the realistic case).
    sig_bytes = bytes([0xFF, 0xFE, 0x80, 0x81, 0x00, 0x9D])

    fc = SimpleNamespace(name="todo", args={"content": "x"})
    fc_part = SimpleNamespace(
        thought=False, function_call=fc, thought_signature=sig_bytes
    )
    text_part = SimpleNamespace(
        text="answer", thought=False, thought_signature=sig_bytes
    )
    response = _make_response([text_part, fc_part])

    provider = _make_provider()
    result = provider._convert_to_chat_response(response)

    text_block, tc_block = result.content[0], result.content[1]

    # Both must be plain base64 str, not bytes -- and both round-trip
    # through the exact JSON serialization path that used to crash.
    assert isinstance(text_block.signature, str)
    assert isinstance(tc_block.signature, str)
    assert isinstance(result.tool_calls[0].signature, str)

    text_block.model_dump(mode="json")
    tc_block.model_dump(mode="json")
    result.tool_calls[0].model_dump(mode="json")

    # And the value is recoverable: decoding it gives back the exact
    # original bytes (nothing lost, nothing mangled).
    assert base64.b64decode(text_block.signature) == sig_bytes
    assert base64.b64decode(tc_block.signature) == sig_bytes


NATIVE_PROVENANCE = {"version": 1, "provider": "gemini"}
IMPORT_SENTINEL = base64.b64encode(b"skip_thought_signature_validator").decode("ascii")


def _persist_response(provider, parts):
    """Use Core's real discriminated block union and both JSON dump/load paths."""
    response = provider._convert_to_chat_response(_make_response(parts))
    message = Message(
        role="assistant",
        content=response.content,
        tool_calls=[tc.model_dump() for tc in response.tool_calls or []],
    )
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored.model_dump() == json.loads(json.dumps(message.model_dump()))
    return restored.model_dump()


def test_native_interleaved_parts_survive_core_json_in_original_order():
    provider = _make_provider()
    raw_parts = [
        types.Part(text="summary", thought=True),
        types.Part(
            function_call=types.FunctionCall(name="first", args={"x": 1}),
            thought_signature=b"\xff\x00first",
        ),
        types.Part(text="between", thought_signature=b"\xfe\x00text"),
        types.Part(function_call=types.FunctionCall(name="second", args={"y": 2})),
        types.Part(text="", thought_signature=b"\x80seal"),
    ]
    message = _persist_response(provider, raw_parts)
    for block in message["content"]:
        assert block["thinking_provenance"] == NATIVE_PROVENANCE
    for call in message["tool_calls"]:
        assert call["thinking_provenance"] == NATIVE_PROVENANCE
    original = copy.deepcopy(message)
    _, contents = provider._convert_messages([message])
    assert message == original
    expected = [
        {"text": "summary", "thought": True},
        {
            "function_call": {"name": "first", "args": {"x": 1}},
            "thought_signature": base64.b64encode(b"\xff\x00first").decode("ascii"),
        },
        {
            "text": "between",
            "thought_signature": base64.b64encode(b"\xfe\x00text").decode("ascii"),
        },
        {"function_call": {"name": "second", "args": {"y": 2}}},
        {
            "text": "",
            "thought_signature": base64.b64encode(b"\x80seal").decode("ascii"),
        },
    ]
    assert contents == [{"role": "model", "parts": expected}]
    # The real SDK must decode our persisted base64 back to the exact opaque bytes.
    sdk_parts = types.Content.model_validate(contents[0]).parts
    assert sdk_parts[1].thought_signature == b"\xff\x00first"
    assert sdk_parts[-1].thought_signature == b"\x80seal"


@pytest.mark.parametrize(
    "provenance",
    [
        None,
        {"version": 1, "provider": "anthropic"},
        {"version": 1, "provider": "github-copilot"},
        {"version": 2, "provider": "gemini"},
        {"version": True, "provider": "gemini"},
        "gemini",
    ],
)
def test_foreign_or_untagged_signatures_are_not_native_replay(provenance):
    opaque = base64.b64encode(b"foreign-opaque").decode("ascii")
    extra = {"signature": opaque}
    if provenance is not None:
        extra["thinking_provenance"] = provenance
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "visible summary", **extra},
            {"type": "text", "text": "visible answer", **extra},
        ],
        "tool_calls": [
            {"id": "one", "name": "first", "arguments": {"x": 1}, **extra},
            {"id": "two", "name": "second", "arguments": {"y": 2}, **extra},
        ],
    }
    provider = _make_provider()
    original = copy.deepcopy(message)
    _, contents = provider._convert_messages([message])
    assert message == original
    assert contents[0]["parts"][:2] == [
        {"text": "visible summary"},
        {"text": "visible answer"},
    ]
    calls = _function_call_parts(contents)
    assert calls[0] == {
        "function_call": {"name": "first", "args": {"x": 1}},
        "thought_signature": IMPORT_SENTINEL,
    }
    assert calls[1] == {"function_call": {"name": "second", "args": {"y": 2}}}
    assert opaque not in json.dumps(contents)


@pytest.mark.asyncio
async def test_stream_signatures_stay_on_original_parts_including_empty_seals():
    provider = _make_provider()
    raw_parts = [
        types.Part(text="think", thought=True, thought_signature=b"thinking"),
        types.Part(text="answer", thought_signature=b"answer"),
        types.Part(
            function_call=types.FunctionCall(name="first", args={"x": 1}),
            thought_signature=b"call",
        ),
        types.Part(function_call=types.FunctionCall(name="second", args={"x": 2})),
        types.Part(text="tail"),
        types.Part(text="", thought_signature=b"\xffseal"),
    ]

    async def chunks():
        for part in raw_parts:
            yield types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(parts=[part]))]
            )

    provider._client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=AsyncMock(return_value=chunks()),
            )
        )
    )
    result = await provider.complete(
        ChatRequest(messages=[Message(role="user", content="go")])
    )
    dumped = Message(
        role="assistant",
        content=result.content,
        tool_calls=[tc.model_dump() for tc in result.tool_calls],
    ).model_dump(mode="json")
    _, contents = provider._convert_messages([dumped])
    parts = contents[0]["parts"]
    assert [p.get("thought_signature") for p in parts] == [
        base64.b64encode(sig).decode("ascii") if sig else None
        for sig in (b"thinking", b"answer", b"call", None, None, b"\xffseal")
    ]
    assert [p.get("text") for p in parts] == ["think", "answer", None, None, "tail", ""]


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "models/gemini-4-pro"])
def test_known_native_missing_signature_is_not_fabricated(model):
    provider = _make_provider()
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "one",
                "name": "first",
                "arguments": {},
                "thinking_provenance": NATIVE_PROVENANCE,
            }
        ],
    }
    _, contents = provider._convert_messages([message], model=model)
    assert _function_call_parts(contents) == [
        {"function_call": {"name": "first", "args": {}}}
    ]


def test_import_to_legacy_model_drops_opaque_signature_without_sentinel():
    provider = _make_provider()
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "one",
                "name": "first",
                "arguments": {},
                "signature": "foreign-opaque",
                "thinking_provenance": {"version": 1, "provider": "anthropic"},
            }
        ],
    }
    _, contents = provider._convert_messages([message], model="gemini-2.5-pro")
    assert _function_call_parts(contents) == [
        {"function_call": {"name": "first", "args": {}}}
    ]


@pytest.mark.parametrize("foreign_on", ["block", "call"])
def test_duplicate_representation_cannot_launder_foreign_provenance(foreign_on):
    provider = _make_provider()
    message = _legacy_loop_message(
        provider,
        [
            types.Part(
                function_call=types.FunctionCall(name="first", args={"x": 1}),
                thought_signature=b"native",
            )
        ],
    )
    target = (
        message["content"][0] if foreign_on == "block" else message["tool_calls"][0]
    )
    target["thinking_provenance"] = {"version": 1, "provider": "github-copilot"}
    _, contents = provider._convert_messages([message])
    assert _function_call_parts(contents)[0]["thought_signature"] == IMPORT_SENTINEL


def test_native_content_order_wins_over_reordered_duplicate_call_list():
    provider = _make_provider()
    message = _legacy_loop_message(
        provider,
        [
            types.Part(
                function_call=types.FunctionCall(name="first", args={"x": 1}),
                thought_signature=b"first",
            ),
            types.Part(text="between"),
            types.Part(function_call=types.FunctionCall(name="second", args={"x": 2})),
        ],
    )
    message["tool_calls"].reverse()
    _, contents = provider._convert_messages([message])
    assert contents[0]["parts"] == [
        {
            "function_call": {"name": "first", "args": {"x": 1}},
            "thought_signature": base64.b64encode(b"first").decode("ascii"),
        },
        {"text": "between"},
        {"function_call": {"name": "second", "args": {"x": 2}}},
    ]


def test_native_duplicate_can_recover_signature_if_only_signature_was_lost():
    provider = _make_provider()
    message = _legacy_loop_message(
        provider,
        [
            types.Part(
                function_call=types.FunctionCall(name="first", args={"x": 1}),
                thought_signature=b"native",
            )
        ],
    )
    message["tool_calls"][0]["thinking_provenance"] = NATIVE_PROVENANCE
    message["tool_calls"][0]["signature"] = None
    _, contents = provider._convert_messages([message])
    assert _function_call_parts(contents)[0]["thought_signature"] == base64.b64encode(
        b"native"
    ).decode("ascii")


@pytest.mark.asyncio
@pytest.mark.parametrize("thought", [False, True])
async def test_stream_signature_only_seals_are_not_lost_or_moved(thought):
    provider = _make_provider()

    async def chunks():
        for part in [
            types.Part(text="before", thought=thought),
            types.Part(thought=thought, thought_signature=b"\xffseal"),
            types.Part(text="after", thought=thought),
        ]:
            yield types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(parts=[part]))]
            )

    provider._client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=AsyncMock(return_value=chunks()),
            )
        )
    )
    result = await provider.complete(
        ChatRequest(messages=[Message(role="user", content="go")])
    )
    blocks = json.loads(result.model_dump_json())["content"]
    _, contents = provider._convert_messages([{"role": "assistant", "content": blocks}])
    parts = contents[0]["parts"]
    assert [p["text"] for p in parts] == ["before", "", "after"]
    assert [p.get("thought_signature") for p in parts] == [
        None,
        base64.b64encode(b"\xffseal").decode("ascii"),
        None,
    ]
    assert [p.get("thought", False) for p in parts] == [thought] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-4-pro"])
async def test_sdk_wire_replays_all_native_steps_and_imports_foreign_step(model):
    """One offline SDK send, with native signatures and import sentinel distinct."""
    provider = _make_provider()
    first = _persist_response(
        provider,
        [
            types.Part(
                function_call=types.FunctionCall(name="first", args={"x": 1}),
                thought_signature=b"\xffnative-one",
            ),
            types.Part(function_call=types.FunctionCall(name="second", args={"x": 2})),
        ],
    )
    second = _persist_response(
        provider,
        [
            types.Part(text="next"),
            types.Part(
                function_call=types.FunctionCall(name="third", args={"x": 3}),
                thought_signature=b"\x80native-two",
            ),
        ],
    )
    foreign = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "imported summary",
                "signature": "FOREIGN-OPAQUE",
                "thinking_provenance": {"version": 1, "provider": "anthropic"},
            }
        ],
        "tool_calls": [
            {
                "id": "foreign-one",
                "name": "fourth",
                "arguments": {"x": 4},
                "signature": "FOREIGN-OPAQUE",
                "thinking_provenance": {"version": 1, "provider": "github-copilot"},
            },
            {"id": "foreign-two", "name": "fifth", "arguments": {"x": 5}},
        ],
    }
    messages = [Message(role="user", content="go")]
    for message in (first, second, foreign):
        messages.append(Message.model_validate(message))
        for call in message["tool_calls"]:
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call["id"],
                    content=f"result-{call['name']}",
                )
            )
    request = ChatRequest(messages=messages, metadata={"stream": False})
    original = request.model_dump_json()
    sent = []

    def receive(http_request):
        sent.append(json.loads(http_request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "done"}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    client = genai.Client(
        api_key="synthetic-key",
        http_options=types.HttpOptions(
            base_url="https://unit.test/",
            async_client_args={"transport": httpx.MockTransport(receive)},
        ),
    )
    provider._client = client
    try:
        await provider.complete(request, model=model)
        assert request.model_dump_json() == original
        assert len(sent) == 1
        wire = sent[0]["contents"]
        calls = [p for c in wire for p in c["parts"] if "functionCall" in p]
        assert [p["functionCall"] for p in calls] == [
            {"name": name, "args": {"x": index}}
            for index, name in enumerate(
                ("first", "second", "third", "fourth", "fifth"), 1
            )
        ]
        # SDK 2.30 serializes bytes as URL-safe base64; stored Core signatures
        # use standard base64. Compare the opaque bytes, not alphabet spelling.
        assert [
            base64.urlsafe_b64decode(p["thoughtSignature"])
            if p.get("thoughtSignature")
            else None
            for p in calls
        ] == [
            b"\xffnative-one",
            None,
            b"\x80native-two",
            b"skip_thought_signature_validator",
            None,
        ]
        results = [
            p["functionResponse"]
            for c in wire
            for p in c["parts"]
            if "functionResponse" in p
        ]
        assert results == [
            {"name": name, "response": {"result": f"result-{name}"}}
            for name in ("first", "second", "third", "fourth", "fifth")
        ]
        assert "thinking_provenance" not in json.dumps(sent)
        assert "FOREIGN-OPAQUE" not in json.dumps(sent)
        assert (
            base64.b64decode(calls[3]["thoughtSignature"])
            == b"skip_thought_signature_validator"
        )
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
async def test_sdk_stream_keeps_distinct_signed_text_parts_and_final_empty_seal():
    """Google FAQ's empty-text signature part is not a delta on the prior part."""
    payloads = [
        {
            "text": "first",
            "thoughtSignature": base64.b64encode(b"\xffone").decode("ascii"),
        },
        {
            "text": "second",
            "thoughtSignature": base64.b64encode(b"\xfetwo").decode("ascii"),
        },
        {"text": "", "thoughtSignature": base64.b64encode(b"\x80seal").decode("ascii")},
    ]
    chunks = [
        {"candidates": [{"content": {"role": "model", "parts": [part]}}]}
        for part in payloads
    ]
    chunks.append({"candidates": [{"finishReason": "STOP"}]})
    sends = []

    def receive(request):
        sends.append(request.url)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks),
        )

    client = genai.Client(
        api_key="synthetic-key",
        http_options=types.HttpOptions(
            base_url="https://unit.test/",
            async_client_args={"transport": httpx.MockTransport(receive)},
        ),
    )
    provider = _make_provider()
    provider._client = client
    try:
        result = await provider.complete(
            ChatRequest(messages=[Message(role="user", content="go")])
        )
        assert len(sends) == 1
        assert result.finish_reason == "stop"
        restored = Message.model_validate_json(
            Message(role="assistant", content=result.content).model_dump_json()
        )
        _, contents = provider._convert_messages([restored.model_dump()])
        parts = contents[0]["parts"]
        assert [p["text"] for p in parts] == ["first", "second", ""]
        assert [base64.b64decode(p["thought_signature"]) for p in parts] == [
            b"\xffone",
            b"\xfetwo",
            b"\x80seal",
        ]
        assert all("thought" not in p for p in parts)
    finally:
        await client.aio.aclose()
        client.close()
