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
    part = SimpleNamespace(text="final answer", thought=False, thought_signature=sig_bytes)
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
    assert "thought_signature" not in function_parts[0]


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
    assert "thought_signature" not in function_parts[0]


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
    assert "thought_signature" not in function_parts[0]


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
    fc_part = SimpleNamespace(thought=False, function_call=fc, thought_signature=sig_bytes)
    text_part = SimpleNamespace(text="answer", thought=False, thought_signature=sig_bytes)
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
