"""Tests for proper token streaming — driven by the provider-streaming-contract.md.

TDD: these tests were written FIRST and drove the implementation.
All assertions reference exact event names and payload shapes from the contract.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_gemini import GeminiProvider


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_usage(prompt=10, candidates=20, total=30, thoughts=None, cached=None):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached,
    )


def _part(text=None, thought=False, function_call=None, thought_signature=None):
    """Construct a minimal fake SDK part (SimpleNamespace)."""
    ns = SimpleNamespace(thought=thought)
    if text is not None:
        ns.text = text
    if function_call is not None:
        ns.function_call = function_call
    if thought_signature is not None:
        ns.thought_signature = thought_signature
    return ns


def _fc(name="do_thing", args=None):
    return SimpleNamespace(name=name, args=args or {"x": 1})


def _chunk(parts, usage_metadata=None):
    """Wrap parts into a minimal fake GenerateContentResponse chunk."""
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content)
    ns = SimpleNamespace(candidates=[candidate])
    if usage_metadata is not None:
        ns.usage_metadata = usage_metadata
    return ns


def _make_provider(config=None) -> tuple:
    cfg = {"max_retries": 0, **(config or {})}
    provider = GeminiProvider(api_key="test-key", config=cfg)
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)

    # Minimal blocking-path mock (used by stream=False tests)
    blocking_response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="hello", thought=False)]
                )
            )
        ],
        usage_metadata=_make_usage(),
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=blocking_response)

    async def _empty_stream_gen():
        return
        yield  # pragma: no cover

    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=_empty_stream_gen()
    )
    provider._client = mock_client
    return provider, coordinator


def _stream_events(coordinator) -> list:
    return [(n, p) for n, p in coordinator.hooks.events if n.startswith("llm:stream_")]


def _by_name(coordinator, name: str) -> list:
    return [p for n, p in coordinator.hooks.events if n == name]


# ---------------------------------------------------------------------------
# Contract: use_streaming default and per-request override
# ---------------------------------------------------------------------------


def test_use_streaming_default_is_true():
    """use_streaming defaults to True."""
    provider, _ = _make_provider()
    assert provider.use_streaming is True


def test_use_streaming_config_false():
    """use_streaming=False in config disables streaming globally."""
    provider, _ = _make_provider(config={"use_streaming": False})
    assert provider.use_streaming is False


@pytest.mark.asyncio
async def test_stream_false_metadata_uses_blocking_path():
    """request.metadata={'stream': False} -> blocking path; zero llm:stream_* events.

    Identity check: 'stream' is False (not falsy); None must NOT disable streaming.
    """
    provider, coordinator = _make_provider()

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        metadata={"stream": False},
    )
    await provider.complete(request)

    assert _stream_events(coordinator) == [], "No stream events on blocking path"
    assert _by_name(coordinator, "llm:request"), "llm:request must still fire"
    assert _by_name(coordinator, "llm:response"), "llm:response must still fire"
    provider._client.aio.models.generate_content.assert_called_once()
    provider._client.aio.models.generate_content_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_none_metadata_uses_streaming_path():
    """metadata={'stream': None} does NOT disable streaming (identity, not truthiness)."""
    provider, coordinator = _make_provider()

    chunks = [
        _chunk([_part(text="Hi", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        metadata={"stream": None},
    )
    await provider.complete(request)

    provider._client.aio.models.generate_content_stream.assert_called_once()
    provider._client.aio.models.generate_content.assert_not_called()


# ---------------------------------------------------------------------------
# Contract: single shared request_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_request_id_across_all_stream_events():
    """All stream events for one call share ONE request_id (uuid4 string)."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="Hi", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="go")]))

    evts = _stream_events(coordinator)
    assert evts, "Expected at least one stream event"
    request_ids = {p["request_id"] for _, p in evts}
    assert len(request_ids) == 1, f"Multiple request_ids: {request_ids}"
    rid = next(iter(request_ids))
    assert rid and len(rid) == 36, f"Not a uuid4 string: {rid!r}"


# ---------------------------------------------------------------------------
# Contract: thinking-only block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_only_produces_correct_events():
    """Thinking parts -> block_start(thinking,0) + thinking_deltas + block_end(thinking,0)."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="I think...", thought=True)]),
        _chunk([_part(text=" therefore I am.", thought=True)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="think")]))

    names = [n for n, _ in _stream_events(coordinator)]
    assert "llm:stream_block_start" in names
    assert "llm:stream_thinking_delta" in names
    assert "llm:stream_block_end" in names
    assert "llm:stream_block_delta" not in names

    starts = _by_name(coordinator, "llm:stream_block_start")
    assert len(starts) == 1
    assert starts[0]["block_index"] == 0
    assert starts[0]["block_type"] == "thinking"

    deltas = _by_name(coordinator, "llm:stream_thinking_delta")
    assert len(deltas) == 2
    assert deltas[0]["text"] == "I think..."
    assert deltas[1]["text"] == " therefore I am."
    assert deltas[0]["sequence"] == 0
    assert deltas[1]["sequence"] == 1
    assert all(d["block_index"] == 0 for d in deltas)

    ends = _by_name(coordinator, "llm:stream_block_end")
    assert len(ends) == 1
    assert ends[0]["block_index"] == 0
    assert ends[0]["block_type"] == "thinking"


# ---------------------------------------------------------------------------
# Contract: text-only block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_produces_correct_events():
    """Text parts -> block_start(text,0) + block_deltas + block_end(text,0)."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="Hello", thought=False)]),
        _chunk([_part(text=" world", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="hi")]))

    names = [n for n, _ in _stream_events(coordinator)]
    assert "llm:stream_block_start" in names
    assert "llm:stream_block_delta" in names
    assert "llm:stream_block_end" in names
    assert "llm:stream_thinking_delta" not in names

    starts = _by_name(coordinator, "llm:stream_block_start")
    assert len(starts) == 1
    assert starts[0]["block_index"] == 0
    assert starts[0]["block_type"] == "text"

    deltas = _by_name(coordinator, "llm:stream_block_delta")
    assert len(deltas) == 2
    assert deltas[0]["text"] == "Hello"
    assert deltas[1]["text"] == " world"
    assert deltas[0]["sequence"] == 0
    assert deltas[1]["sequence"] == 1

    ends = _by_name(coordinator, "llm:stream_block_end")
    assert len(ends) == 1
    assert ends[0]["block_index"] == 0
    assert ends[0]["block_type"] == "text"


# ---------------------------------------------------------------------------
# Contract: block type transitions (synthesized boundaries)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_then_text_transition_synthesizes_boundaries():
    """Thinking -> text: synthesize block_end(thinking) + block_start(text)."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="Hmm...", thought=True)]),
        _chunk([_part(text="Answer.", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="solve")]))

    stream_names = [n for n, _ in _stream_events(coordinator)]

    # Ordering: block_start(thinking) < thinking_delta < block_end(thinking) < block_delta
    first_start = stream_names.index("llm:stream_block_start")
    first_thinking = stream_names.index("llm:stream_thinking_delta")
    first_end = stream_names.index("llm:stream_block_end")
    first_delta = stream_names.index("llm:stream_block_delta")
    assert first_start < first_thinking < first_end < first_delta, (
        f"Wrong ordering: {stream_names}"
    )

    starts = _by_name(coordinator, "llm:stream_block_start")
    assert len(starts) == 2
    assert starts[0]["block_index"] == 0
    assert starts[0]["block_type"] == "thinking"
    assert starts[1]["block_index"] == 1
    assert starts[1]["block_type"] == "text"

    ends = _by_name(coordinator, "llm:stream_block_end")
    assert len(ends) == 2
    assert ends[0]["block_index"] == 0
    assert ends[0]["block_type"] == "thinking"
    assert ends[1]["block_index"] == 1
    assert ends[1]["block_type"] == "text"

    text_deltas = _by_name(coordinator, "llm:stream_block_delta")
    assert len(text_deltas) == 1
    assert text_deltas[0]["block_index"] == 1
    assert text_deltas[0]["sequence"] == 0  # resets per block
    assert text_deltas[0]["text"] == "Answer."


# ---------------------------------------------------------------------------
# Contract: multiple parts in ONE chunk (state machine)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_parts_in_one_chunk_all_processed():
    """A single chunk with [thinking_part, text_part] processes ALL parts (not just first)."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk(
            [
                _part(text="Think", thought=True),
                _part(text="Answer", thought=False),
            ],
            usage_metadata=_make_usage(),
        ),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="go")]))

    starts = _by_name(coordinator, "llm:stream_block_start")
    ends = _by_name(coordinator, "llm:stream_block_end")
    assert len(starts) == 2, f"Expected 2 starts, got {len(starts)}: {starts}"
    assert len(ends) == 2, f"Expected 2 ends, got {len(ends)}: {ends}"
    assert starts[0]["block_type"] == "thinking"
    assert starts[1]["block_type"] == "text"


# ---------------------------------------------------------------------------
# Contract: empty fragments never emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_text_fragment_not_emitted_as_delta():
    """Empty text fragments are never emitted (contract: guard every delta with if text:)."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="", thought=False)]),    # empty — skip
        _chunk([_part(text="Real", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="hi")]))

    text_deltas = _by_name(coordinator, "llm:stream_block_delta")
    assert all(d["text"] for d in text_deltas), "Empty delta was emitted"
    assert len(text_deltas) == 1
    assert text_deltas[0]["text"] == "Real"


# ---------------------------------------------------------------------------
# Contract: heartbeat chunks (empty parts list)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_chunks_handled_gracefully():
    """Chunks with empty parts list are silently skipped; no spurious events."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([]),  # heartbeat
        _chunk([_part(text="Hello", thought=False)]),
        _chunk([]),  # another heartbeat
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="hi")]))

    starts = _by_name(coordinator, "llm:stream_block_start")
    assert len(starts) == 1
    assert starts[0]["block_type"] == "text"


# ---------------------------------------------------------------------------
# Contract: per-block sequence counter (resets per block)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequence_counter_is_per_block_and_resets():
    """Sequence is per-block 0-based. New block resets to 0."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="A", thought=True)]),
        _chunk([_part(text="B", thought=True)]),   # seq=1 in thinking block
        _chunk([_part(text="C", thought=False)]),  # new block, seq resets to 0
        _chunk([_part(text="D", thought=False)]),  # seq=1 in text block
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="go")]))

    thinking_deltas = _by_name(coordinator, "llm:stream_thinking_delta")
    assert len(thinking_deltas) == 2
    assert thinking_deltas[0]["sequence"] == 0
    assert thinking_deltas[1]["sequence"] == 1

    text_deltas = _by_name(coordinator, "llm:stream_block_delta")
    assert len(text_deltas) == 2
    assert text_deltas[0]["sequence"] == 0  # reset
    assert text_deltas[1]["sequence"] == 1


# ---------------------------------------------------------------------------
# Contract: tool_use blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_function_call_emits_tool_use_block_no_deltas():
    """Function call -> block_start('tool_use', name=...) + block_end; NO deltas.
    Tool call must appear in ChatResponse.tool_calls.
    """
    provider, coordinator = _make_provider()
    fc = _fc(name="get_weather", args={"city": "Seattle"})
    chunks = [
        _chunk([_part(function_call=fc)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    response = await provider.complete(
        ChatRequest(messages=[Message(role="user", content="weather?")])
    )

    starts = _by_name(coordinator, "llm:stream_block_start")
    ends = _by_name(coordinator, "llm:stream_block_end")
    assert len(starts) == 1
    assert starts[0]["block_type"] == "tool_use"
    assert starts[0]["name"] == "get_weather"
    assert len(ends) == 1
    assert ends[0]["block_type"] == "tool_use"

    assert not _by_name(coordinator, "llm:stream_block_delta"), "No text deltas for tool_use"
    assert not _by_name(coordinator, "llm:stream_thinking_delta"), "No thinking deltas for tool_use"

    assert response.tool_calls, "Tool call must be in ChatResponse"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Seattle"}


@pytest.mark.asyncio
async def test_text_then_function_call_block_indices_increment():
    """Text block (idx=0) + tool_use block (idx=1): shared 0-based block_index space."""
    provider, coordinator = _make_provider()
    fc = _fc(name="search")
    chunks = [
        _chunk([_part(text="Looking...", thought=False)]),
        _chunk([_part(function_call=fc)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="search")]))

    starts = _by_name(coordinator, "llm:stream_block_start")
    assert len(starts) == 2
    assert starts[0]["block_index"] == 0
    assert starts[0]["block_type"] == "text"
    assert starts[1]["block_index"] == 1
    assert starts[1]["block_type"] == "tool_use"


# ---------------------------------------------------------------------------
# Contract: llm:stream_aborted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_after_partial_emits_stream_aborted():
    """Exception after at least one delta -> llm:stream_aborted before re-raise."""
    provider, coordinator = _make_provider()

    async def _gen():
        yield _chunk([_part(text="Hello", thought=False)])
        raise RuntimeError("Network blip")

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    with pytest.raises(Exception):
        await provider.complete(ChatRequest(messages=[Message(role="user", content="hi")]))

    aborted = _by_name(coordinator, "llm:stream_aborted")
    assert len(aborted) == 1, f"Expected one aborted event, got: {aborted}"
    assert "request_id" in aborted[0]
    assert "error" in aborted[0]
    assert aborted[0]["error"]["type"] == "RuntimeError"
    assert "Network blip" in aborted[0]["error"]["msg"]


@pytest.mark.asyncio
async def test_error_before_any_delta_no_stream_aborted():
    """Exception before any delta -> NO llm:stream_aborted (partial_emitted=False)."""
    provider, coordinator = _make_provider()

    async def _gen():
        raise RuntimeError("Immediate error")
        yield  # pragma: no cover

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    with pytest.raises(Exception):
        await provider.complete(ChatRequest(messages=[Message(role="user", content="hi")]))

    aborted = _by_name(coordinator, "llm:stream_aborted")
    assert len(aborted) == 0, (
        f"Should NOT emit aborted when no delta emitted, got: {aborted}"
    )


# ---------------------------------------------------------------------------
# Contract: llm:request + llm:response still fire on streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_still_emits_llm_request_and_response():
    """Streaming path emits llm:request and llm:response (not just stream events)."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="Hi", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="hello")]))

    assert _by_name(coordinator, "llm:request"), "llm:request must fire"
    assert _by_name(coordinator, "llm:response"), "llm:response must fire"


# ---------------------------------------------------------------------------
# Contract: streaming path uses generate_content_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_path_calls_generate_content_stream():
    """Default (use_streaming=True) -> generate_content_stream used, not generate_content."""
    provider, coordinator = _make_provider()
    assert provider.use_streaming is True

    chunks = [
        _chunk([_part(text="Hi", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    await provider.complete(ChatRequest(messages=[Message(role="user", content="hi")]))

    provider._client.aio.models.generate_content_stream.assert_called_once()
    provider._client.aio.models.generate_content.assert_not_called()


# ---------------------------------------------------------------------------
# Contract: assembled ChatResponse is valid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_assembles_valid_chat_response():
    """Streaming path returns a valid ChatResponse with content and usage."""
    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="A", thought=True)]),
        _chunk([_part(text="Answer here.", thought=False)]),
        _chunk([], usage_metadata=_make_usage(prompt=5, candidates=10, total=15)),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    response = await provider.complete(
        ChatRequest(messages=[Message(role="user", content="solve")])
    )

    assert response is not None
    assert response.content, "ChatResponse must have content"
    assert response.usage is not None, "ChatResponse must have usage"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_streaming_thinking_appears_in_content():
    """Thinking parts appear as ThinkingBlock in the assembled ChatResponse."""
    from amplifier_core.message_models import ThinkingBlock

    provider, coordinator = _make_provider()
    chunks = [
        _chunk([_part(text="I reason...", thought=True)]),
        _chunk([_part(text="Final answer.", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    response = await provider.complete(
        ChatRequest(messages=[Message(role="user", content="think")])
    )

    block_types = [type(b).__name__ for b in (response.content or [])]
    assert "ThinkingBlock" in block_types, f"Expected ThinkingBlock: {block_types}"
    assert "TextBlock" in block_types, f"Expected TextBlock: {block_types}"


# ---------------------------------------------------------------------------
# Contract: semaphore held for whole stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semaphore_held_for_whole_stream_no_error():
    """Smoke: streaming path completes without error with max_concurrent_requests=1."""
    provider, coordinator = _make_provider(config={"max_concurrent_requests": 1})

    chunks = [
        _chunk([_part(text="A", thought=False)]),
        _chunk([_part(text="B", thought=False)]),
        _chunk([], usage_metadata=_make_usage()),
    ]

    async def _gen():
        for c in chunks:
            yield c

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

    response = await provider.complete(
        ChatRequest(messages=[Message(role="user", content="hi")])
    )
    assert response is not None
