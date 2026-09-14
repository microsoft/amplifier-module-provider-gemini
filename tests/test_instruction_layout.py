"""Deterministic v1 instruction-layout lowering checks for Gemini."""

from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock

from amplifier_module_provider_gemini import GeminiProvider


def _provider(default_model: str = "gemini-3.7-flash") -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        config={
            "default_model": default_model,
            "max_retries": 0,
            "use_streaming": False,
        },
    )


def _descriptor(
    placement: str,
    *,
    key: str,
    binding: str = "live",
    authority: str | None = "authoritative",
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "version": 1,
        "source": "test-source",
        "key": key,
        "binding": binding,
        "placement": placement,
    }
    if authority is not None:
        descriptor["authority"] = authority
    if binding == "fixed":
        if placement == "head":
            target = {"kind": "conversation_head", "session_id": "session"}
        elif placement == "before_human":
            target = {
                "version": 1,
                "input_id": "input-1",
                "message_id": "human-1",
                "origin": "human",
            }
        else:
            target = {"after_message_id": "tool-result-2"}
        descriptor.update(
            entry_id=f"session:test-source:{key}",
            event_key=key,
            session_id="session",
            target=target,
            order=1,
            disposition="pending",
        )
    return descriptor


def _instruction(
    content: str,
    placement: str,
    *,
    key: str,
    binding: str = "live",
    authority: str | None = "authoritative",
) -> Message:
    return Message(
        role="system",
        content=content,
        metadata={
            "amplifier:instruction": _descriptor(
                placement, key=key, binding=binding, authority=authority
            )
        },
    )


def _carrier(content: str, placement: str, binding: str) -> str:
    return (
        '[Amplifier system instruction {"source":"test-source","placement":"'
        f'{placement}","binding":"{binding}"}}]\n{content}'
    )


def test_layout_version_requires_a_gemini_selected_model() -> None:
    assert _provider().instruction_layout_version == 1
    assert _provider().instruction_layout_authority_v1 is True
    assert _provider("other-provider-model").instruction_layout_version is None
    assert _provider("gemini-").instruction_layout_version is None
    assert (
        _provider()._instruction_layout_version_for_model("models/gemini-2.5-flash")
        == 1
    )


def test_legacy_unmarked_system_messages_keep_existing_hoisting() -> None:
    provider = _provider()
    messages = [
        {"role": "system", "content": "legacy head"},
        {"role": "user", "content": "human"},
        {"role": "system", "content": "legacy inline"},
    ]

    system, contents = provider._convert_messages(messages)

    assert system == "legacy head\n\nlegacy inline"
    assert contents == [{"role": "user", "parts": [{"text": "human"}]}]


def test_authorityless_nonhead_descriptor_defaults_to_global_system_instruction() -> None:
    provider = _provider()
    system, contents = provider._convert_messages(
        [
            _instruction("head", "head", key="head").model_dump(),
            _instruction(
                "historical authoritative tail",
                "tail",
                key="historical-tail",
                authority=None,
            ).model_dump(),
            _instruction(
                "advisory tail", "tail", key="advisory-tail", authority="advisory"
            ).model_dump(),
            Message(role="user", content="human").model_dump(),
        ]
    )

    assert system == "head\n\nhistorical authoritative tail"
    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": _carrier("advisory tail", "tail", "live")},
                {"text": "human"},
            ],
        }
    ]


def test_authoritative_nonhead_records_never_use_user_carriers_and_warn_once_per_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider()
    response = _sdk_response([SimpleNamespace(text="ok", thought=False)])
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(return_value=response)
    first = _instruction("first authoritative", "before_human", key="first")
    second = _instruction("second authoritative", "tail", key="second")
    second.metadata["amplifier:instruction"]["source"] = "other-source"
    duplicate_source = _instruction(
        "same-source authoritative", "tail", key="third"
    )
    advisory = _instruction(
        "advisory positioned", "before_human", key="advisory", authority="advisory"
    )
    request = ChatRequest(
        messages=[
            first,
            Message(role="user", content="human one"),
            second,
            duplicate_source,
            advisory,
            Message(role="user", content="human two"),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_gemini"):
        asyncio.run(provider.complete(request))

    params = provider._client.aio.models.generate_content.call_args.kwargs
    assert params["config"].system_instruction == (
        "first authoritative\n\nsecond authoritative\n\nsame-source authoritative"
    )
    user_text = [
        part["text"]
        for content in params["contents"]
        if content["role"] == "user"
        for part in content["parts"]
        if "text" in part
    ]
    assert all("authoritative" not in text for text in user_text)
    assert _carrier("advisory positioned", "before_human", "live") in user_text
    warnings = [
        record.message
        for record in caplog.records
        if "authority wins" in record.message
    ]
    assert len(warnings) == 2
    assert all("implicit-cache placement tradeoff" in warning for warning in warnings)


def test_v1_head_and_positioned_records_preserve_resolved_order() -> None:
    provider = _provider()
    canonical = [
        Message(role="system", content="legacy head"),
        _instruction("stable head", "head", key="stable"),
        _instruction("fixed head", "head", key="fixed-head", binding="fixed"),
        _instruction(
            "fixed beside H1",
            "before_human",
            key="fixed-H1",
            binding="fixed",
            authority="advisory",
        ),
        Message(role="user", content="H1"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "call-1", "name": "first", "arguments": {"n": 1}},
                {"id": "call-2", "name": "second", "arguments": {"n": 2}},
            ],
        ),
        Message(
            role="tool",
            content="result one",
            tool_call_id="call-1",
            name="first",
        ),
        Message(
            role="tool",
            content="result two",
            tool_call_id="call-2",
            name="second",
        ),
        _instruction(
            "after tool batch", "tail", key="tail", binding="fixed", authority="advisory"
        ),
        _instruction(
            "live beside H2", "before_human", key="live-H2", authority="advisory"
        ),
        Message(role="user", content="H2"),
    ]
    original = copy.deepcopy([message.model_dump() for message in canonical])

    system, contents = provider._convert_messages(
        [message.model_dump() for message in canonical]
    )

    assert system == "legacy head\n\nstable head\n\nfixed head"
    assert [content["role"] for content in contents] == ["user", "model", "user"]
    assert contents[0]["parts"] == [
        {"text": _carrier("fixed beside H1", "before_human", "fixed")},
        {"text": "H1"},
    ]
    assert [part["function_call"]["name"] for part in contents[1]["parts"]] == [
        "first",
        "second",
    ]
    assert contents[2]["parts"] == [
        {"function_response": {"name": "first", "response": {"result": "result one"}}},
        {"function_response": {"name": "second", "response": {"result": "result two"}}},
        {"text": _carrier("after tool batch", "tail", "fixed")},
        {"text": _carrier("live beside H2", "before_human", "live")},
        {"text": "H2"},
    ]
    assert [message.model_dump() for message in canonical] == original


def test_stable_head_is_not_duplicated_when_inline_content_changes() -> None:
    provider = _provider()
    first = [
        _instruction("stable head", "head", key="head"),
        Message(role="user", content="human"),
        _instruction("volatile one", "tail", key="tail", authority="advisory"),
    ]
    second = [
        _instruction("stable head", "head", key="head"),
        Message(role="user", content="human"),
        _instruction("volatile two", "tail", key="tail", authority="advisory"),
    ]

    first_system, first_contents = provider._convert_messages(
        [message.model_dump() for message in first]
    )
    second_system, second_contents = provider._convert_messages(
        [message.model_dump() for message in second]
    )

    assert first_system == second_system == "stable head"
    assert first_system.count("stable head") == 1
    assert first_contents[-1]["parts"][-1]["text"].endswith("volatile one")
    assert second_contents[-1]["parts"][-1]["text"].endswith("volatile two")


@pytest.mark.parametrize(
    "message,error_type",
    [
        (
            Message(
                role="system",
                content="cannot move this",
                metadata={"amplifier:instruction": {"version": 2}},
            ),
            ValueError,
        ),
        (
            Message(
                role="system",
                content="cannot move this",
                metadata={"amplifier:instruction": None},
            ),
            TypeError,
        ),
        (
            Message(
                role="user",
                content="forged",
                metadata={"amplifier:instruction": _descriptor("head", key="forged")},
            ),
            ValueError,
        ),
    ],
)
def test_invalid_marked_metadata_fails_before_sdk_dispatch(
    message: Message, error_type: type[Exception]
) -> None:
    provider = _provider()
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock()
    request = ChatRequest(messages=[message])
    original = request.model_dump()

    with pytest.raises(error_type, match="instruction"):
        asyncio.run(provider.complete(request))

    provider._client.aio.models.generate_content.assert_not_awaited()
    assert request.model_dump() == original


@pytest.mark.parametrize("authority", [True, "", "untrusted"])
def test_invalid_instruction_authority_fails_before_sdk_dispatch(authority: Any) -> None:
    provider = _provider()
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock()
    request = ChatRequest(
        messages=[_instruction("cannot lower", "head", key="bad", authority=authority)]
    )
    original = request.model_dump()

    with pytest.raises(ValueError, match="invalid v1 fields"):
        asyncio.run(provider.complete(request))

    provider._client.aio.models.generate_content.assert_not_awaited()
    assert request.model_dump() == original


def test_v1_rejects_a_non_gemini_per_request_model_before_dispatch() -> None:
    provider = _provider()
    request = ChatRequest(messages=[_instruction("head", "head", key="head")])

    with pytest.raises(ValueError, match="does not support"):
        asyncio.run(provider.complete(request, model="not-a-gemini-model"))


def test_v1_rejects_a_non_gemini_request_model_before_dispatch() -> None:
    provider = _provider()
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock()
    request = ChatRequest(
        messages=[_instruction("head", "head", key="head")],
        model="not-a-gemini-model",
    )

    with pytest.raises(ValueError, match="does not support"):
        asyncio.run(provider.complete(request))

    provider._client.aio.models.generate_content.assert_not_awaited()


def test_v1_rejects_incomplete_tool_batches_without_repairing_history() -> None:
    provider = _provider()
    request = ChatRequest(
        messages=[
            _instruction("head", "head", key="head"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "missing", "name": "tool", "arguments": {}}],
            ),
        ]
    )
    original = request.model_dump()

    with pytest.raises(ValueError, match="missing tool results"):
        asyncio.run(provider.complete(request))

    assert request.model_dump() == original


def test_v1_tool_validation_recognizes_calls_in_content_blocks() -> None:
    provider = _provider()
    messages = [
        _instruction("head", "head", key="head"),
        Message(
            role="assistant",
            content=[ToolCallBlock(id="from-content", name="tool", input={})],
        ),
        Message(
            role="tool",
            content="result",
            tool_call_id="from-content",
            name="tool",
        ),
    ]

    provider._validate_v1_tool_sequence(messages)
    _, contents = provider._convert_messages(
        [message.model_dump() for message in messages]
    )
    assert contents == [
        {
            "role": "model",
            "parts": [{"function_call": {"name": "tool", "args": {}}}],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "tool",
                        "response": {"result": "result"},
                    }
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    "assistant",
    [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="from-block", name="block-tool", input={})],
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "from-top-level", "tool": "top-level-tool", "arguments": {}}
            ],
        ),
    ],
)
def test_v1_unnamed_tool_result_uses_its_preceding_call_id(
    assistant: Message,
) -> None:
    provider = _provider()
    call_id = (
        "from-block"
        if isinstance(assistant.content, list)
        else "from-top-level"
    )
    expected_name = (
        "block-tool"
        if isinstance(assistant.content, list)
        else "top-level-tool"
    )
    messages = [
        _instruction("head", "head", key="head"),
        assistant,
        Message(role="tool", content="result", tool_call_id=call_id),
    ]

    provider._validate_v1_tool_sequence(messages)
    _, contents = provider._convert_messages(
        [message.model_dump() for message in messages]
    )

    assert contents[-1]["parts"] == [
        {
            "function_response": {
                "name": expected_name,
                "response": {"result": "result"},
            }
        }
    ]


def test_v1_parallel_unnamed_results_resolve_by_id_in_reversed_order() -> None:
    provider = _provider()
    messages = [
        _instruction("head", "head", key="head"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "first", "name": "first-tool", "arguments": {}},
                {"id": "second", "tool": "second-tool", "arguments": {}},
            ],
        ),
        Message(role="tool", content="second result", tool_call_id="second"),
        Message(role="tool", content="first result", tool_call_id="first"),
    ]

    provider._validate_v1_tool_sequence(messages)
    _, contents = provider._convert_messages(
        [message.model_dump() for message in messages]
    )

    responses = [
        part["function_response"]
        for content in contents
        for part in content["parts"]
        if "function_response" in part
    ]
    assert responses == [
        {"name": "second-tool", "response": {"result": "second result"}},
        {"name": "first-tool", "response": {"result": "first result"}},
    ]


def test_v1_tool_validation_rejects_duplicate_or_mismatched_pairs() -> None:
    provider = _provider()
    duplicate_calls = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "same", "name": "first", "arguments": {}},
                {"id": "same", "name": "second", "arguments": {}},
            ],
        )
    ]
    mismatched_result = [
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "one", "name": "expected", "arguments": {}}],
        ),
        Message(
            role="tool",
            content="result",
            tool_call_id="one",
            name="wrong",
        ),
    ]

    with pytest.raises(ValueError, match="duplicate tool-call IDs"):
        provider._validate_v1_tool_sequence(duplicate_calls)
    with pytest.raises(ValueError, match="does not match"):
        provider._validate_v1_tool_sequence(mismatched_result)


@pytest.mark.parametrize(
    "content,tool_calls",
    [
        (
            [
                ToolCallBlock(id="same", name="first", input={}),
                ToolCallBlock(id="same", name="second", input={}),
            ],
            None,
        ),
        (
            [ToolCallBlock(id="same", name="from-content", input={})],
            [{"id": "same", "name": "from-top-level", "arguments": {}}],
        ),
    ],
)
def test_v1_tool_validation_rejects_duplicate_content_or_conflicting_hybrid_calls(
    content: list[ToolCallBlock], tool_calls: list[dict[str, Any]] | None
) -> None:
    provider = _provider()
    messages = [
        Message(role="assistant", content=content, tool_calls=tool_calls),
        Message(
            role="tool",
            content="result",
            tool_call_id="same",
            name="first",
        ),
    ]

    with pytest.raises(ValueError, match="duplicate|conflicting"):
        provider._validate_v1_tool_sequence(messages)


def test_v1_top_level_only_tool_calls_remain_supported() -> None:
    provider = _provider()
    messages = [
        _instruction("head", "head", key="head"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "top-only", "name": "tool", "arguments": {"x": 1}}],
        ),
        Message(
            role="tool",
            content="result",
            tool_call_id="top-only",
            name="tool",
        ),
    ]

    provider._validate_v1_tool_sequence(messages)
    _, contents = provider._convert_messages(
        [message.model_dump() for message in messages]
    )
    assert contents[0]["parts"] == [
        {"function_call": {"name": "tool", "args": {"x": 1}}}
    ]


def test_v1_tool_validation_rejects_malformed_top_level_arguments() -> None:
    provider = _provider()
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "malformed", "name": "tool", "arguments": []}],
        )
    ]

    with pytest.raises(ValueError, match="malformed arguments"):
        provider._validate_v1_tool_sequence(messages)


def _sdk_response(parts: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        usage_metadata=None,
    )


@pytest.mark.asyncio
async def test_v1_hybrid_response_replays_once_with_signature_and_fixed_instruction() -> (
    None
):
    provider = _provider()
    source_response = _sdk_response(
        [
            SimpleNamespace(
                thought=False,
                function_call=SimpleNamespace(name="example", args={"x": 1}),
                thought_signature=b"\x01\x02",
            )
        ]
    )
    converted = provider._convert_to_chat_response(source_response)
    assert converted.content and converted.tool_calls
    assistant = Message(
        role="assistant",
        content=converted.content,
        tool_calls=converted.tool_calls,
    )
    request = ChatRequest(
        messages=[
            _instruction("head", "head", key="head"),
            assistant,
            Message(
                role="tool",
                content="result",
                tool_call_id=converted.tool_calls[0].id,
                name="example",
            ),
            _instruction(
                "delivered fixed",
                "tail",
                key="fixed",
                binding="fixed",
                authority="advisory",
            ),
            Message(role="user", content="next"),
        ]
    )
    original = copy.deepcopy(request.model_dump())
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(
        return_value=_sdk_response([SimpleNamespace(text="ok", thought=False)])
    )

    await provider.complete(request)

    params = provider._client.aio.models.generate_content.call_args.kwargs
    assert params["contents"][0] == {
        "role": "model",
        "parts": [
            {
                "function_call": {"name": "example", "args": {"x": 1}},
                "thought_signature": "AQI=",
            }
        ],
    }
    assert params["contents"][1]["role"] == "user"
    assert params["contents"][1]["parts"] == [
        {"function_response": {"name": "example", "response": {"result": "result"}}},
        {"text": _carrier("delivered fixed", "tail", "fixed")},
        {"text": "next"},
    ]
    assert request.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_v1_effective_model_prefers_kwargs_and_reaches_each_dispatch(
    streaming: bool,
) -> None:
    provider = GeminiProvider(
        api_key="fake",
        config={
            "default_model": "gemini-2.5-flash",
            "max_retries": 0,
            "use_streaming": streaming,
        },
    )
    request = ChatRequest(
        messages=[
            _instruction("head", "head", key="head"),
            Message(role="user", content="go"),
        ],
        model="gemini-3.5-flash",
    )
    provider._client = MagicMock()
    expected_model = "models/gemini-3.7-flash"
    if streaming:

        async def stream():
            yield _sdk_response([SimpleNamespace(text="ok", thought=False)])

        provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=stream()
        )
    else:
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=_sdk_response([SimpleNamespace(text="ok", thought=False)])
        )

    await provider.complete(request, model=expected_model)

    dispatch = (
        provider._client.aio.models.generate_content_stream
        if streaming
        else provider._client.aio.models.generate_content
    )
    assert dispatch.call_args.kwargs["model"] == expected_model


@pytest.mark.asyncio
async def test_v1_request_model_reaches_nonstreaming_dispatch_without_kwargs() -> None:
    provider = _provider()
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(
        return_value=_sdk_response([SimpleNamespace(text="ok", thought=False)])
    )
    request = ChatRequest(
        messages=[
            _instruction("head", "head", key="head"),
            Message(role="user", content="go"),
        ],
        model="gemini-3.5-flash",
    )

    await provider.complete(request)

    assert (
        provider._client.aio.models.generate_content.call_args.kwargs["model"]
        == "gemini-3.5-flash"
    )


def test_mocked_sdk_receives_head_as_system_and_inline_as_contents() -> None:
    provider = _provider()
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="ok", thought=False)]
                )
            )
        ],
        usage_metadata=None,
    )
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(return_value=response)
    request = ChatRequest(
        messages=[
            _instruction("head", "head", key="head"),
            _instruction("inline", "before_human", key="inline", authority="advisory"),
            Message(role="user", content="human"),
        ]
    )

    asyncio.run(provider.complete(request))

    params = provider._client.aio.models.generate_content.call_args.kwargs
    assert params["config"].system_instruction == "head"
    assert params["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": _carrier("inline", "before_human", "live")},
                {"text": "human"},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_streaming_v1_dispatch_does_not_mutate_canonical_messages() -> None:
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 0, "use_streaming": True},
    )
    provider._client = MagicMock()

    async def stream():
        yield SimpleNamespace(
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[])),
            ],
            usage_metadata=None,
        )

    provider._client.aio.models.generate_content_stream = AsyncMock(
        return_value=stream()
    )
    request = ChatRequest(
        messages=[
            _instruction("head", "head", key="head"),
            _instruction("inline", "before_human", key="inline", authority="advisory"),
            Message(role="user", content="human"),
        ]
    )
    original = request.model_dump()

    await provider.complete(request)

    params = provider._client.aio.models.generate_content_stream.call_args.kwargs
    assert params["config"].system_instruction == "head"
    assert params["contents"][0]["parts"][0]["text"] == _carrier(
        "inline", "before_human", "live"
    )
    assert request.model_dump() == original
