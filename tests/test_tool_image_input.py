"""No-network public-entry and real google-genai wire regressions for chat images."""

from __future__ import annotations

import base64
import json
import socket
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from google import genai
from google.genai import types

from amplifier_module_provider_gemini import GeminiProvider

PIXELS = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+jK1kAAAAASUVORK5CYII="
)
URL = "https://private.invalid/image?secret=do-not-retain"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Network is forbidden in image input fixtures")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    for name in (
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def image(*, url=False):
    return ImageBlock(
        source=(
            {"type": "url", "url": URL}
            if url
            else {"type": "base64", "media_type": "image/png", "data": PIXELS}
        )
    )


def provider(*, stream=False, model="gemini-3.7-flash", raw=False):
    return GeminiProvider(
        api_key="offline-placeholder",
        config={
            "default_model": model,
            "use_streaming": stream,
            "max_retries": 0,
            "raw": raw,
        },
    )


def tool_request(shape="direct"):
    blocks = [TextBlock(text="before"), image(), TextBlock(text="after"), image()]
    output = [block.model_dump() for block in blocks]
    result = Message(
        role="tool", tool_call_id="call-1", name="read_file", content=blocks
    )
    if shape == "user-envelope":
        result = Message(
            role="user",
            content=[
                ToolResultBlock(tool_call_id="call-1", output=output),
            ],
        )
    if shape == "tool-envelope":
        result = Message(
            role="tool",
            tool_call_id="call-1",
            name="read_file",
            content=[
                ToolResultBlock(
                    tool_call_id="call-1",
                    output={"success": False, "error": "partial", "output": output},
                ),
            ],
        )
    return ChatRequest(
        messages=[
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call-1", name="read_file", input={"path": "fixture.png"}
                    ),
                ],
            ),
            result,
        ]
    )


def wire_client(p, receive):
    client = genai.Client(
        api_key="offline-placeholder",
        http_options=types.HttpOptions(
            base_url="https://offline.invalid/",
            async_client_args={"transport": httpx.MockTransport(receive)},
        ),
    )
    p._client = p._count_route_client = client
    p._client_uses_canonical_developer_route = True
    return client


def reply(request):
    payload = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "OK"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 1,
            "totalTokenCount": 4,
        },
    }
    if "alt=sse" in str(request.url):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(payload) + "\n\n",
        )
    return httpx.Response(200, json=payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("shape", ["direct", "user-envelope", "tool-envelope"])
async def test_public_entry_sends_native_call_bound_tool_pixels(stream, shape):
    p = provider(stream=stream)
    request = tool_request(shape)
    before = request.model_dump()
    sent = []
    client = wire_client(
        p, lambda req: sent.append(json.loads(req.content)) or reply(req)
    )
    try:
        result = await p.complete(request)
        assert result.content
        assert request.model_dump() == before
        assert not p._repaired_tool_ids
        contents = sent[0]["contents"]
        assert [entry["role"] for entry in contents] == ["model", "user"]
        call = contents[0]["parts"][0]["functionCall"]
        assert call == {
            "id": "call-1",
            "name": "read_file",
            "args": {"path": "fixture.png"},
        }
        fr = contents[1]["parts"][0]["functionResponse"]
        assert (fr["id"], fr["name"]) == ("call-1", "read_file")
        assert len(fr["parts"]) == 2
        for index, part in enumerate(fr["parts"]):
            assert base64.urlsafe_b64decode(
                part["inline_data"]["data"]
            ) == base64.b64decode(PIXELS)
            assert part["inline_data"] == {
                "mime_type": "image/png",
                "data": base64.urlsafe_b64encode(base64.b64decode(PIXELS)).decode(),
                "display_name": f"tool-image-{index}",
            }
        expected = [
            {"type": "text", "text": "before", "visibility": None},
            {"image": {"$ref": "tool-image-0"}},
            {"type": "text", "text": "after", "visibility": None},
            {"image": {"$ref": "tool-image-1"}},
        ]
        projected = fr["response"]["result"]
        if shape == "tool-envelope":
            assert projected == {
                "success": False,
                "error": "partial",
                "output": expected,
            }
        else:
            assert projected == expected
        assert PIXELS not in json.dumps(fr["response"])
        # Count uses the same typed image-bearing plan, not a base64 text estimate.
        count = p._count_request_payload(p._build_request_plan(request))[
            "generateContentRequest"
        ]
        assert count.pop("model") == "models/gemini-3.7-flash"
        assert count == sent[0]
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(
    "shape,output",
    [
        ("direct", ""),
        ("direct", []),
        *[
            (shape, output)
            for shape in ("user-envelope", "tool-envelope")
            for output in (False, 0, None, "", [], {})
        ],
    ],
)
async def test_same_name_parallel_status_results_have_two_sided_ids(
    stream, shape, flag, output
):
    """Count/send equality alone cannot detect a one-sided call ID."""
    p = provider(stream=stream)
    request = tool_request()
    image_call = request.messages[0].content[0]
    image_signature = base64.b64encode(b"image-call-signature").decode()
    plain_signature = base64.b64encode(b"plain-call-signature").decode()
    native = {"version": 1, "provider": "gemini"}
    request.messages[0].content = [
        image_call.model_copy(
            update={
                "signature": image_signature,
                "thinking_provenance": native,
            }
        ),
        ToolCallBlock(
            id="plain-call",
            name="read_file",
            input={"path": "plain.txt"},
            signature=plain_signature,
            thinking_provenance=native,
        ),
    ]
    # Return the second same-name call FIRST. Name/order matching is ambiguous.
    if shape == "direct":
        plain = Message(
            role="tool",
            tool_call_id="plain-call",
            name="read_file",
            content=output,
            is_error=flag,
        )
    else:
        plain = Message(
            role="user" if shape == "user-envelope" else "tool",
            tool_call_id=None if shape == "user-envelope" else "plain-call",
            name=None if shape == "user-envelope" else "read_file",
            content=[
                ToolResultBlock(
                    tool_call_id="plain-call",
                    output=output,
                    is_error=flag,
                )
            ],
        )
    request.messages.insert(1, plain)
    before = request.model_dump()
    sent = []
    client = wire_client(
        p, lambda req: sent.append(json.loads(req.content)) or reply(req)
    )
    try:
        await p.complete(request)
        assert request.model_dump() == before
        assert not p._repaired_tool_ids
        parts = sent[0]["contents"][0]["parts"]
        calls = {part["functionCall"]["id"]: part for part in parts}
        assert set(calls) == {"call-1", "plain-call"}
        assert calls["call-1"]["functionCall"] == {
            "id": "call-1",
            "name": "read_file",
            "args": {"path": "fixture.png"},
        }
        assert calls["plain-call"]["functionCall"] == {
            "id": "plain-call",
            "name": "read_file",
            "args": {"path": "plain.txt"},
        }
        assert calls["call-1"]["thoughtSignature"] == image_signature
        assert calls["plain-call"]["thoughtSignature"] == plain_signature
        results = [
            content["parts"][0]["functionResponse"]
            for content in sent[0]["contents"][1:]
        ]
        assert [result["id"] for result in results] == ["plain-call", "call-1"]
        for result in results:
            assert result["id"] in calls
            assert result["name"] == calls[result["id"]]["functionCall"]["name"]
        assert results[0]["response"] == {"error" if flag else "output": output}
        assert "parts" not in results[0]
        assert len(results[1]["parts"]) == 2
        for media in results[1]["parts"]:
            assert base64.urlsafe_b64decode(
                media["inline_data"]["data"]
            ) == base64.b64decode(PIXELS)
        count = p._count_request_payload(p._build_request_plan(request))[
            "generateContentRequest"
        ]
        assert count.pop("model") == "models/gemini-3.7-flash"
        assert sent == [count]
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
async def test_existing_native_counter_counts_the_same_tool_pixels(monkeypatch):
    p = provider()
    request = tool_request("user-envelope")
    before = deepcopy(request.model_dump())
    sent = []
    real_client = httpx.AsyncClient

    def count_client(*, timeout):
        return real_client(
            timeout=timeout,
            transport=httpx.MockTransport(
                lambda req: sent.append(json.loads(req.content))
                or httpx.Response(200, json={"totalTokens": 123}),
            ),
        )

    monkeypatch.setattr(httpx, "AsyncClient", count_client)
    budget = await p.request_budget(request, context_estimate=1000)
    assert budget["estimated_input_tokens"] == 123
    fr = sent[0]["generateContentRequest"]["contents"][1]["parts"][0][
        "functionResponse"
    ]
    assert base64.urlsafe_b64decode(
        fr["parts"][0]["inline_data"]["data"]
    ) == base64.b64decode(PIXELS)
    assert request.model_dump() == before
    assert not p._repaired_tool_ids


@pytest.mark.parametrize("position", ["assistant", "system", "developer"])
def test_unqualified_history_marks_original_slot_without_poisoning_followup(position):
    p = provider()
    request = ChatRequest(
        messages=[
            Message(
                role=position,
                content=[TextBlock(text="before"), image(), TextBlock(text="after")],
            ),
            Message(role="user", content="plain followup"),
        ]
    )
    before = request.model_dump()
    plan = p._build_request_plan(request)
    serialized = json.dumps(p._count_request_payload(plan))
    assert "Image unavailable" in serialized
    assert PIXELS not in serialized
    assert plan.contents[-1] == {"role": "user", "parts": [{"text": "plain followup"}]}
    assert request.model_dump() == before


@pytest.mark.parametrize(
    "model", ["gemini-2.5-flash", "unknown-alias", "gemini-3.7-flash-unqualified"]
)
def test_unqualified_tool_model_keeps_call_bound_marker_and_plain_followup(model):
    p = provider(model=model)
    request = tool_request()
    request.messages.append(Message(role="user", content="plain followup"))
    plan = p._build_request_plan(request)
    fr = plan.contents[1]["parts"][0]["function_response"]
    assert fr["name"] == "read_file"
    assert fr["id"] == "call-1"
    assert plan.contents[0]["parts"][0]["function_call"]["id"] == "call-1"
    assert "parts" not in fr
    assert "Image unavailable" in json.dumps(fr["response"])
    assert PIXELS not in json.dumps(plan.contents)
    assert plan.contents[-1]["parts"] == [{"text": "plain followup"}]


@pytest.mark.parametrize("role", ["user", "tool"])
def test_url_is_explicit_original_slot_unavailable_not_silently_dropped(role):
    p = provider()
    request = tool_request() if role == "tool" else ChatRequest(messages=[])
    if role == "tool":
        request.messages[-1].content = [
            TextBlock(text="before"),
            image(url=True),
            TextBlock(text="after"),
        ]
    else:
        request.messages.append(Message(role="user", content=[image(url=True)]))
    plan = p._build_request_plan(request)
    serialized = json.dumps(plan.contents)
    assert "Supply inline base64" in serialized
    assert URL not in serialized
    assert (
        "function_response" in serialized if role == "tool" else len(plan.contents) == 1
    )


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "image/png", "data": "%%%"},
        {"type": "base64", "media_type": "image/png", "data": ""},
    ],
)
def test_malformed_image_error_does_not_echo_source(source):
    p = provider()
    with pytest.raises(ValueError) as caught:
        p._build_request_plan(
            ChatRequest(
                messages=[
                    Message(role="user", content=[ImageBlock(source=source)]),
                ]
            )
        )
    assert (
        source["data"] not in str(caught.value) if source["data"] else str(caught.value)
    )


def test_parallel_result_identity_and_block_order_are_not_first_call_wins():
    p = provider()
    request = tool_request()
    request.messages[0].content.insert(
        0, ToolCallBlock(id="other", name="other", input={})
    )
    request.messages.append(
        Message(role="tool", tool_call_id="other", name="other", content="text result")
    )
    plan = p._build_request_plan(request)
    assert plan.contents[0]["parts"][1]["function_call"]["id"] == "call-1"
    assert plan.contents[1]["parts"][0]["function_response"]["name"] == "read_file"
    assert plan.contents[2]["parts"][0]["function_response"]["name"] == "other"


@pytest.mark.parametrize("route", ["vertex", "custom", "injected"])
def test_unqualified_route_has_explicit_tool_image_marker(monkeypatch, route):
    p = provider()
    if route == "vertex":
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    elif route == "custom":
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://custom.invalid/")
    else:
        p._client = SimpleNamespace()
    plan = p._build_request_plan(tool_request())
    assert "Image unavailable" in json.dumps(plan.contents)
    assert PIXELS not in json.dumps(plan.contents)


class Hooks:
    def __init__(self):
        self.events = []

    async def emit(self, name, payload):
        # This is the FIRST subscriber, before any downstream redactor.
        serialized = json.dumps(payload, default=str)
        assert PIXELS not in serialized
        assert (
            base64.urlsafe_b64encode(base64.b64decode(PIXELS)).decode()
            not in serialized
        )
        assert URL not in serialized
        self.events.append((name, deepcopy(payload)))


@pytest.mark.asyncio
async def test_raw_diagnostics_omit_media_before_first_subscriber():
    p = provider(raw=True)
    hooks = Hooks()
    p.coordinator = SimpleNamespace(hooks=hooks)
    request = tool_request()
    request.messages.insert(0, Message(role="user", content=[image(url=True), image()]))
    sent = []
    client = wire_client(
        p, lambda req: sent.append(json.loads(req.content)) or reply(req)
    )
    try:
        await p.complete(request)
        raw = next(
            payload["raw"] for name, payload in hooks.events if name == "llm:request"
        )
        assert "diagnostic-image-omitted" in json.dumps(raw)
        assert base64.urlsafe_b64encode(
            base64.b64decode(PIXELS)
        ).decode() in json.dumps(sent)
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_media_error_echo_is_not_published_and_does_not_retry(stream, caplog):
    p = provider(stream=stream, raw=True)
    hooks = Hooks()
    p.coordinator = SimpleNamespace(hooks=hooks)
    from google.genai.errors import ClientError

    error = ClientError(400, {"error": {"message": f"invalid {PIXELS} {URL}"}})
    method = AsyncMock(side_effect=error)
    p._client = p._count_route_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=method,
                generate_content_stream=method,
            )
        )
    )
    p._client_uses_canonical_developer_route = True
    with pytest.raises(Exception) as caught:
        await p.complete(tool_request())
    assert PIXELS not in str(caught.value)
    assert URL not in str(caught.value)
    assert method.await_count == 1
    assert PIXELS not in caplog.text
    assert URL not in caplog.text
    if not stream:
        assert any(
            name == "llm:response" and payload["status"] == "error"
            for name, payload in hooks.events
        )


@pytest.mark.parametrize(
    "output",
    [
        None,
        False,
        0,
        "",
        {},
        [],
        {"type": "image", "name": "ordinary business value"},
        {"thumbnail": {"type": "image", "source": {"type": "base64", "data": PIXELS}}},
        {"success": False, "output": False},
        {"success": False, "output": {"type": "image", "name": "business value"}},
    ],
)
def test_ordinary_business_and_falsey_tool_output_is_exact(output):
    request = tool_request()
    request.messages[-1] = Message(
        role="user",
        content=[
            ToolResultBlock(tool_call_id="call-1", output=output),
        ],
    )
    plan = provider()._build_request_plan(request)
    fr = plan.contents[1]["parts"][0]["function_response"]
    assert fr == {"name": "read_file", "response": {"result": output}}


@pytest.mark.parametrize(
    "mime", ["image/png", "image/jpeg", "image/webp", "image/heic"]
)
def test_tool_mime_qualification_is_narrower_than_user_images(mime):
    request = tool_request()
    request.messages[-1].content = [
        ImageBlock(source={"type": "base64", "media_type": mime, "data": PIXELS}),
    ]
    fr = (
        provider()
        ._build_request_plan(request)
        .contents[1]["parts"][0]["function_response"]
    )
    if mime == "image/heic":
        assert "Image unavailable" in json.dumps(fr)
        assert "parts" not in fr
    else:
        assert fr["parts"][0]["inline_data"]["data"] == base64.b64decode(PIXELS)
        assert fr["parts"][0]["inline_data"]["mime_type"] == mime


@pytest.mark.parametrize(
    "call_id,name",
    [
        ("missing-call", "read_file"),
        ("call-1", "different_function"),
        (None, "read_file"),
    ],
)
def test_mismatched_tool_images_fail_without_payload_echo(call_id, name):
    request = tool_request()
    request.messages[-1].tool_call_id = call_id
    request.messages[-1].name = name
    with pytest.raises(ValueError, match="matching") as caught:
        provider()._build_request_plan(request)
    assert PIXELS not in str(caught.value)


@pytest.mark.asyncio
async def test_catalog_discovery_does_not_select_reroute_or_infer_new_vision_models():
    p = provider(model="gemini-3.7-flash")
    p._list_models_from_api = AsyncMock(
        return_value=[
            {"id": "gemini-2.5-flash", "generation_methods": ["generateContent"]},
            {"id": "gemini-3.8-flash", "generation_methods": ["generateContent"]},
            {"id": "gemini-3.99-flash", "generation_methods": ["generateContent"]},
        ]
    )
    catalog = await p.list_models(filters={})
    assert [item.id for item in catalog] == [
        "gemini-2.5-flash",
        "gemini-3.8-flash",
        "gemini-3.99-flash",
    ]
    assert p.default_model == "gemini-3.7-flash"
    assert p._client is None
    for model in [item.id for item in catalog]:
        plan = p._build_request_plan(tool_request(), model=model)
        fr = plan.contents[1]["parts"][0]["function_response"]
        assert ("parts" in fr) is (model == "gemini-3.8-flash")
        assert plan.model == model


def test_nested_known_output_envelope_retains_status_and_native_media():
    request = tool_request()
    request.messages[-1].content = [
        ToolResultBlock(
            tool_call_id="call-1",
            output={
                "success": False,
                "output": {
                    "success": True,
                    "output": [
                        TextBlock(text="caption").model_dump(),
                        image().model_dump(),
                    ],
                },
            },
        )
    ]
    fr = (
        provider()
        ._build_request_plan(request)
        .contents[1]["parts"][0]["function_response"]
    )
    assert fr["response"]["result"] == {
        "success": False,
        "output": {
            "success": True,
            "output": [
                TextBlock(text="caption").model_dump(),
                {"image": {"$ref": "tool-image-0"}},
            ],
        },
    }
    assert fr["parts"][0]["inline_data"]["data"] == base64.b64decode(PIXELS)


def test_native_signature_replay_remains_byte_exact_with_tool_images():
    request = tool_request()
    signature = base64.b64encode(b"native-signature").decode()
    request.messages[0].content[0] = (
        request.messages[0]
        .content[0]
        .model_copy(
            update={
                "signature": signature,
                "thinking_provenance": {"version": 1, "provider": "gemini"},
            }
        )
    )
    p = provider()
    count = p._count_request_payload(p._build_request_plan(request))[
        "generateContentRequest"
    ]
    call_part = count["contents"][0]["parts"][0]
    assert call_part["thoughtSignature"] == signature
    assert call_part["functionCall"]["id"] == "call-1"


@pytest.mark.asyncio
async def test_raw_known_media_echoes_omitted_but_sdk_and_response_text_unchanged():
    p = provider(raw=True)
    hooks = Hooks()
    p.coordinator = SimpleNamespace(hooks=hooks)
    echo = f"original sources {URL} {PIXELS}"
    request = tool_request()
    request.messages.insert(
        0,
        Message(
            role="user",
            content=[
                TextBlock(text=echo),
                image(url=True),
                image(),
                TextBlock(text="ordinary https://public.invalid/keep-this-text"),
            ],
        ),
    )
    before = request.model_dump()
    sent = []

    def receive(req):
        sent.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": echo},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
            },
        )

    client = wire_client(p, receive)
    try:
        response = await p.complete(request)
        assert response.content[0].text == echo
        assert request.model_dump() == before
        assert sent[0]["contents"][0]["parts"][0] == {"text": echo}
        count = p._count_request_payload(p._build_request_plan(request))[
            "generateContentRequest"
        ]
        count.pop("model")
        # Ordinary user inline MIME spelling is an established SDK difference.
        sent[0]["contents"][0]["parts"][2]["inlineData"]["mimeType"] = sent[0][
            "contents"
        ][0]["parts"][2]["inlineData"].pop("mime_type")
        assert sent[0] == count
        request_raw = next(
            payload["raw"] for name, payload in hooks.events if name == "llm:request"
        )
        response_raw = next(
            payload["raw"] for name, payload in hooks.events if name == "llm:response"
        )
        assert "diagnostic-image-echo-omitted" in json.dumps(request_raw)
        assert "diagnostic-image-echo-omitted" in json.dumps(response_raw)
        assert "https://public.invalid/keep-this-text" in json.dumps(request_raw)
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("shape", ["direct", "user-envelope", "tool-envelope"])
async def test_explicit_tool_error_status_keeps_pair_pixels_and_count_send(flag, shape):
    p = provider()
    request = tool_request(shape)
    if shape == "direct":
        request.messages[-1] = request.messages[-1].model_copy(
            update={"is_error": flag}
        )
    else:
        request.messages[-1].content[0] = (
            request.messages[-1]
            .content[0]
            .model_copy(
                update={"is_error": flag},
            )
        )
    request.messages[0].content.append(
        ToolCallBlock(id="call-2", name="other", input={})
    )
    request.messages.append(
        Message(
            role="tool",
            tool_call_id="call-2",
            name="other",
            content="",
            is_error=not flag,
        )
    )
    before = request.model_dump()
    sent = []
    client = wire_client(
        p, lambda req: sent.append(json.loads(req.content)) or reply(req)
    )
    try:
        await p.complete(request)
        assert request.model_dump() == before
        first = sent[0]["contents"][1]["parts"][0]["functionResponse"]
        second = sent[0]["contents"][2]["parts"][0]["functionResponse"]
        assert (first["id"], first["name"]) == ("call-1", "read_file")
        assert set(first["response"]) == {"error" if flag else "output"}
        assert len(first["parts"]) == 2
        assert (second["id"], second["name"]) == ("call-2", "other")
        assert second["response"] == {"output" if flag else "error": ""}
        count = p._count_request_payload(p._build_request_plan(request))[
            "generateContentRequest"
        ]
        count.pop("model")
        assert sent == [count]
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("output", [False, 0, None, "", [], {}])
def test_explicit_tool_status_preserves_falsey_envelope_output(flag, output):
    request = tool_request()
    request.messages[-1] = Message(
        role="user",
        content=[
            ToolResultBlock(tool_call_id="call-1", output=output, is_error=flag),
        ],
    )
    fr = (
        provider()
        ._build_request_plan(request)
        .contents[1]["parts"][0]["function_response"]
    )
    assert fr == {
        "id": "call-1",
        "name": "read_file",
        "response": {"error" if flag else "output": output},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize(
    "source",
    [
        {"type": "file", "file_id": "old-file"},
        {"type": "file_id", "file_id": "old-id"},
        {"type": "base64", "media_type": "text/plain", "data": PIXELS},
    ],
)
async def test_unsupported_historical_image_allows_plain_followup_sdk_dispatch(
    role, source
):
    p = provider()
    request = (
        tool_request()
        if role == "tool"
        else ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="before"),
                        ImageBlock(source=source),
                        TextBlock(text="after"),
                    ],
                ),
            ]
        )
    )
    if role == "tool":
        request.messages[-1].content = [
            TextBlock(text="before"),
            ImageBlock(source=source),
            TextBlock(text="after"),
        ]
    request.messages.append(Message(role="assistant", content="historical response"))
    request.messages.append(Message(role="user", content="plain followup"))
    before = request.model_dump()
    sent = []
    client = wire_client(
        p, lambda req: sent.append(json.loads(req.content)) or reply(req)
    )
    try:
        response = await p.complete(request)
        assert response.content[0].text == "OK"
        assert request.model_dump() == before
        assert sent[0]["contents"][-1] == {
            "role": "user",
            "parts": [{"text": "plain followup"}],
        }
        slot = sent[0]["contents"][1 if role == "tool" else 0]["parts"]
        if role == "tool":
            slot = slot[0]["functionResponse"]["response"]["result"]
        assert slot[0]["text"] == "before"
        assert "Image unavailable" in slot[1]["text"]
        assert slot[2]["text"] == "after"
        assert "old-file" not in json.dumps(sent)
        assert PIXELS not in json.dumps(sent)
    finally:
        await client.aio.aclose()
        client.close()
