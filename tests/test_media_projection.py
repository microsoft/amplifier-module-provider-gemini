"""Offline media-v1 exact-wire, pretransport and privacy conformance."""

import base64
import asyncio
from copy import deepcopy
import hashlib
import io
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image
from google import genai
from google.genai import types
from amplifier_core.message_models import ChatRequest, ImageBlock, Message, TextBlock
from amplifier_core.llm_errors import InvalidRequestError, ProviderUnavailableError
from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini._media import NAMESPACE, diagnostic_copy


PIXELS = b"offline-synthetic-original-image-bytes"
# Literal catalogue snapshot, intentionally independent of production sets:
# adding a declaration without adding actual SDK wire coverage must fail.
CATALOGUE_USER_IDS = (
    "gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen",
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
    "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.5-flash-lite", "gemini-robotics-er-2-preview",
)
CATALOGUE_TOOL_IDS = (
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
    "gemini-3-flash-preview", "gemini-4-hydrogen",
)


def image(data=PIXELS):
    return ImageBlock(source={
        "type": "base64", "media_type": "image/png",
        "data": base64.b64encode(data).decode(),
    })


def request():
    return ChatRequest(messages=[
        Message(role="user", content=[TextBlock(text="before"), image(), TextBlock(text="after")],
                metadata={NAMESPACE: {"refs": []}, "owner": {"keep": True}}),
        Message(role="assistant", content="", tool_calls=[
            {"id": "a", "name": "first", "arguments": {}},
            {"id": "b", "name": "second", "arguments": {}},
        ]),
        Message(role="tool", tool_call_id="a", content=[TextBlock(text="result-a"), image(b"one"), image(b"two")],
                metadata={NAMESPACE: {"refs": []}}),
        Message(role="tool", tool_call_id="b", content="result-b"),
    ], max_output_tokens=99)


class Capability:
    version = 1

    def __init__(self):
        self.calls = []
        self.valid = True

    def validate(self, value, *, purpose, request_options):
        self.calls.append((value, purpose, deepcopy(request_options)))
        if not self.valid:
            raise ValueError("expired/revoked: no transport")
        return {"valid": True}


def provider(cap=None, streaming=False):
    coordinator = SimpleNamespace(
        get_capability=lambda key: cap,
        hooks=SimpleNamespace(emit=AsyncMock()),
    )
    return GeminiProvider(api_key="offline", coordinator=coordinator, config={
        "default_model": "gemini-3.1-pro-preview", "use_streaming": streaming,
        "max_retries": 2, "min_retry_delay": 0.001, "raw": True,
    })


def response():
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}]}


def bind(p, receive):
    client = genai.Client(api_key="offline", http_options=types.HttpOptions(
        base_url="https://offline.invalid",
        async_client_args={"transport": httpx.MockTransport(receive)},
    ))
    p._client = client
    return client


async def close(client):
    await client.aio.aclose()
    client.close()


def normalize(body):
    # genai2 emits snake MIME names at this exact known vendor location.
    for content in body.get("contents", []):
        for part in content["parts"]:
            blobs = [part.get("inlineData")]
            blobs += [p.get("inlineData") for p in part.get("functionResponse", {}).get("parts", [])]
            for blob in blobs:
                if blob and "mime_type" in blob:
                    blob["mimeType"] = blob.pop("mime_type")
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_actual_sdk_wire_count_identity_sha_order_and_preemit_privacy(streaming, record_property):
    cap = Capability()
    p = provider(cap, streaming)
    sent = []

    def receive(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=response())

    client = bind(p, receive)
    req = request()
    before = req.model_dump()
    try:
        plan = p._request_plan(req, model="gemini-3.1-pro-preview")
        count = p._count_request_payload(plan)["generateContentRequest"]
        count.pop("model")
        assert p._request_plan(req, model="gemini-3.1-pro-preview") is plan
        await p._validate_media_attempt(plan, purpose="count")
        await p.complete(req, model="gemini-3.1-pro-preview")
        assert normalize(sent[0]) == count
        parts = sent[0]["contents"][0]["parts"]
        assert [parts[0]["text"], parts[2]["text"]] == ["before", "after"]
        assert hashlib.sha256(base64.b64decode(parts[1]["inlineData"]["data"])).hexdigest() == hashlib.sha256(PIXELS).hexdigest()
        tool = sent[0]["contents"][2]["parts"][0]["functionResponse"]
        assert tool["name"] == "first"
        assert [base64.b64decode(p["inline_data"]["data"]) for p in tool["parts"]] == [b"one", b"two"]
        assert sent[0]["contents"][3]["parts"][0]["functionResponse"]["name"] == "second"
        assert req.model_dump() == before
        assert all(call[0] is req for call in cap.calls)
        assert {call[1] for call in cap.calls} == {"count", "transport"}
        events = [call.args for call in p.coordinator.hooks.emit.call_args_list]
        encoded = json.dumps(events)
        assert base64.b64encode(PIXELS).decode() not in encoded
        assert "not-retained" in encoded
        assert not p._media_plans
        record_property("gemini.media.wire", json.dumps({
            "transport": "stream" if streaming else "complete",
            "sdk": "google-genai2.30.1",
            "count_send_identity": True,
            "model": "gemini-3.1-pro-preview",
            "sanitized_actual_wire": diagnostic_copy(sent[0]),
            "decoded_image_order": [
                {"position": "user", "sha256": hashlib.sha256(PIXELS).hexdigest()},
                {"position": "tool:first:0", "sha256": hashlib.sha256(b"one").hexdigest()},
                {"position": "tool:first:1", "sha256": hashlib.sha256(b"two").hexdigest()},
            ],
        }))
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CATALOGUE_USER_IDS)
@pytest.mark.parametrize("image_format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
async def test_every_advertised_user_model_valid_image_actual_sdk_wire(
    model, image_format, mime, streaming, record_property, monkeypatch,
):
    """Literal model/MIME matrix is wire proof, not a capability self-assertion."""
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (19, 83, 147)).save(output, format=image_format)
    pixels = output.getvalue()
    with Image.open(io.BytesIO(pixels)) as decoded:
        assert decoded.format == image_format
        assert decoded.size == (3, 2)
        decoded.verify()
    cap = Capability()
    p = provider(cap, streaming)
    sent = []

    def receive(req):
        sent.append((str(req.url), json.loads(req.content)))
        if streaming:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps(response()) + "\n\n",
            )
        return httpx.Response(200, json=response())

    client = bind(p, receive)
    p._count_route_client = client
    p._client_uses_canonical_developer_route = True  # Mock route only.
    req = ChatRequest(messages=[
        Message(role="user", content=[
            TextBlock(text="before-user-image"),
            ImageBlock(source={"type": "base64", "media_type": mime,
                               "data": base64.b64encode(pixels).decode("ascii")}),
            TextBlock(text="after-user-image"),
        ], metadata={NAMESPACE: {"refs": []}, "owner": {"keep": "unchanged"}}),
    ], max_output_tokens=73)
    before = req.model_dump()
    options = {"model": model, "temperature": 0.25, "tool_choice": "none"}
    try:
        plan = p._request_plan(req, **options)
        count_body = p._count_request_payload(plan)["generateContentRequest"]
        assert count_body.pop("model") == f"models/{model}"
        assert p._request_plan(req, **options) is plan
        await p._validate_media_attempt(plan, purpose="count")
        real_client = httpx.AsyncClient
        counted = []

        def count_client(*, timeout):
            def count(http_request):
                counted.append(json.loads(http_request.content))
                return httpx.Response(200, json={"totalTokens": 42})
            return real_client(timeout=timeout, transport=httpx.MockTransport(count))

        with monkeypatch.context() as patch:
            patch.setattr(httpx, "AsyncClient", count_client)
            budget = await p.request_budget(req, context_estimate=100, request_options=options)
        if model in ("gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen", "gemini-robotics-er-2-preview"):
            # New unified/specialized window semantics are not covered by the
            # existing native budget counter; do not pretend a wire plan is a
            # successful HTTP count or silently borrow another model's limits.
            assert budget is None and not counted
            native_count = "unavailable"
        else:
            assert budget["measurement"]["input_tokens"] == 42
            counted_body = counted[0]["generateContentRequest"]
            assert counted_body.pop("model") == f"models/{model}"
            assert counted_body == count_body
            native_count = "mock-http-count"
        await p.complete(req, request_options=options)
        assert len(sent) == 1
        url, actual = sent[0]
        method = "streamGenerateContent" if streaming else "generateContent"
        assert f"/models/{model}:{method}" in url
        assert normalize(deepcopy(actual)) == count_body
        parts = actual["contents"][0]["parts"]
        assert len(parts) == 3
        assert parts[0] == {"text": "before-user-image"}
        assert parts[2] == {"text": "after-user-image"}
        blob = parts[1]["inlineData"]
        assert blob.get("mime_type", blob.get("mimeType")) == mime
        # Locked google-genai2.30.1 encodes bytes with urlsafe_b64encode.
        assert base64.b64decode(blob["data"], altchars=b"-_", validate=True) == pixels
        assert count_body["generationConfig"]["temperature"] == 0.25
        assert count_body["generationConfig"]["maxOutputTokens"] == 73
        assert count_body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
        assert all(value is req and seen_options == options
                   for value, _, seen_options in cap.calls)
        assert {purpose for _, purpose, _ in cap.calls} == {"count", "transport"}
        assert req.model_dump() == before
        events = json.dumps([call.args for call in p.coordinator.hooks.emit.call_args_list])
        assert base64.b64encode(pixels).decode("ascii") not in events
        assert base64.urlsafe_b64encode(pixels).decode("ascii") not in events
        assert "not-retained" in events
        if streaming:
            assert any(call.args[0] == "llm:stream_block_delta"
                       for call in p.coordinator.hooks.emit.call_args_list)
        assert not p._media_plans
        record_property("gemini.media.advertised-user-wire", json.dumps({
            "model": model, "mime": mime, "format": image_format,
            "transport": method, "count_send_options_identity": True,
            "native_budget_measurement": native_count,
            "decoded_sha256": hashlib.sha256(pixels).hexdigest(),
            "sanitized_actual_wire": diagnostic_copy(actual),
        }))
    finally:
        await close(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CATALOGUE_TOOL_IDS)
@pytest.mark.parametrize("image_format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
async def test_catalogued_tool_models_valid_pixels_original_call_wire(
    model, image_format, mime, streaming, record_property,
):
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (19, 83, 147)).save(output, format=image_format)
    pixels = output.getvalue()
    cap = Capability()
    p = provider(cap, streaming)
    sent = []

    def receive(req):
        sent.append((str(req.url), json.loads(req.content)))
        if streaming:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps(response()) + "\n\n")
        return httpx.Response(200, json=response())

    client = bind(p, receive)
    req = ChatRequest(messages=[
        Message(role="user", content="look at the tool result"),
        Message(role="assistant", content="", tool_calls=[
            {"id": "call-1", "name": "read_file", "arguments": {}},
        ]),
        Message(role="tool", tool_call_id="call-1", name="read_file", content=[
            TextBlock(text="before-tool"),
            ImageBlock(source={"type": "base64", "media_type": mime,
                               "data": base64.b64encode(pixels).decode("ascii")}),
            TextBlock(text="after-tool"),
        ], metadata={NAMESPACE: {"refs": []}, "owner": {"retained": True}}),
        Message(role="user", content="unchanged-tail"),
    ], max_output_tokens=73)
    before = req.model_dump()
    options = {"model": model, "temperature": 0.25, "tool_choice": "none"}
    try:
        plan = p._request_plan(req, **options)
        count_body = p._count_request_payload(plan)["generateContentRequest"]
        assert count_body.pop("model") == f"models/{model}"
        await p._validate_media_attempt(plan, purpose="count")
        await p.complete(req, request_options=options)
        assert len(sent) == 1
        url, actual = sent[0]
        assert f"/models/{model}:" in url
        assert normalize(deepcopy(actual)) == count_body
        assert len(actual["contents"]) == 4
        tool = actual["contents"][2]["parts"][0]["functionResponse"]
        assert tool["name"] == "read_file"
        assert tool["response"]["result"] == [
            {"text": "before-tool"}, {"image": {"$ref": "session-image-0"}},
            {"text": "after-tool"},
        ]
        blob = tool["parts"][0]["inline_data"]
        assert blob["mime_type"] == mime
        assert base64.b64decode(blob["data"], altchars=b"-_", validate=True) == pixels
        assert actual["contents"][3]["parts"] == [{"text": "unchanged-tail"}]
        assert req.model_dump() == before
        assert all(value is req and seen_options == options for value, _, seen_options in cap.calls)
        events = json.dumps([call.args for call in p.coordinator.hooks.emit.call_args_list])
        assert base64.b64encode(pixels).decode("ascii") not in events
        assert base64.urlsafe_b64encode(pixels).decode("ascii") not in events
        assert "not-retained" in events
        record_property("gemini.media.catalogued-tool-wire", json.dumps({
            "model": model, "mime": mime, "transport": "stream" if streaming else "complete",
            "decoded_sha256": hashlib.sha256(pixels).hexdigest(),
            "count_projection_send_identity": True, "sanitized_actual_wire": diagnostic_copy(actual),
        }))
    finally:
        await close(client)

@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["expired", "revoked"])
async def test_retry_revalidates_before_second_physical_send(reason):
    cap = Capability()
    p = provider(cap)
    sent = []

    def receive(req):
        sent.append(req)
        cap.valid = False
        return httpx.Response(503, json={"error": {"code": 503, "message": reason}})

    client = bind(p, receive)
    try:
        with pytest.raises(InvalidRequestError, match="details not retained"):
            await p.complete(request())
        assert len(sent) == 1
        assert len(cap.calls) >= 2
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_unknown_send_is_not_retried_and_safe():
    p = provider(Capability())
    sent = []

    def receive(req):
        sent.append(req)
        raise httpx.ReadError(base64.b64encode(PIXELS).decode())

    client = bind(p, receive)
    try:
        with pytest.raises(ProviderUnavailableError) as caught:
            await p.complete(request())
        assert not caught.value.retryable
        assert len(sent) == 1
        assert base64.b64encode(PIXELS).decode() not in str(caught.value)
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_first_stream_chunk_then_disconnect_never_replays():
    cap = Capability()
    p = provider(cap, streaming=True)
    sends = []

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}\n\n'
            raise httpx.ReadError(base64.b64encode(PIXELS).decode())

    def receive(req):
        sends.append(req)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Chunks())

    client = bind(p, receive)
    try:
        with pytest.raises(ProviderUnavailableError) as caught:
            await p.complete(request())
        assert not caught.value.retryable
        assert len(sends) == 1
        assert any(call.args[0] == "llm:stream_block_delta" for call in p.coordinator.hooks.emit.call_args_list)
        assert base64.b64encode(PIXELS).decode() not in repr(p.coordinator.hooks.emit.call_args_list)
        assert not p._media_plans
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_cancellation_releases_private_plan_and_no_retry():
    p = provider(Capability())
    started = asyncio.Event()
    sends = []

    async def receive(req):
        sends.append(req)
        started.set()
        await asyncio.Future()

    client = bind(p, receive)
    task = asyncio.create_task(p.complete(request()))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(sends) == 1
        assert not p._media_plans
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_present_capability_missing_witness_refuses_text_too():
    class Missing(Capability):
        def validate(self, *args, **kwargs):
            raise ValueError("unknown association")

    p = provider(Missing())
    with pytest.raises(InvalidRequestError, match="refused preparation"):
        await p.complete(ChatRequest(messages=[Message(role="user", content="text")]))
    assert p._client is None


@pytest.mark.asyncio
async def test_close_releases_counted_vendor_plan_without_dispatch():
    p = provider(Capability())
    req = request()
    p._request_plan(req)
    assert p._media_plans
    await p.close()
    assert not p._media_plans


@pytest.mark.asyncio
async def test_legacy_retry_diagnostic_omits_echoed_image_payload():
    p = provider()
    sentinel = base64.b64encode(PIXELS).decode()
    sent = []

    def receive(req):
        sent.append(req)
        return httpx.Response(503, json={"error": {"code": 503, "message": sentinel}})

    client = bind(p, receive)
    req = ChatRequest(messages=[Message(role="user", content=[image()])])
    try:
        with pytest.raises(Exception) as caught:
            await p.complete(req)
        assert len(sent) >= 1
        assert sentinel not in repr(p.coordinator.hooks.emit.call_args_list)
        assert sentinel not in str(caught.value)
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_counted_options_change_requires_new_candidate_and_finalization_none():
    p = provider(Capability())
    req = request()
    req.tool_choice = "none"
    plan = p._request_plan(req)
    assert plan.config.tool_config.function_calling_config.mode == "NONE"
    assert p._request_plan(req, model=p.default_model) is plan
    with pytest.raises(InvalidRequestError, match="options changed"):
        p._request_plan(req, temperature=0.1)
    await p.close()


@pytest.mark.asyncio
async def test_request_hook_cannot_revoke_or_mutate_body_between_witness_and_send():
    cap = Capability()
    p = provider(cap)
    sent = []
    client = bind(p, lambda req: sent.append(req) or httpx.Response(200, json=response()))

    async def revoke(req):
        cap.valid = False

    client._api_client._async_httpx_client.event_hooks["request"].append(revoke)
    try:
        with pytest.raises(ProviderUnavailableError):
            await p.complete(request())
        assert not sent
        assert cap.valid
    finally:
        await close(client)


@pytest.mark.asyncio
async def test_mutation_after_count_refuses_without_refit_or_transport():
    p = provider(Capability())
    req = request()
    p._request_plan(req)
    req.messages[0].metadata["owner"]["keep"] = False
    with pytest.raises(InvalidRequestError, match="changed"):
        await p.complete(req)
    assert not p._media_plans
    await close(p._client)


def test_exact_support_unknown_models_positions_and_diagnostic_copy():
    p = provider()
    assert p.get_media_request_support(model="gemini-3.1-pro-preview")["positions"] == ["user", "tool"]
    assert p.get_media_request_support(model="gemini-2.5-flash")["positions"] == ["user"]
    for model in ("gemini-unknown", "gemini-3.1-pro-preview-not-a-model"):
        assert p.get_media_request_support(model=model)["positions"] == []
        contents = p._build_request_plan(request(), model=model).contents
        assert "inline_data" not in repr(contents)
        assert "Image unavailable" in repr(contents)
    value = {"inline_data": {"data": PIXELS}, "thought_signature": "opaque-signature"}
    safe = diagnostic_copy(value)
    assert value["inline_data"]["data"] == PIXELS
    assert safe["thought_signature"] == "opaque-signature"
    assert safe["inline_data"]["media_status"] == "not-retained"


def test_media_catalogue_matches_literal_wire_matrix_and_unknown_is_not_incapable():
    from amplifier_module_provider_gemini._media import USER_IMAGE_MODELS, TOOL_IMAGE_MODELS
    assert USER_IMAGE_MODELS == frozenset(CATALOGUE_USER_IDS)
    assert TOOL_IMAGE_MODELS == frozenset(CATALOGUE_TOOL_IDS)
    for model in CATALOGUE_USER_IDS:
        p = GeminiProvider(config={"default_model": model})
        facts = p.get_media_request_support(model=model)
        assert facts["positions"] == (["user", "tool"] if model in CATALOGUE_TOOL_IDS else ["user"])
        assert facts["evidence"].startswith("https://aistudio.google.com/docs/")
        assert "vision" in p.get_info().capabilities
    for model in ("gemini-future", "gemini-flash-latest", "gemini-3-pro-preview",
                  "gemini-4-hydrogen-new", "gemini-3.1-flash-image", "gemini-3.8-live"):
        facts = provider().get_media_request_support(model=model)
        assert facts["positions"] == []
        assert "incapability" in facts["reason"]
    evidence = json.loads(
        (Path(__file__).parents[1] / "contracts" / "gemini-media-evidence.v1.json").read_text()
    )
    assert evidence["audit_date_utc"] == "2026-11-14"
    assert {row["id"] for row in evidence["matrix"]} == set(CATALOGUE_USER_IDS)
    assert {
        row["id"] for row in evidence["matrix"]
        if row["tool"] in ("explicit-G3-guide-scope", "named-multimodal-function-response-example")
    } == set(CATALOGUE_TOOL_IDS)


@pytest.mark.asyncio
async def test_listed_vision_tags_and_media_positions_share_exact_catalogue_not_name_guesses():
    extras = ["gemini-4-future", "gemini-3.8-flash-not-a-model",
              "gemini-flash-latest", "gemini-3.1-flash-image", "gemini-3.8-live"]
    p = provider()
    p._list_models_from_api = AsyncMock(return_value=[
        {"id": model, "generation_methods": ["generateContent", "streamGenerateContent"]}
        for model in (*CATALOGUE_USER_IDS, *extras)
    ])
    listed = await p.list_models(filters={})
    assert {model.id for model in listed} == set(CATALOGUE_USER_IDS) | set(extras)
    for model in listed:
        assert ("vision" in model.capabilities) == (model.id in CATALOGUE_USER_IDS)
        assert ("user" in p.get_media_request_support(model=model.id)["positions"]) == (
            "vision" in model.capabilities
        )