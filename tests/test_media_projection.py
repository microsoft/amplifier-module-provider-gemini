"""Offline media-v1 exact-wire, pretransport and privacy conformance."""

import base64
import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from google import genai
from google.genai import types
from amplifier_core.message_models import ChatRequest, ImageBlock, Message, TextBlock
from amplifier_core.llm_errors import InvalidRequestError, ProviderUnavailableError
from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini._media import NAMESPACE, diagnostic_copy


PIXELS = b"offline-synthetic-original-image-bytes"


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