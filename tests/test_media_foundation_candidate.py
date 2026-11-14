"""Optional F1 candidate graph fixture; no runtime Foundation dependency.

Run explicitly with the selected Foundation checkout on PYTHONPATH. Absence is
reported as a skip in the standalone provider environment, not integration proof.
"""

import base64
import hashlib
import io
import json

import httpx
import pytest
from PIL import Image
from amplifier_core.message_models import ChatRequest, Message, TextBlock
from amplifier_core.llm_errors import InvalidRequestError

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini._media import NAMESPACE, diagnostic_copy

foundation = pytest.importorskip("amplifier_foundation.session.media")


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke", [False, "epoch", "expiry"])
async def test_real_f1_store_projection_count_send_and_epoch_retry(tmp_path, monkeypatch, revoke):
    from google import genai
    from google.genai import types

    store = foundation.SessionMediaStore(
        tmp_path / "media", "session1",
        policy=foundation.MediaPolicy(1024 * 1024, 10000, 4),
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    cap = foundation.SessionMediaCapability(store, enabled=True)
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    p = GeminiProvider(api_key="offline", coordinator=SimpleNamespace(
        get_capability=lambda key: cap,
        hooks=SimpleNamespace(emit=AsyncMock()),
    ), config={"default_model": "gemini-3.1-pro-preview", "use_streaming": False,
               "max_retries": 1, "min_retry_delay": 0.001})
    image = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(image, format="PNG")
    pixels = image.getvalue()
    origin = {"kind": "tool", "session_id": "session1", "id": "call1"}
    ref = (await cap.ingest(pixels, source_format="image-bytes", origin=origin, policy=None))["value"]
    await cap.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    canonical = ChatRequest(messages=[
        Message(role="assistant", content="", tool_calls=[{"id": "call1", "name": "read_file", "arguments": {}}]),
        Message(role="tool", name="read_file", tool_call_id="call1", content="original",
                metadata={NAMESPACE: {"refs": [ref]}, "owner": {"retained": True}}),
    ])
    before = canonical.model_dump()
    prepared = await cap.prepare_request(canonical, provider_identity={
        "provider": "gemini", "account": "offline", "generation": 1,
        "media_support": p.get_media_request_support(model=p.default_model),
    }, model=p.default_model, request_options={}, purpose="model",
        execution_scope={"nested": False, "text_only": False})
    sent = []

    def generate(req):
        sent.append(json.loads(req.content))
        if revoke:
            if revoke == "epoch":
                store.epoch += 1
            else:
                from amplifier_foundation.session import media_projection
                # Advance the actual witness's expiry decision after the first
                # received error, without renewing/copying the observation.
                monkeypatch.setattr(media_projection, "_expired", lambda ref: True)
            return httpx.Response(503, json={"error": {"code": 503, "message": "retryable"}})
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}]})

    client = genai.Client(api_key="offline", http_options=types.HttpOptions(
        base_url="https://offline.invalid",
        async_client_args={"transport": httpx.MockTransport(generate)},
    ))
    p._client = client
    # Deliberately mark the fixture's mock route as this provider's selected
    # Developer route so request_budget exercises the actual HTTP count seam.
    p._count_route_client = client
    p._client_uses_canonical_developer_route = True
    real_client = httpx.AsyncClient
    counted = []

    def count_client(*, timeout):
        def count(req):
            counted.append(json.loads(req.content))
            return httpx.Response(200, json={"totalTokens": 42})
        return real_client(timeout=timeout, transport=httpx.MockTransport(count))

    try:
        # Unprepared canonical objects never borrow the prepared copy's witness.
        with pytest.raises(InvalidRequestError, match="refused preparation"):
            await p.complete(canonical)
        monkeypatch.setattr(httpx, "AsyncClient", count_client)
        budget = await p.request_budget(prepared.request, context_estimate=100)
        monkeypatch.setattr(httpx, "AsyncClient", real_client)
        assert budget["measurement"]["input_tokens"] == 42
        if revoke:
            with pytest.raises(InvalidRequestError):
                await p.complete(prepared.request)
        else:
            await p.complete(prepared.request)
        assert len(sent) == 1
        blob = sent[0]["contents"][1]["parts"][0]["functionResponse"]["parts"][0]["inline_data"]
        assert hashlib.sha256(base64.b64decode(blob["data"])).hexdigest() == hashlib.sha256(pixels).hexdigest()
        count_body = counted[0]["generateContentRequest"]
        count_body.pop("model")
        assert count_body == sent[0]
        assert canonical.model_dump() == before
    finally:
        prepared.release()
        store.close()
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
async def test_real_f1_flash_user_wire_and_tool_unavailable_frozen_before_count(
    tmp_path, monkeypatch, image_format, mime, streaming, record_property,
):
    """Flash gets admitted user pixels, but tool pixels stay at their warning slot."""
    from google import genai
    from google.genai import types
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    store = foundation.SessionMediaStore(
        tmp_path / "media", "session1",
        policy=foundation.MediaPolicy(1024 * 1024, 10000, 4),
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    cap = foundation.SessionMediaCapability(store, enabled=True)
    p = GeminiProvider(api_key="offline", coordinator=SimpleNamespace(
        get_capability=lambda key: cap,
        hooks=SimpleNamespace(emit=AsyncMock()),
    ), config={"default_model": "gemini-3.1-pro-preview", "use_streaming": streaming,
               "max_retries": 0, "raw": True})
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (19, 83, 147)).save(output, format=image_format)
    pixels = output.getvalue()
    refs = []
    for kind, identity in [("upload", "attachment1"), ("tool", "call1")]:
        origin = {"kind": kind, "session_id": "session1", "id": identity}
        ref = (await cap.ingest(
            pixels, source_format="image-bytes", origin=origin, policy=None,
        ))["value"]
        assert ref["mime"] == mime
        await cap.publish([ref], origin=origin, disposition="retain")
        refs.append(ref)
    store.acknowledge_checkpoint({"refs": refs})
    canonical = ChatRequest(messages=[
        Message(role="user", content=[
            TextBlock(text="before-user"), TextBlock(text=f"[Session image {refs[0]['sha256']}]"),
            TextBlock(text="after-user"),
        ], metadata={NAMESPACE: {"refs": [refs[0]]}, "owner": {"keep": True}}),
        Message(role="assistant", content="", tool_calls=[
            {"id": "call1", "name": "read_file", "arguments": {}},
        ]),
        Message(role="tool", name="read_file", tool_call_id="call1", content=[
            TextBlock(text="before-tool"), TextBlock(text=f"[Session image {refs[1]['sha256']}]"),
            TextBlock(text="after-tool"),
        ], metadata={NAMESPACE: {"refs": [refs[1]]}, "owner": {"call": "call1"}}),
        Message(role="user", content="unchanged-tail"),
    ], max_output_tokens=73)
    canonical_before = canonical.model_dump()
    model = "gemini-2.5-flash"
    options = {"model": model, "temperature": 0.25, "tool_choice": "none"}
    prepared = await cap.prepare_request(canonical, provider_identity={
        "provider": "gemini", "account": "offline", "generation": 1,
        "media_support": p.get_media_request_support(model=model, request_options=options),
    }, model=model, request_options=options, purpose="model",
        execution_scope={"nested": False, "text_only": False})
    projected = prepared.request
    # These assertions run before any count: unavailable is a fitted position,
    # not an SDK fallback or a synthetic user turn added after measuring.
    assert [message.role for message in projected.messages] == ["user", "assistant", "tool", "user"]
    assert [block.type for block in projected.messages[0].content] == ["text", "image", "text"]
    tool_message = projected.messages[2]
    assert tool_message.tool_call_id == "call1"
    assert tool_message.metadata == canonical.messages[2].metadata
    assert [block.type for block in tool_message.content] == ["text", "text", "text"]
    warning = tool_message.content[1].text
    assert "Image unavailable" in warning and "no pixels supplied" in warning
    assert [tool_message.content[0].text, tool_message.content[2].text] == ["before-tool", "after-tool"]
    projected_before = projected.model_dump()
    sent = []
    reply = {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}]}

    def generate(req):
        sent.append((str(req.url), json.loads(req.content)))
        if streaming:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps(reply) + "\n\n",
            )
        return httpx.Response(200, json=reply)

    client = genai.Client(api_key="offline", http_options=types.HttpOptions(
        base_url="https://offline.invalid",
        async_client_args={"transport": httpx.MockTransport(generate)},
    ))
    p._client = p._count_route_client = client
    p._client_uses_canonical_developer_route = True  # Mock route, not a live-route claim.
    real_client = httpx.AsyncClient
    counted = []

    def count_client(*, timeout):
        def count(req):
            assert projected.model_dump() == projected_before
            counted.append((str(req.url), json.loads(req.content)))
            return httpx.Response(200, json={"totalTokens": 42})
        return real_client(timeout=timeout, transport=httpx.MockTransport(count))

    try:
        monkeypatch.setattr(httpx, "AsyncClient", count_client)
        budget = await p.request_budget(projected, context_estimate=100, request_options=options)
        monkeypatch.setattr(httpx, "AsyncClient", real_client)
        assert budget["measurement"]["input_tokens"] == 42
        await p.complete(projected, request_options=options)
        assert len(counted) == len(sent) == 1
        count_url, count_envelope = counted[0]
        send_url, actual = sent[0]
        assert f"/models/{model}:countTokens" in count_url
        method = "streamGenerateContent" if streaming else "generateContent"
        assert f"/models/{model}:{method}" in send_url
        count_body = json.loads(json.dumps(count_envelope["generateContentRequest"]))
        assert count_body.pop("model") == f"models/{model}"
        normalized = json.loads(json.dumps(actual))
        user_blob = normalized["contents"][0]["parts"][1]["inlineData"]
        if "mime_type" in user_blob:
            user_blob["mimeType"] = user_blob.pop("mime_type")
        assert normalized == count_body
        assert user_blob["mimeType"] == mime
        # Match the actual SDK's URL-safe byte encoding, still strict decoding.
        assert base64.b64decode(user_blob["data"], altchars=b"-_", validate=True) == pixels
        tool_wire = actual["contents"][2]["parts"][0]["functionResponse"]
        assert tool_wire["name"] == "read_file"
        assert "parts" not in tool_wire
        assert tool_wire["response"]["result"] == [
            {"text": "before-tool"}, {"text": warning}, {"text": "after-tool"},
        ]
        assert len(actual["contents"]) == 4
        assert actual["contents"][3]["parts"] == [{"text": "unchanged-tail"}]
        assert count_body["generationConfig"]["temperature"] == 0.25
        assert count_body["generationConfig"]["maxOutputTokens"] == 73
        assert count_body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
        assert canonical.model_dump() == canonical_before
        assert projected.model_dump() == projected_before
        events = json.dumps([call.args for call in p.coordinator.hooks.emit.call_args_list])
        assert base64.b64encode(pixels).decode("ascii") not in events
        assert base64.urlsafe_b64encode(pixels).decode("ascii") not in events
        assert "not-retained" in events
        record_property("gemini.media.flash-f1-unavailable-wire", json.dumps({
            "model": model, "mime": mime, "format": image_format, "transport": method,
            "count_send_options_identity": True,
            "user_decoded_sha256": hashlib.sha256(pixels).hexdigest(),
            "tool_projection": "unavailable at original call-bound position before count",
            "sanitized_actual_wire": diagnostic_copy(actual),
        }))
    finally:
        prepared.release()
        store.close()
        await client.aio.aclose()
        client.close()