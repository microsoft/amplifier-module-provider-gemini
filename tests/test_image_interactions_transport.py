"""Qualify generated SDK retry control at the actual HTTP boundary, offline."""
import asyncio
import base64
import json
import struct
from io import BytesIO

import httpx
import pytest
from google import genai
from google.genai import errors, types
from PIL import Image

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini.image_operation import PROTOCOL


def png():
    output = BytesIO()
    Image.new("RGB", (8, 8), "green").save(output, format="PNG")
    return output.getvalue()


def request():
    return {"protocol": PROTOCOL, "request_id": "one", "operation": "generate", "prompt": "one image", "inputs": [],
            "selection": {"model": "gemini-3.1-pro-preview", "config": {"image_generation": {"model": "gemini-nano-banana-2.1"}}},
            "output": {"count": 1, "mime_type": "image/png", "size": "auto", "quality": "auto", "background": "auto"}}


def fixture(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = genai.Client(api_key="fixture-blue", http_options=types.HttpOptions(
        httpx_async_client=http, retry_options=types.HttpRetryOptions(attempts=5)))
    provider = GeminiProvider(api_key="fixture-blue")
    # Test transport replaces only network, with provider-owned provenance.
    provider._client = provider._count_route_client = client
    provider._client_uses_canonical_developer_route = True
    return provider, client, http


@pytest.mark.asyncio
async def test_interactions_final_order_storage_and_single_send():
    sent = []
    def respond(req):
        sent.append(req)
        assert str(req.url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
        assert req.headers["x-goog-api-key"] == "fixture-blue"
        body = json.loads(req.content)
        assert body["model"] == "gemini-nano-banana-2.1"
        assert body["store"] is False and body["stream"] is False
        assert body["response_format"] == {"type": "image", "mime_type": "image/png"}
        assert "previous_interaction_id" not in body
        return httpx.Response(200, json={"id": "interaction-fixture", "status": "completed", "model": "gemini-nano-banana-2.1", "steps": [
            {"type": "thought", "summary": []},
            {"type": "model_output", "content": [{"type": "text", "text": "first"},
                {"type": "image", "mime_type": "image/png", "data": base64.b64encode(png()).decode()},
                {"type": "text", "text": "last"}]}
        ], "usage": {"total_input_tokens": 10, "total_output_tokens": 20, "total_tokens": 30}})
    provider, client, http = fixture(respond)
    try:
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        assert not sent
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "completed", result
        assert [o["kind"] for o in result["outputs"]] == ["text", "image", "text"]
        assert result["outputs"][1]["data"] == png()
        assert len(sent) == 1
        assert client._api_client._http_options.retry_options.attempts == 5
    finally:
        await http.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,expected", [
    pytest.param("omitted", "unknown", id="omitted"),
    pytest.param(None, "unknown", id="null"),
    pytest.param("FINISH_REASON_UNSPECIFIED", "unknown", id="unspecified"),
    pytest.param("FUTURE_FINISH_REASON", "unknown", id="unfamiliar"),
    pytest.param("OTHER", "failed", id="other"),
    pytest.param("MAX_TOKENS", "failed", id="max-tokens"),
    pytest.param("SAFETY", "blocked", id="safety"),
    pytest.param("STOP", "completed", id="stop"),
])
async def test_generate_content_wire_finish_reason_policy(finish, expected):
    original = png()
    candidate = {"content": {"role": "model", "parts": [
        {"text": "first"},
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(original).decode()}},
        {"text": "last"},
    ]}}
    if finish != "omitted":
        candidate["finishReason"] = finish
    sent = []

    def respond(req):
        sent.append(req)
        return httpx.Response(200, json={
            "candidates": [candidate], "responseId": "wire-response",
            "modelVersion": "reported-image-version",
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20,
                              "totalTokenCount": 30, "thoughtsTokenCount": 7},
        })

    provider, client, http = fixture(respond)
    try:
        item = request()
        item["selection"]["config"]["image_generation"]["transport"] = "generate_content"
        preparation = await provider.prepare_image_generation(item)
        assert preparation["supported"] is True, preparation
        plan = preparation["plan"]
        assert not sent
        result = await provider.generate_image(item, plan=plan)
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1
        assert sent[0].method == "POST"
        assert str(sent[0].url) == (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-nano-banana-2.1:generateContent"
        )  # One physical generateContent send, never Interactions or fallback.
        assert result["dispatch_state"] == "response_received"
        assert result["continuation"] is None
        assert result["outputs"] == [
            {"kind": "text", "text": "first"},
            {"kind": "image", "mime_type": "image/png", "data": original},
            {"kind": "text", "text": "last"},
        ]
        assert result["models"] == [
            {"purpose": "image", "requested": plan["model"], "resolved": "reported-image-version"},
        ]
        assert result["upstream"] == {
            "request_id": None, "response_id": "wire-response", "image_call_id": None,
        }
        assert result["usage"]["normalized"] == {
            "input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "reasoning_tokens": 7,
        }
        native = result["usage"]["provider"]
        assert {name: native[name] for name in (
            "prompt_token_count", "candidates_token_count", "total_token_count", "thoughts_token_count",
        )} == {
            "prompt_token_count": 10, "candidates_token_count": 20,
            "total_token_count": 30, "thoughts_token_count": 7,
        }
        assert client._api_client._http_options.retry_options.attempts == 5
        assert result["status"] == expected
        reason = {"completed": None, "blocked": "provider_blocked"}.get(expected, "incomplete_response")
        assert result.get("reason_code") == reason
    finally:
        await http.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["interactions", "generate_content"])
@pytest.mark.parametrize("bad_position", ["before", "after"])
@pytest.mark.parametrize("malformed", ["oversized_text", "image_crc"])
async def test_malformed_text_keeps_independent_image_evidence_once(transport, bad_position, malformed):
    encoded = base64.b64encode(png()).decode()
    corrupt = bytearray(png())
    idat = corrupt.index(b"IDAT")
    length = struct.unpack(">I", corrupt[idat - 4:idat])[0]
    corrupt[idat + 4 + length] ^= 1
    if malformed == "image_crc":
        with Image.open(BytesIO(corrupt)) as image, pytest.raises(SyntaxError):
            image.verify()
    bad_image = base64.b64encode(corrupt).decode()
    if transport == "interactions":
        image = {"type": "image", "mime_type": "image/png", "data": encoded}
        bad = ({"type": "text", "text": "x" * 32001} if malformed == "oversized_text" else
               {"type": "image", "mime_type": "image/png", "data": bad_image})
        parts = [bad, image] if bad_position == "before" else [image, bad]
        payload = {"id": "interaction-fixture", "status": "completed",
                   "model": "gemini-nano-banana-2.1",
                   "steps": [{"type": "model_output", "content": parts}]}
    else:
        image = {"inlineData": {"mimeType": "image/png", "data": encoded}}
        bad = ({"text": "x" * 32001} if malformed == "oversized_text" else
               {"inlineData": {"mimeType": "image/png", "data": bad_image}})
        parts = [bad, image] if bad_position == "before" else [image, bad]
        payload = {"candidates": [{"content": {"role": "model", "parts": parts},
                                   "finishReason": "STOP"}]}
    sent = []

    def respond(req):
        sent.append(req)
        return httpx.Response(200, json=payload)

    provider, client, http = fixture(respond)
    try:
        item = request()
        item["selection"]["config"]["image_generation"]["transport"] = transport
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "failed" and result["reason_code"] == "invalid_response"
        assert result["outputs"] == [{"kind": "image", "mime_type": "image/png", "data": png()}]
        assert result["dispatch_state"] == "response_received"
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1
    finally:
        await http.aclose()
        client.close()


@pytest.mark.asyncio
async def test_prepared_binding_rejects_http_header_change_without_send():
    sent = []
    provider, client, http = fixture(lambda req: sent.append(req))
    try:
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        http.headers["authorization"] = "fixture-changed"
        with pytest.raises(ValueError):
            await provider.generate_image(item, plan=plan)
        assert not sent
    finally:
        await http.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["server", "disconnect", "redirect"])
async def test_legacy_generate_content_also_sends_once(failure):
    from amplifier_module_provider_gemini.images import GeminiImageBackend
    sent = []
    def respond(req):
        sent.append(req)
        if failure == "disconnect":
            raise httpx.RemoteProtocolError("fixture disconnected after possible send")
        if failure == "redirect":
            return httpx.Response(307, headers={"location": "https://other.example.invalid/generate"})
        return httpx.Response(500, json={"error": {"code": 500, "message": "fixture", "status": "INTERNAL"}})
    provider, client, http = fixture(respond)
    http.follow_redirects = True
    try:
        backend = GeminiImageBackend(provider, {"model": "gemini-nano-banana-2.1"})
        with pytest.raises((errors.APIError, httpx.HTTPError)):
            await asyncio.wait_for(backend.generate(action="generate", prompt="fixture",
                images=[], size="auto", quality="auto", background="auto"), 3)
        assert len(sent) == 1 and "generateContent" in sent[0].url.path
        assert http.follow_redirects is True and not http.is_closed
    finally:
        await http.aclose()
        client.close()


@pytest.mark.asyncio
async def test_interactions_server_error_does_not_retry_or_use_generate_content():
    sent = []
    def respond(req):
        sent.append(req)
        return httpx.Response(500, json={"error": {"message": "fixture server error", "code": 500, "status": "INTERNAL"}})
    provider, client, http = fixture(respond)
    try:
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await asyncio.wait_for(provider.generate_image(item, plan=plan), 3)
        assert result["status"] == "unknown"
        assert len(sent) == 1 and sent[0].url.path.endswith("/interactions")
        with pytest.raises(ValueError):
            await provider.generate_image(item, plan=plan)
    finally:
        await http.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["server", "disconnect"])
async def test_generate_content_uses_httpx_once_despite_aiohttp_availability(failure):
    sent = []
    def respond(req):
        sent.append(req)
        if failure == "disconnect":
            raise httpx.RemoteProtocolError("fixture disconnect after possible send")
        return httpx.Response(500, json={"error": {"message": "fixture", "code": 500, "status": "INTERNAL"}})
    provider, client, http = fixture(respond)
    try:
        item = request()
        item["selection"]["config"]["image_generation"]["transport"] = "generate_content"
        original = client._api_client._use_aiohttp
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await asyncio.wait_for(provider.generate_image(item, plan=plan), 3)
        assert result["status"] == "unknown"
        assert len(sent) == 1 and "generateContent" in sent[0].url.path
        assert client._api_client._use_aiohttp == original
    finally:
        await http.aclose()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["interactions", "generate_content"])
async def test_redirect_does_not_replay_or_move_to_another_endpoint(transport):
    sent = []
    def respond(req):
        sent.append(req)
        return httpx.Response(307, headers={"location": "https://other.example.invalid/generate"})
    provider, client, http = fixture(respond)
    http.follow_redirects = True
    try:
        item = request()
        item["selection"]["config"]["image_generation"]["transport"] = transport
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] in {"unknown", "failed"} and len(sent) == 1
        assert http.follow_redirects is True and not http.is_closed
    finally:
        await http.aclose()
        client.close()
