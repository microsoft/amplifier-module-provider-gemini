"""Qualify generated SDK retry control at the actual HTTP boundary, offline."""
import asyncio
import base64
import json
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
