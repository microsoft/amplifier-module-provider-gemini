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
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.llm_errors import InvalidRequestError

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini._media import NAMESPACE

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