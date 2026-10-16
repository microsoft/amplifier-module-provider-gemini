"""Offline optional-exchange checks; not transport or paid-call qualification.

The SDK's real generateContent types are used with a local dummy client factory.
These fixtures exercise the optional request/parser with a dummy client.
Actual Interactions and generateContent transport qualification lives separately
in test_image_interactions_transport.py.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini import image_operation as operation
from google.genai import types
from PIL import Image


@pytest.fixture(autouse=True)
def canonical_environment(monkeypatch):
    for name in (
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
    ):
        monkeypatch.delenv(name, raising=False)


def picture(fmt="PNG", size=(4, 4), color="blue"):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format=fmt)
    return stream.getvalue()


def image_input(data=None, mime_type="image/png", purpose="target"):
    data = picture() if data is None else data
    return {
        "purpose": purpose,
        "mime_type": mime_type,
        "data": data,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def request(*, inputs=None, transport="generate_content", model="gemini-3.1-flash-image"):
    return {
        "protocol": operation.PROTOCOL,
        "request_id": "offline-image-1",
        "operation": "edit" if inputs else "generate",
        "prompt": "A blue square",
        "inputs": inputs or [],
        "selection": {
            "model": "gemini-3.8-flash",
            "config": {
                "reasoning_effort": "high",
                "image_generation": {"model": model, "transport": transport},
            },
        },
        "output": {
            "count": 1,
            "mime_type": "image/png",
            "size": "auto",
            "quality": "auto",
            "background": "auto",
        },
    }


def image_part(data=None, mime_type="image/png", **kwargs):
    return types.Part(
        inline_data=types.Blob(data=picture() if data is None else data, mime_type=mime_type),
        **kwargs,
    )


def response(parts=None, *, finish_reason="STOP", model_version=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=[image_part()] if parts is None else parts),
            finish_reason=finish_reason,
        )],
        model_version=model_version,
        response_id="offline-response-id",
    )


@pytest.fixture
def mounted(monkeypatch):
    """A dummy factory follows the provider-owned creation path, not injection."""
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=response()), list=AsyncMock()),
            interactions=SimpleNamespace(create=AsyncMock()),
        ),
        _api_client=SimpleNamespace(
            api_key="synthetic-credential",
            vertexai=False,
            _http_options=SimpleNamespace(
                base_url="https://generativelanguage.googleapis.com/",
                api_version="v1beta",
                headers={"x-goog-api-key": "synthetic-credential"},
            ),
        ),
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr("google.genai.Client", factory)
    monkeypatch.setattr(operation, "_compatibility_client", lambda value, transport=None: value.aio.models)
    class BoundHTTP:
        def __init__(self, _original):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    client._api_client._async_httpx_client = object()
    monkeypatch.setattr(operation, "_BoundHTTP", BoundHTTP)
    provider = GeminiProvider(
        api_key="synthetic-credential",
        config={"default_model": "gemini-3.8-flash", "max_retries": 99},
    )
    provider.complete = AsyncMock(side_effect=AssertionError("Image operation called chat"))
    return provider, client, factory


async def prepared(provider, intent):
    result = await provider.prepare_image_generation(intent)
    assert result["supported"] is True, result
    return result["plan"]


def test_capability_is_local_optional_and_does_not_require_enabled(mounted):
    provider, client, factory = mounted
    descriptor = provider.get_image_generation_capabilities()
    assert descriptor["protocol"] == operation.PROTOCOL
    assert descriptor["supported"] is True
    assert descriptor["transports"] == ["interactions", "generate_content"]
    assert descriptor["entitlement"] == "unverified"
    assert provider._client is None
    factory.assert_not_called()
    client.aio.models.list.assert_not_called()
    client.aio.models.generate_content.assert_not_called()
    client.aio.interactions.create.assert_not_called()
    for method in ("get_image_generation_capabilities", "prepare_image_generation", "generate_image"):
        assert callable(getattr(provider, method))


@pytest.mark.asyncio
async def test_enabled_false_denies_without_constructing_client(mounted):
    provider, _, factory = mounted
    provider.config["image_generation"] = {"enabled": False}
    assert provider.get_image_generation_capabilities()["supported"] is False
    result = await provider.prepare_image_generation(request())
    assert result["reason_code"] == "image_generation_disabled"
    factory.assert_not_called()


@pytest.mark.parametrize("name,value", [
    ("GOOGLE_GEMINI_BASE_URL", "https://proxy.invalid/"),
    ("GOOGLE_GENAI_USE_VERTEXAI", "true"),
    ("GOOGLE_GENAI_USE_ENTERPRISE", "1"),
])
@pytest.mark.asyncio
async def test_noncanonical_environment_is_not_advertised(mounted, monkeypatch, name, value):
    provider, _, factory = mounted
    monkeypatch.setenv(name, value)
    assert provider.get_image_generation_capabilities()["supported"] is False
    assert (await provider.prepare_image_generation(request()))["reason_code"] == "unsupported_connection"
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_injected_client_is_not_claimed_as_canonical(mounted):
    provider, client, _ = mounted
    provider._client = client
    assert provider._count_route_client is None
    assert provider.get_image_generation_capabilities()["supported"] is False
    result = await provider.prepare_image_generation(request())
    assert result["reason_code"] == "unsupported_connection"
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["endpoint", "vertexai", "credential"])
async def test_mutation_before_preparation_cannot_requalify_a_connection(mounted, change):
    provider, client, _ = mounted
    assert provider.client is client
    if change == "endpoint":
        client._api_client._http_options.base_url = "https://proxy.invalid/"
    elif change == "vertexai":
        client._api_client.vertexai = True
    else:
        client._api_client.api_key = "replacement-credential"
    assert provider.get_image_generation_capabilities()["supported"] is False
    assert (await provider.prepare_image_generation(request()))["reason_code"] == "unsupported_connection"
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_default_interactions_fails_closed_without_silent_compatibility(mounted):
    provider, client, _ = mounted
    assert provider.client is client  # Its create callable is present in this fixture.
    intent = request()
    del intent["selection"]["config"]["image_generation"]["transport"]
    result = await provider.prepare_image_generation(intent)
    assert result["supported"] is False
    assert result["reason_code"] == "unsupported_sdk"
    assert "plan" not in result
    client.aio.interactions.create.assert_not_called()
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,method", [("interactions", "create"), ("generate_content", "generate_content")])
async def test_missing_actual_sdk_callable_is_unsupported(mounted, transport, method):
    provider, client, _ = mounted
    assert provider.client is client
    resource = client.aio.interactions if transport == "interactions" else client.aio.models
    setattr(resource, method, None)
    result = await provider.prepare_image_generation(request(transport=transport))
    assert result["supported"] is False
    assert result["reason_code"] in {"unsupported_sdk", "unsupported_interactions_sdk"}


@pytest.mark.asyncio
async def test_explicit_compatibility_sends_native_request_and_no_chat(mounted):
    provider, client, _ = mounted
    client.aio.models.generate_content.return_value = response([
        types.Part(text="Before"), image_part(), types.Part(text="After"),
    ], model_version="reported-image-version")
    intent = request(inputs=[image_input(), image_input(purpose="reference")])
    plan = await prepared(provider, intent)
    assert plan["model"] == "gemini-3.1-flash-image" and plan["controller"] is None
    result = await provider.generate_image(intent, plan=plan)
    sent = client.aio.models.generate_content.call_args.kwargs
    assert sent["model"] == "gemini-3.1-flash-image"
    assert sent["contents"][0].parts[0].text == intent["prompt"]
    assert [p.inline_data.data for p in sent["contents"][0].parts[1:]] == [i["data"] for i in intent["inputs"]]
    assert sent["config"].http_options.retry_options.attempts == 1
    assert sent["config"].http_options.timeout is None
    assert sent["config"].candidate_count == 1
    assert sent["config"].automatic_function_calling.disable is True
    assert sent["config"].response_modalities == ["TEXT", "IMAGE"]
    assert sent["config"].thinking_config is None
    assert result["status"] == "completed"
    assert [o["kind"] for o in result["outputs"]] == ["text", "image", "text"]
    assert result["outputs"][1]["data"] == picture()
    assert result["models"] == [{"purpose": "image", "requested": plan["model"], "resolved": "reported-image-version"}]
    assert result["usage"] == {"normalized": None, "provider": None}
    assert result["upstream"]["response_id"] == "offline-response-id"
    assert result["continuation"] is None
    assert provider.default_model == "gemini-3.8-flash"
    assert provider.config["max_retries"] == 99
    provider.complete.assert_not_called()
    client.aio.models.list.assert_not_called()
    client.aio.models.generate_content.assert_awaited_once()
    client.aio.interactions.create.assert_not_called()


@pytest.mark.parametrize("field,value,reason", [
    ("size", "1536x1024", "unsupported_controls"),
    ("quality", "high", "unsupported_controls"),
    ("background", "transparent", "unsupported_controls"),
    ("count", 2, "unsupported_output"),
    ("count", True, "unsupported_output"),
    ("mime_type", "image/jpeg", "unsupported_output"),
])
@pytest.mark.asyncio
async def test_controls_are_not_ignored(mounted, field, value, reason):
    provider, client, factory = mounted
    intent = request()
    intent["output"][field] = value
    assert (await provider.prepare_image_generation(intent))["reason_code"] == reason
    factory.assert_not_called()
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.parametrize("model", ["auto", "gemini-3.8-flash", "models/gemini-3.1-flash-image", "unknown-image"])
@pytest.mark.asyncio
async def test_model_must_be_exact_qualified_native_id(mounted, model):
    provider, _, factory = mounted
    result = await provider.prepare_image_generation(request(model=model))
    assert result["reason_code"] == "unsupported_model_or_transport"
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_model_specific_reference_limit(mounted):
    provider, client, _ = mounted
    intent = request(model="gemini-3.1-flash-lite-image", inputs=[image_input(), image_input(purpose="reference")])
    assert (await provider.prepare_image_generation(intent))["reason_code"] == "unsupported_reference_count"
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.parametrize("kind", ["hash", "purpose", "url", "bytes", "dimensions", "mime", "truncated", "count", "account_override", "continuation"])
@pytest.mark.asyncio
async def test_invalid_inputs_and_overrides_fail_before_preparation(mounted, kind):
    provider, _, factory = mounted
    intent = request(inputs=[image_input()])
    image = intent["inputs"][0]
    if kind == "hash":
        image["sha256"] = "0" * 64
    elif kind == "purpose":
        image["purpose"] = "reference"
    elif kind == "url":
        image["url"] = "https://example.invalid/image.png"
    elif kind == "bytes":
        intent["inputs"] = [image_input(b"x" * (operation.MAX_BYTES + 1))]
    elif kind == "dimensions":
        intent["inputs"] = [image_input(picture(size=(4097, 1)))]
    elif kind == "mime":
        image["mime_type"] = "image/jpeg"
    elif kind == "truncated":
        intent["inputs"] = [image_input(picture()[:-10])]
    elif kind == "count":
        intent["inputs"] *= 5
    elif kind == "account_override":
        intent["selection"]["config"]["api_key"] = "forbidden"
    elif kind == "continuation":
        intent["previous_interaction_id"] = "unsupported"
    with pytest.raises(ValueError):
        await provider.prepare_image_generation(intent)
    factory.assert_not_called()


def test_intent_hash_includes_verified_digest_but_not_bytes():
    intent = request(inputs=[image_input()])
    plain = deepcopy(intent)
    del plain["inputs"][0]["data"]
    expected = hashlib.sha256(json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert operation.validate(intent) == expected
    changed = deepcopy(intent)
    changed["inputs"] = [image_input(picture(color="red"))]
    assert operation.validate(changed) != expected


@pytest.mark.asyncio
async def test_plan_is_bounded_opaque_random_and_credential_free(mounted):
    provider, client, _ = mounted
    intent = request()
    plans = [await prepared(provider, intent) for _ in range(64)]
    assert len({p["id"] for p in plans}) == 64
    public = json.dumps(plans)
    assert "synthetic-credential" not in public
    assert hashlib.sha256(b"synthetic-credential").hexdigest() not in public
    with pytest.raises(ValueError, match="Too many"):
        await provider.prepare_image_generation(intent)
    assert len(provider._prepared_image_operations) == 64
    private = provider._prepared_image_operations[plans[0]["id"]]
    assert private[1][0] is client
    assert "synthetic-credential" in private[1]
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.parametrize("change", ["plan", "request", "credential", "client", "client_credential", "endpoint", "headers", "permission", "timeout"])
@pytest.mark.asyncio
async def test_tampered_request_plan_or_connection_cannot_dispatch(mounted, change):
    provider, client, _ = mounted
    intent = request()
    plan = await prepared(provider, intent)
    if change == "plan":
        plan["model"] = "gemini-3-pro-image"
    elif change == "request":
        intent["prompt"] = "Changed prompt"
    elif change == "credential":
        provider._api_key = "replacement-credential"
    elif change == "client":
        provider._client = deepcopy(client)
    elif change == "client_credential":
        client._api_client.api_key = "replacement-credential"
    elif change == "endpoint":
        client._api_client._http_options.base_url = "https://proxy.invalid/"
    elif change == "headers":
        client._api_client._http_options.headers["x-goog-api-key"] = "replacement-credential"
    elif change == "permission":
        provider.config["image_generation"] = {"enabled": False}
    elif change == "timeout":
        provider.timeout = 2
    with pytest.raises(ValueError):
        await provider.generate_image(intent, plan=plan)
    client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_single_use_including_concurrent_direct_calls(mounted):
    provider, client, _ = mounted
    intent = request()
    plan = await prepared(provider, intent)
    results = await asyncio.gather(
        provider.generate_image(intent, plan=plan),
        provider.generate_image(intent, plan=deepcopy(plan)),
        return_exceptions=True,
    )
    assert sum(isinstance(r, ValueError) for r in results) == 1
    assert sum(isinstance(r, dict) and r["status"] == "completed" for r in results) == 1
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_caller_mutation_during_await_cannot_change_admitted_output_count(mounted):
    provider, client, _ = mounted
    intent = request()
    plan = await prepared(provider, intent)

    async def dispatched(**kwargs):
        plan["output"]["count"] = 2
        return response([image_part(), image_part()])

    client.aio.models.generate_content.side_effect = dispatched
    result = await provider.generate_image(intent, plan=plan)
    assert result["status"] == "failed"
    assert result["reason_code"] == "unexpected_image_count"
    assert result["effective_options"]["count"] == 1


@pytest.mark.asyncio
async def test_uncertain_send_is_safe_unknown_and_consumes_plan(mounted):
    provider, client, _ = mounted
    client.aio.models.generate_content.side_effect = TimeoutError("synthetic-credential raw body")
    intent = request()
    plan = await prepared(provider, intent)
    result = await provider.generate_image(intent, plan=plan)
    assert result["status"] == "unknown"
    assert result["dispatch_state"] == "unknown"
    assert result["reason_code"] == "generation_outcome_unknown"
    assert "synthetic-credential" not in json.dumps(result)
    with pytest.raises(ValueError, match="consumed"):
        await provider.generate_image(intent, plan=plan)
    client.aio.models.generate_content.assert_awaited_once()
    client.aio.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_propagates_with_plan_spent(mounted):
    provider, client, _ = mounted
    client.aio.models.generate_content.side_effect = asyncio.CancelledError()
    intent = request()
    plan = await prepared(provider, intent)
    with pytest.raises(asyncio.CancelledError):
        await provider.generate_image(intent, plan=plan)
    with pytest.raises(ValueError, match="consumed"):
        await provider.generate_image(intent, plan=plan)
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("parts,expected", [
    ([types.Part(text="Text only")], "unexpected_image_count"),
    ([image_part(), image_part()], "unexpected_image_count"),
    ([image_part(b"invalid")], "invalid_response"),
    ([image_part(picture(size=(4097, 1)))], "invalid_response"),
])
async def test_invalid_cardinality_and_raster_are_not_success(mounted, parts, expected):
    provider, client, _ = mounted
    client.aio.models.generate_content.return_value = response(parts)
    intent = request()
    result = await provider.generate_image(intent, plan=await prepared(provider, intent))
    assert result["status"] == "failed" and result["reason_code"] == expected
    if len(parts) == 2:
        assert len(result["outputs"]) == 2  # bounded evidence retained, no regeneration
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,expected", [("SAFETY", "blocked"), ("MAX_TOKENS", "failed"), (None, "failed")])
async def test_blocked_or_incomplete_response_cannot_complete(mounted, finish, expected):
    provider, client, _ = mounted
    client.aio.models.generate_content.return_value = response(finish_reason=finish)
    intent = request()
    result = await provider.generate_image(intent, plan=await prepared(provider, intent))
    assert result["status"] == expected
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_thoughts_signatures_and_missing_usage_or_version_are_not_invented(mounted):
    provider, client, _ = mounted
    reply = response([
        image_part(b"unvalidated-thought", thought=True),
        types.Part(text="private-thought", thought=True, thought_signature=b"private-signature"),
        image_part(thought_signature=b"private-signature"),
    ])
    reply.usage_metadata = types.GenerateContentResponseUsageMetadata(thoughts_token_count=7)
    client.aio.models.generate_content.return_value = reply
    intent = request()
    result = await provider.generate_image(intent, plan=await prepared(provider, intent))
    assert result["status"] == "completed" and len(result["outputs"]) == 1
    assert result["models"][0]["resolved"] is None
    assert result["usage"]["normalized"] == {"reasoning_tokens": 7}
    assert result["usage"]["provider"]["thoughts_token_count"] == 7
    assert "private-thought" not in repr(result)
    assert "private-signature" not in repr(result)


@pytest.mark.parametrize("fmt,mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
@pytest.mark.asyncio
async def test_output_normalization_is_recorded_without_changing_original_inputs(mounted, fmt, mime):
    provider, client, _ = mounted
    original = picture(fmt)
    client.aio.models.generate_content.return_value = response([image_part(original, mime)])
    intent = request(inputs=[image_input(original, mime)])
    result = await provider.generate_image(intent, plan=await prepared(provider, intent))
    assert result["status"] == "completed"
    assert result["warnings"] == ["native_image_normalized_to_png"]
    assert intent["inputs"][0]["data"] == original
    with Image.open(BytesIO(result["outputs"][0]["data"])) as decoded:
        assert decoded.format == "PNG" and decoded.size == (4, 4)
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_configured_timeout_is_frozen_and_sent_in_sdk_milliseconds(mounted):
    provider, client, _ = mounted
    provider.config["image_generation"] = {"timeout": 2.5}
    intent = request()
    plan = await prepared(provider, intent)
    await provider.generate_image(intent, plan=plan)
    assert plan["timeout"] == 2.5
    assert client.aio.models.generate_content.call_args.kwargs["config"].http_options.timeout == 2500


def test_interactions_request_uses_typed_inline_base64_and_no_server_state():
    intent = request(inputs=[image_input(), image_input(purpose="reference")])
    payload = operation._interaction_request(intent, {"model": "gemini-3.1-flash-image"})
    assert payload["model"] == "gemini-3.1-flash-image"
    assert payload["store"] is False and payload["stream"] is False
    assert payload["response_modalities"] == ["text", "image"]
    assert payload["input"][0] == {"type": "text", "text": "A blue square"}
    assert all(part["type"] == "image" for part in payload["input"][1:])
    assert [base64.b64decode(part["data"], validate=True) for part in payload["input"][1:]] == [i["data"] for i in intent["inputs"]]
    assert "previous_interaction_id" not in payload
    assert "retry_options" not in payload  # not a documented Interactions body field


def test_interactions_parser_walks_final_steps_in_order_not_convenience_fields():
    image = {"type": "image", "mime_type": "image/png", "data": base64.b64encode(picture()).decode()}
    reply = {
        "status": "completed",
        "output_image": {"data": "WRONG convenience last image"},
        "output_text": "WRONG convenience text",
        "steps": [
            {"type": "thought", "content": [{"type": "image", "data": "private-thought"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "Before"}, image]},
            {"type": "model_output", "content": [{"type": "text", "text": "After"}, image]},
        ],
    }
    outputs, warnings = [], []
    assert operation._parse_interactions(reply, outputs, warnings) == ("completed", None)
    assert [o["kind"] for o in outputs] == ["text", "image", "text", "image"]
    assert outputs[0]["text"] == "Before" and outputs[2]["text"] == "After"
    assert "private-thought" not in repr(outputs) and "WRONG" not in repr(outputs)
    # The generation wrapper, not a convenience field, enforces count == 1.
    assert sum(o["kind"] == "image" for o in outputs) == 2


@pytest.mark.parametrize("part", [
    {"type": "image", "mime_type": "image/png", "uri": "https://example.invalid/image.png"},
    {"type": "image", "mime_type": "image/png", "data": "invalid-base64"},
    {"type": "audio", "data": "not-an-image"},
])
def test_interactions_parser_rejects_remote_or_unsupported_final_outputs(part):
    with pytest.raises(ValueError):
        operation._parse_interactions(
            {"status": "completed", "steps": [{"type": "model_output", "content": [part]}]}, [], [],
        )
