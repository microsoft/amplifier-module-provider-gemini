"""Optional, instance-bound native image operations, independent of chat.

The Interactions adapter isolates request-local retry options while retaining
the selected provider's HTTP transport and authentication. Chat is not mutated.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import uuid
from copy import deepcopy
from io import BytesIO

import httpx

from .images import _MODELS, _PREVIEW, normalize_image

PROTOCOL = "amplifier.image-operation.v1"
MAX_BYTES = 8 * 1024 * 1024
MAX_OUTPUTS = 16
MAX_TEXT = 32000
MIME_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
MODELS = frozenset((*_MODELS, *_PREVIEW))

class _BoundHTTP(httpx.AsyncClient):
    """Request-local no-redirect view; the owning provider retains its client."""
    def __init__(self, original):
        super().__init__(follow_redirects=False, base_url=original.base_url)
        self.original = original

    def build_request(self, *args, **kwargs):
        return self.original.build_request(*args, **kwargs)

    async def request(self, *args, **kwargs):
        kwargs["follow_redirects"] = False
        return await self.original.request(*args, **kwargs)

    async def send(self, request, **kwargs):
        kwargs["follow_redirects"] = False
        return await self.original.send(request, **kwargs)


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def unavailable(reason):
    return {
        "protocol": PROTOCOL,
        "supported": False,
        "reason_code": reason,
        "safe_message": "This connection cannot prepare the requested image operation.",
    }


def _connection_reason(provider):
    settings = provider.config.get("image_generation", {})
    if not isinstance(settings, dict):
        return "invalid_image_configuration"
    if settings.get("enabled") is False:
        return "image_generation_disabled"
    if "enabled" in settings and settings["enabled"] is not True:
        return "invalid_image_permission"
    if not isinstance(provider._api_key, str) or not provider._api_key:
        return "unsupported_connection"
    if provider._client is not None:
        canonical = (
            provider._client is provider._count_route_client
            and provider._client_uses_canonical_developer_route is True
        )
        # A creation marker cannot bless a subsequently mutated SDK endpoint
        # or credential. These checks only narrow our own client provenance;
        # they never qualify an injected client's route.
        api_client = _get(provider._client, "_api_client")
        options = _get(api_client, "_http_options")
        endpoint = _get(options, "base_url")
        key = _get(api_client, "api_key")
        if _get(_get(api_client, "_async_httpx_client"), "_auth") is not None:
            return "unsupported_http_auth"
        if (
            _get(api_client, "vertexai") is True
            or _get(api_client, "enterprise") is True
            or endpoint not in (
                None,
                "https://generativelanguage.googleapis.com",
                "https://generativelanguage.googleapis.com/",
            )
            or (key is not None and key != provider._api_key)
        ):
            canonical = False
    else:
        from . import _sdk_environment_uses_canonical_developer_route

        canonical = _sdk_environment_uses_canonical_developer_route()
    return None if canonical else "unsupported_connection"


def _compatibility_sdk_available():
    try:
        from google.genai import models, types

        return (
            callable(getattr(models.AsyncModels, "generate_content", None))
            and "retry_options" in types.HttpOptions.model_fields
            and "attempts" in types.HttpRetryOptions.model_fields
            and {"http_options", "candidate_count", "response_modalities", "automatic_function_calling"}
            <= types.GenerateContentConfig.model_fields.keys()
            and "disable" in types.AutomaticFunctionCallingConfig.model_fields
        )
    except (ImportError, AttributeError):
        return False


def _interactions_reason(client=None):
    try:
        from importlib.metadata import version

        from google.genai._gaos.google_genai import AsyncGeminiNextGenInteractions
        installed = tuple(int(part) for part in version("google-genai").split(".")[:3])
        if installed < (2, 30, 1) or not callable(AsyncGeminiNextGenInteractions.create):
            return "unsupported_sdk"
    except (ImportError, AttributeError, ValueError):
        return "unsupported_interactions_sdk"
    if client is not None and not callable(getattr(_get(_get(client, "_api_client"), "_http_options"), "model_copy", None)):
        return "unsupported_sdk"
    return None


def _borrow_api_client(owner, transport=None):
    """Borrow only the qualified SDK base, without inheriting cleanup ownership."""
    from google.genai._api_client import BaseApiClient

    api = owner._api_client
    state = getattr(api, "__dict__", None)
    if (
        type(api) is not BaseApiClient
        or not isinstance(state, dict)
        or not {"_http_options", "_httpx_client", "_async_httpx_client"} <= state.keys()
        or not callable(getattr(state["_http_options"], "model_copy", None))
    ):
        raise TypeError("Unsupported SDK API client state for image operations.")

    class _BorrowedApiClient(BaseApiClient):
        # No constructor, transport allocation, ownership flags or async finalizer.
        # Generated request methods bind to this view and its isolated options.
        def __del__(self):
            pass

        def close(self):
            pass

        async def aclose(self):
            pass

    view = object.__new__(_BorrowedApiClient)
    view.__dict__ = state.copy()
    view._borrowed_owner = owner
    if transport is not None:
        view._async_httpx_client = transport
    return view


def _interaction_client(client, timeout, transport=None):
    """Use an isolated SDK options view, never alter a shared client's retries.

    Qualified with google-genai 2.30.1. Its generated Interactions builder maps
    attempts=1 to TWO physical sends; attempts=0 is required here, unlike
    generateContent's attempts=1. The borrowed
    HTTP transport stays owned by the original provider; view cleanup is a no-op.
    Physical one-request tests guard this private SDK seam.
    """
    from google.genai import types
    from google.genai._gaos.google_genai import AsyncGeminiNextGenInteractions
    api = _borrow_api_client(client, transport)
    api._http_options = api._http_options.model_copy(update={
        "retry_options": types.HttpRetryOptions(attempts=0),
        "timeout": None if timeout is None else timeout * 1000,
    })
    return AsyncGeminiNextGenInteractions(api)

def _compatibility_client(client, transport=None):
    """Borrow the bound HTTPX transport; aiohttp has an inner disconnect retry."""
    from google.genai import models
    api = _borrow_api_client(client, transport)
    # This request-local view is never installed on the chat client.
    # The generated model resource uses _use_aiohttp to select its request path.
    api._use_aiohttp = lambda: False
    return models.AsyncModels(api)


def _transport_reason(provider, transport):
    client = provider._client
    if transport == "interactions":
        reason = _interactions_reason(client)
        if reason is not None:
            return reason
    elif (
        transport != "generate_content"
        or not _compatibility_sdk_available()
        or (
            client is not None
            and not callable(
                _get(_get(_get(client, "aio"), "models"), "generate_content")
            )
        )
    ):
        return "unsupported_sdk"
    if client is not None:
        try:
            # These local factories validate the borrowed base/state without
            # constructing transports or sending. Keep the same adapter seam
            # for capability checks, preparation and execution.
            if transport == "interactions":
                _interaction_client(client, None)
            else:
                _compatibility_client(client)
        except (ImportError, AttributeError, TypeError):
            return "unsupported_sdk"
    return None


def capabilities(provider):
    """Local inspection only: no client construction, catalog lookup or login."""
    reason = _connection_reason(provider)
    transports = []
    blocked = {}
    if reason is None:
        for transport in ("interactions", "generate_content"):
            issue = _transport_reason(provider, transport)
            if issue is None:
                transports.append(transport)
            else:
                blocked[transport] = issue
    supported = bool(transports)
    result = {
        "protocol": PROTOCOL,
        "supported": supported,
        "operations": ["generate", "edit"] if supported else [],
        "input_formats": list(MIME_TYPES),
        "output_formats": ["image/png"],
        "transports": transports,
        "unavailable_transports": blocked,
        "entitlement": "unverified",
        "automatic_retries": False,
    }
    if not supported:
        result.update(unavailable(reason or "unsupported_sdk"))
    return result


def raster(data, mime_type):
    """Verify bounded, single-frame raster bytes without altering input bytes."""
    from PIL import Image

    if (
        not isinstance(data, bytes)
        or not 0 < len(data) <= MAX_BYTES
        or not isinstance(mime_type, str)
        or mime_type not in MIME_TYPES
    ):
        raise ValueError("Invalid image type or byte limit.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != MIME_TYPES[mime_type] or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Image format does not match its declared type.")
            if not 0 < image.width <= 4096 or not 0 < image.height <= 4096:
                raise ValueError("Image dimensions exceed the artifact limit.")
            image.verify()
        # verify() alone does not decode every format's pixels.
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (ValueError, OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("Invalid image bytes, format or dimensions.") from None
    return data


def validate(request):
    """Hash the exact plain request, omitting bytes but retaining verified hashes."""
    if not isinstance(request, dict) or set(request) != {
        "protocol", "request_id", "operation", "prompt", "inputs", "selection", "output"
    } or request["protocol"] != PROTOCOL:
        raise ValueError("Unsupported image operation request.")
    if not isinstance(request["request_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", request["request_id"]
    ):
        raise ValueError("Invalid image request identity.")
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_TEXT:
        raise ValueError("Invalid image prompt.")
    inputs = request["inputs"]
    if (
        not isinstance(inputs, list)
        or len(inputs) > 4
        or request["operation"] not in ("generate", "edit")
        or (request["operation"] == "edit") != bool(inputs)
    ):
        raise ValueError("Generate takes no images; edit takes one to four images.")
    compact_inputs = []
    for index, image in enumerate(inputs):
        if not isinstance(image, dict) or set(image) != {"purpose", "mime_type", "data", "sha256"}:
            raise ValueError("Invalid image input fields.")
        data = raster(image["data"], image["mime_type"])
        if hashlib.sha256(data).hexdigest() != image["sha256"]:
            raise ValueError("Invalid image input identity.")
        if image["purpose"] != ("target" if index == 0 else "reference"):
            raise ValueError("Invalid image input order.")
        compact_inputs.append({k: v for k, v in image.items() if k != "data"})
    selection = request["selection"]
    if (
        not isinstance(selection, dict)
        or set(selection) != {"model", "config"}
        or not isinstance(selection["model"], str)
        or not isinstance(selection["config"], dict)
    ):
        raise ValueError("Invalid image selection.")
    config = selection["config"]
    # These top-level preferences describe the conversational role only; they
    # never become native-image controls or permit account/wire overrides.
    if set(config) - {"image_generation", "reasoning_effort", "max_output_tokens"}:
        raise ValueError("Unsupported image preference override.")
    image_config = config.get("image_generation", {})
    if not isinstance(image_config, dict) or set(image_config) - {"model", "transport"}:
        raise ValueError("Unsupported image execution override.")
    output = request["output"]
    if not isinstance(output, dict) or set(output) != {"count", "mime_type", "size", "quality", "background"}:
        raise ValueError("Invalid image output requirements.")
    compact = {k: v for k, v in request.items() if k != "inputs"}
    compact["inputs"] = compact_inputs
    try:
        serialized = json.dumps(compact, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Image preferences must be plain JSON data.") from None
    return hashlib.sha256(serialized.encode()).hexdigest()


def binding(provider):
    """Private identity/credential snapshot: never serialized or hashed publicly."""
    client = provider.client
    api_client = _get(client, "_api_client")
    http_options = _get(api_client, "_http_options")
    http = _get(api_client, "_async_httpx_client")
    return (
        client,
        api_client,
        http,
        deepcopy(dict(_get(http, "headers") or {})),
        _get(http, "_auth"),
        _get(http, "_transport"),
        provider._api_key,
        _get(api_client, "api_key"),
        _get(api_client, "vertexai"),
        _get(api_client, "enterprise"),
        _get(http_options, "base_url"),
        _get(http_options, "api_version"),
        deepcopy(_get(http_options, "headers")),
        deepcopy(provider.config.get("image_generation")),
        provider.timeout,
    )


def _timeout(provider):
    settings = provider.config.get("image_generation", {})
    value = settings.get("timeout", provider.timeout)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Image timeout must be positive finite seconds or None.")
    return float(value)


def _compatibility_request(request, plan):
    from google.genai import types

    parts = [types.Part.from_text(text=request["prompt"])]
    parts.extend(
        types.Part.from_bytes(data=image["data"], mime_type=image["mime_type"])
        for image in request["inputs"]
    )
    return {
        "model": plan["model"],
        "contents": [types.Content(role="user", parts=parts)],
        "config": types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(
                timeout=None if plan["timeout"] is None else plan["timeout"] * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ),
    }


def _interaction_request(request, plan):
    content = [{"type": "text", "text": request["prompt"]}]
    content.extend(
        {
            "type": "image",
            "mime_type": image["mime_type"],
            "data": base64.b64encode(image["data"]).decode("ascii"),
        }
        for image in request["inputs"]
    )
    return {
        "model": plan["model"],
        "input": content,
        "response_modalities": ["text", "image"],
        "response_format": {"type": "image", "mime_type": "image/png"},
        "store": False,
        "stream": False,
    }


async def prepare(provider, request):
    request_hash = validate(request)
    reason = _connection_reason(provider)
    if reason:
        return unavailable(reason)
    config = request["selection"]["config"].get("image_generation", {})
    model = config.get("model")
    transport = config.get("transport", "interactions")
    if not isinstance(model, str) or model not in MODELS or transport not in ("interactions", "generate_content"):
        return unavailable("unsupported_model_or_transport")
    if model == "gemini-3.1-flash-lite-image" and len(request["inputs"]) > 1:
        return unavailable("unsupported_reference_count")
    output = request["output"]
    if type(output["count"]) is not int or output["count"] != 1 or output["mime_type"] != "image/png":
        return unavailable("unsupported_output")
    if any(output[key] != "auto" for key in ("size", "quality", "background")):
        return unavailable("unsupported_controls")
    reason = _transport_reason(provider, transport)
    if reason:
        return unavailable(reason)
    plans = getattr(provider, "_prepared_image_operations", None)
    if plans is None:
        plans = provider._prepared_image_operations = {}
    if len(plans) >= 64:
        raise ValueError("Too many unconsumed image preparations.")
    plan = {
        "protocol": PROTOCOL,
        "id": uuid.uuid4().hex,
        "request_hash": request_hash,
        "transport": transport,
        "model": model,
        "controller": None,
        "output": deepcopy(output),
        "timeout": _timeout(provider),
    }
    try:
        # Force local type/SDK validation before admission. There is no catalog
        # request and no inference about an account's generation entitlement.
        if transport == "generate_content":
            _compatibility_request(request, plan)
        private_binding = binding(provider)
        reason = _transport_reason(provider, transport)
        if reason:
            return unavailable(reason)
        if transport == "interactions":
            _interaction_client(private_binding[0], plan["timeout"])
    except Exception:  # noqa: BLE001 - optional SDK preparation refuses safely
        return unavailable("unsupported_sdk_or_binding")
    reason = _connection_reason(provider) or _transport_reason(provider, transport)
    if reason:
        return unavailable(reason)
    plans[plan["id"]] = (deepcopy(plan), private_binding)
    return {"protocol": PROTOCOL, "supported": True, "plan": plan}


def _append(outputs, item):
    if (
        len(outputs) >= MAX_OUTPUTS
        or sum(len(o.get("text", "")) for o in outputs) + len(item.get("text", "")) > MAX_TEXT
        or sum(len(o.get("data", b"")) for o in outputs) + len(item.get("data", b"")) > 4 * MAX_BYTES
    ):
        raise ValueError("Image operation output exceeds the evidence limit.")
    outputs.append(item)


def _append_image(outputs, data, mime_type, warnings):
    original = raster(data, mime_type)
    normalized = normalize_image(original, mime_type)
    raster(normalized, "image/png")
    if normalized != original:
        warnings.append("native_image_normalized_to_png")
    _append(outputs, {"kind": "image", "mime_type": "image/png", "data": normalized})


def _parts(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError("Invalid or oversized final content.")
    return value


def _parse_interactions(response, outputs, warnings):
    """Walk ordered final steps, never output_image/output_text conveniences."""
    invalid = False
    for step in _parts(_get(response, "steps")):
        if _get(step, "type") != "model_output":
            continue
        try:
            parts = _parts(_get(step, "content"))
        except ValueError:
            invalid = True
            continue
        for part in parts:
            try:
                if _get(part, "thought") is True or _get(part, "type") == "thought":
                    continue
                if _get(part, "type") == "text":
                    text = _get(part, "text")
                    if not isinstance(text, str):
                        raise ValueError("Invalid final text.")
                    _append(outputs, {"kind": "text", "text": text})
                elif _get(part, "type") == "image":
                    data = _get(part, "data")
                    if not isinstance(data, str) or len(data) > 4 * ((MAX_BYTES + 2) // 3):
                        raise ValueError("Missing or oversized inline image data.")
                    # No URI fetch, output convenience property or signature replay.
                    _append_image(outputs, base64.b64decode(data, validate=True), _get(part, "mime_type"), warnings)
                else:
                    raise ValueError("Unsupported final content type.")
            except (ValueError, TypeError, OSError):
                # Keep scanning bounded independent final parts. A malformed
                # part cannot hide an otherwise recoverable paid image.
                invalid = True
    if invalid:
        raise ValueError("Invalid final content; bounded evidence retained.")
    state = _get(response, "status")
    if state == "completed":
        return "completed", None
    if state in ("failed", "cancelled", "incomplete", "budget_exceeded"):
        return "failed", "incomplete_response"
    return "unknown", "incomplete_response"


def _parse_compatibility(response, outputs, warnings):
    feedback = _get(response, "prompt_feedback")
    block = _get(feedback, "block_reason")
    if block and _get(block, "value", block) != "BLOCK_REASON_UNSPECIFIED":
        return "blocked", "provider_blocked"
    candidates = _parts(_get(response, "candidates"))
    state, reason = "completed", None
    invalid = False
    for candidate in candidates:
        finish = _get(candidate, "finish_reason")
        finish = _get(finish, "value", finish)
        if finish in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "SPII", "RECITATION"):
            state, reason = "blocked", "provider_blocked"
        elif (
            finish
            in (
                "MAX_TOKENS",
                "LANGUAGE",
                "OTHER",
                "MALFORMED_FUNCTION_CALL",
                "UNEXPECTED_TOOL_CALL",
                "TOO_MANY_TOOL_CALLS",
                "IMAGE_OTHER",
                "NO_IMAGE",
                "IMAGE_RECITATION",
                "CONTINUATION",
                "MISSING_THOUGHT_SIGNATURE",
                "MALFORMED_RESPONSE",
                "ESCALATION",
                "PUP_LIMITED_DISABLED",
            )
            and state != "blocked"
        ):
            # OTHER is a recognized unsuccessful reason, not an unfamiliar token.
            state, reason = "failed", "incomplete_response"
        elif finish != "STOP" and state == "completed":
            # Missing, unspecified or unfamiliar stopping evidence is unknown.
            # Do not erase an earlier failed/blocked state or skip paid evidence.
            state, reason = "unknown", "incomplete_response"
        try:
            parts = _parts(_get(_get(candidate, "content"), "parts"))
        except ValueError:
            invalid = True
            continue
        for part in parts:
            try:
                if _get(part, "thought") is True:
                    continue
                blob = _get(part, "inline_data")
                text = _get(part, "text")
                if blob is not None:
                    if text:
                        raise ValueError("Ambiguous final content.")
                    _append_image(outputs, _get(blob, "data"), _get(blob, "mime_type"), warnings)
                elif text is not None:
                    if not isinstance(text, str):
                        raise ValueError("Invalid final text.")
                    _append(outputs, {"kind": "text", "text": text})
                else:
                    raise ValueError("Unsupported final content type.")
            except (ValueError, TypeError, OSError):
                invalid = True
    if invalid:
        raise ValueError("Invalid final content; bounded evidence retained.")
    if len(candidates) != 1 and state == "completed":
        state, reason = "failed", "unexpected_candidate_count"
    return state, reason


def _dump(value):
    dump = _get(value, "model_dump")
    result = dump(mode="json") if callable(dump) else value
    if result is not None:
        if not isinstance(result, dict) or len(json.dumps(result, allow_nan=False)) > 65536:
            raise ValueError("Invalid provider usage.")
        result = deepcopy(result)
    return result


def _identity(value):
    return value if isinstance(value, str) and len(value) <= 256 else None


async def generate(provider, request, *, plan):
    request_hash = validate(request)
    plans = getattr(provider, "_prepared_image_operations", {})
    plan_id = plan.get("id") if isinstance(plan, dict) else None
    saved = plans.get(plan_id) if isinstance(plan_id, str) else None
    if saved is None or saved[0] != plan or request_hash != plan["request_hash"]:
        raise ValueError("The image preparation is invalid, consumed or changed.")
    reason = _connection_reason(provider) or _transport_reason(provider, plan["transport"])
    current = binding(provider) if reason is None else None
    if (
        current is None
        or saved[1][0] is not current[0]
        or saved[1][1] is not current[1]
        or saved[1][2:] != current[2:]
    ):
        raise ValueError("The image preparation is bound to a changed connection.")
    # Consume before the first await. Timeout, cancellation or malformed output
    # cannot authorize a direct replay of this plan. Durable identity is the
    # caller's responsibility, not a vendor idempotency guarantee.
    del plans[plan_id]
    # Use our saved copy after admission, not caller-owned mutable options.
    plan = deepcopy(saved[0])
    client = current[0]
    params = (
        _interaction_request(request, plan)
        if plan["transport"] == "interactions"
        else _compatibility_request(request, plan)
    )
    result = {
        "protocol": PROTOCOL,
        "status": "unknown",
        "outputs": [],
        "models": [{"purpose": "image", "requested": plan["model"], "resolved": None}],
        "upstream": {"request_id": None, "response_id": None, "image_call_id": None},
        "usage": {"normalized": None, "provider": None},
        "effective_options": deepcopy(plan["output"]),
        "warnings": [],
        "continuation": None,
        "dispatch_state": "unknown",
    }
    try:
        async with _BoundHTTP(client._api_client._async_httpx_client) as transport:
            if plan["transport"] == "interactions":
                response = await _interaction_client(client, plan["timeout"], transport).create(**params, timeout=plan["timeout"])
            else:
                response = await _compatibility_client(client, transport).generate_content(**params)
    except Exception:  # noqa: BLE001 - unknown paid outcome, never retry
        # Raw SDK errors may contain bodies, URLs or credentials. Cancellation
        # is a BaseException and propagates; either outcome leaves the plan spent.
        result["reason_code"] = "generation_outcome_unknown"
        return result
    result["dispatch_state"] = "response_received"
    try:
        if plan["transport"] == "interactions":
            parser = _parse_interactions
            reported_model = _get(response, "model")
            response_id = _get(response, "id")
            usage = _get(response, "usage")
        else:
            parser = _parse_compatibility
            reported_model = _get(response, "model_version")
            response_id = _get(response, "response_id")
            usage = _get(response, "usage_metadata")
        result["models"][0]["resolved"] = _identity(reported_model)
        result["upstream"]["response_id"] = _identity(response_id)
        result["upstream"]["request_id"] = _identity(_get(response, "_request_id"))
        state, reason = parser(response, result["outputs"], result["warnings"])
        result["usage"]["provider"] = _dump(usage)
        fields = ({"total_input_tokens": "input_tokens", "total_output_tokens": "output_tokens",
                   "total_tokens": "total_tokens", "total_thought_tokens": "reasoning_tokens"}
                  if plan["transport"] == "interactions" else
                  {"prompt_token_count": "input_tokens", "candidates_token_count": "output_tokens",
                   "total_token_count": "total_tokens", "thoughts_token_count": "reasoning_tokens"})
        normalized = {target: _get(usage, source) for source, target in fields.items()
                      if type(_get(usage, source)) is int and _get(usage, source) >= 0}
        result["usage"]["normalized"] = normalized or None
        count = sum(item["kind"] == "image" for item in result["outputs"])
        if state == "completed" and count != plan["output"]["count"]:
            state, reason = "failed", "unexpected_image_count"
        result["status"] = state
        if reason:
            result["reason_code"] = reason
    except Exception:  # noqa: BLE001 - preserve bounded evidence on invalid output
        # Keep previously validated bounded outputs as evidence, not success.
        result["status"] = "failed"
        result["reason_code"] = "invalid_response"
    return result
