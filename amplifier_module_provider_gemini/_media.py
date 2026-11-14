"""Provider-owned media wire and diagnostic projections (no storage owner imports)."""

from __future__ import annotations

import base64
import hashlib
import json
from pydantic import BaseModel
import httpx

NAMESPACE = "experimental.media.v1"
UNAVAILABLE = "[Image unavailable: unsupported model or position; no pixels supplied]"
NOT_RETAINED = {"media_status": "not-retained", "reason": "diagnostic-image-omitted"}

# Exact entries only: input modality evidence is separate from context limits,
# recommendation names, account entitlement and live inference qualification.
USER_IMAGE_MODELS = frozenset({"gemini-3.1-pro-preview", "gemini-2.5-flash"})
TOOL_IMAGE_MODELS = frozenset({"gemini-3.1-pro-preview"})
MODEL_EVIDENCE = {
    "gemini-3.1-pro-preview": "https://aistudio.google.com/docs/models/gemini-3.1-pro-preview",
    "gemini-2.5-flash": "https://aistudio.google.com/docs/models/gemini-2.5-flash",
}
TOOL_EVIDENCE = "https://aistudio.google.com/docs/function-calling?generate-content=true#multimodal"


class AttemptHTTP(httpx.AsyncClient):
    """Borrow transport without cleanup ownership or a hidden redirect/auth send."""

    def __init__(self, original, validate):
        # No parent constructor: this view owns no pool, session or descriptors.
        if original.auth is not None:
            raise ValueError("Unqualified media HTTP authentication flow")
        if original.event_hooks.get("request"):
            raise ValueError("Unqualified media HTTP request hooks")
        for transport in [original._transport, *original._mounts.values()]:
            if transport is None or isinstance(transport, httpx.MockTransport):
                continue
            if (
                type(transport) is not httpx.AsyncHTTPTransport
                or getattr(getattr(transport, "_pool", None), "_retries", None) != 0
            ):
                raise ValueError("Unqualified retrying media HTTP transport")
        self.original, self.validate = original, validate

    def build_request(self, *args, **kwargs):
        return self.original.build_request(*args, **kwargs)

    async def request(self, *args, **kwargs):
        self.validate()
        kwargs["follow_redirects"] = False
        return await self.original.request(*args, **kwargs)

    async def send(self, request, **kwargs):
        self.validate()
        kwargs["follow_redirects"] = False
        return await self.original.send(request, **kwargs)

    async def aclose(self):
        pass


def support(model):
    return {
        "version": 1,
        "provider": "gemini",
        "model": model,
        "positions": {
            "user": model in USER_IMAGE_MODELS,
            "tool": model in TOOL_IMAGE_MODELS,
            "assistant": False,
            "system": False,
            "developer": False,
        },
        "mime_types": ["image/png", "image/jpeg"],
        "evidence": [MODEL_EVIDENCE[model]] if model in MODEL_EVIDENCE else [],
        "tool_evidence": TOOL_EVIDENCE if model in TOOL_IMAGE_MODELS else None,
        "qualification": "offline-wire-only; not account entitlement or live inference",
    }


def has_media(request):
    return any(
        NAMESPACE in (message.metadata or {})
        or (
            isinstance(message.content, list)
            and any(getattr(block, "type", None) == "image" for block in message.content)
        )
        for message in request.messages
    )


def has_refs(request):
    return any(NAMESPACE in (message.metadata or {}) for message in request.messages)


def fingerprint(value):
    """Private equality witness; never emitted as a resolvable media reference."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def image_part(block):
    source = block.get("source", {})
    if (
        source.get("type") != "base64"
        or source.get("media_type") not in ("image/png", "image/jpeg")
    ):
        return {"text": "[Image unavailable: unsupported image source; no pixels supplied]"}
    try:
        # SDK dictionary byte fields accept *bytes*, not an already base64 string.
        # Typed count and SDK generation then each encode precisely once.
        data = base64.b64decode(source["data"], validate=True)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid inline image encoding") from None
    return {"inline_data": {"mime_type": source["media_type"], "data": data}}


def content_parts(content, *, images):
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append({"text": block.get("text", "")})
        elif block.get("type") == "image":
            parts.append(image_part(block) if images else {"text": UNAVAILABLE})
    return parts


def diagnostic_copy(value):
    """Remove known image transports BEFORE hooks can serialize their arguments.

    This is a copy, not a wire mutation. An omission is deliberately not a blob
    ref: this provider has no authority to publish diagnostic assets. Never parse
    arbitrary text or stringify an SDK object as a fallback.
    """
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return dict(NOT_RETAINED)
        result = {}
        for key, item in value.items():
            if key in ("inline_data", "inlineData", "file_data", "fileData"):
                result[key] = dict(NOT_RETAINED)
            else:
                result[key] = diagnostic_copy(item)
        return result
    if isinstance(value, bytes):
        return dict(NOT_RETAINED)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, BaseModel):
        return diagnostic_copy(value.model_dump(mode="json", exclude_none=True))
    return {"diagnostic_status": "not-retained"}

