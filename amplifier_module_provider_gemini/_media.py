"""Provider-owned media wire and diagnostic projections (no storage owner imports)."""

from __future__ import annotations

import base64
import hashlib
import json
from pydantic import BaseModel
import httpx

NAMESPACE = "experimental.media.v1"
UNAVAILABLE = "[Image unavailable: model or position not qualified for this route; no pixels supplied]"
NOT_RETAINED = {"media_status": "not-retained", "reason": "diagnostic-image-omitted"}

# Exact entries only: input modality evidence is separate from context limits,
# recommendation names, account entitlement and live inference qualification.
MEDIA_EVIDENCE_DATE = "2026-11-14"
# Every key was read in an official model's input-modality table (customtools
# is an explicitly listed version on the Pro page). Never extrapolate a suffix,
# infer vision from a token-limit/recommendation entry, or resolve a latest alias.
MODEL_EVIDENCE = {
    model: f"https://aistudio.google.com/docs/models/{model}"
    for model in (
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    )
}
MODEL_EVIDENCE["gemini-3.1-pro-preview-customtools"] = (
    "https://aistudio.google.com/docs/models/gemini-3.1-pro-preview#gemini-31-pro-preview-customtools"
)
MODEL_EVIDENCE["gemini-robotics-er-2-preview"] = (
    "https://aistudio.google.com/docs/robotics-overview?generate-content=true#model-versions"
)
USER_IMAGE_MODELS = frozenset(MODEL_EVIDENCE)
# The guide explicitly scopes multimodal function responses to Gemini 3 series
# and separately shows Hydrogen by name. Enumerate the documented text-model
# endpoints, not all gemini-3* (e.g. image/TTS models cannot call functions).
TOOL_IMAGE_MODELS = frozenset({
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
    "gemini-3-flash-preview", "gemini-4-hydrogen",
})
TOOL_EVIDENCE = "https://aistudio.google.com/docs/function-calling?generate-content=true#multimodal"
MODEL_NOTES = {
    "gemini-3.7-flash": "Vendor documents automatic routing to gemini-3.8-flash; no local reroute.",
    "gemini-3.5-flash": "Vendor documents automatic routing to gemini-3.6-flash; no local reroute.",
    "gemini-2.5-pro": "Existing-user access; vendor routing changes December 1, 2026; tool-image placement unqualified.",
    "gemini-2.5-flash": "Existing-user access; vendor routing changes December 1, 2026; tool-image placement unqualified.",
    "gemini-2.5-flash-lite": "Existing-user access; tool-image placement unqualified.",
    "gemini-4-carbon": "User image documented; nested tool-image model scope unconfirmed; native count budget unavailable.",
    "gemini-4-argon": "User image documented; nested tool-image model scope unconfirmed; native count budget unavailable.",
    "gemini-4-hydrogen": "Named multimodal function-response example; native count budget unavailable (unified-window semantics).",
    "gemini-robotics-er-2-preview": "GenerateContent image example documented; tool-image scope unconfirmed; native count budget unavailable.",
}
# These models may accept images at the vendor, but their distinct output/API
# contracts are not qualified by this text-provider adapter. This is deliberately
# NOT a declaration of vendor incapability, nor permission to route to them.
SPECIALIZED_IMAGE_MODELS = frozenset({
    "gemini-nano-banana-2.1", "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image", "gemini-3-pro-image", "gemini-2.5-flash-image",
    "gemini-3.8-live", "gemini-3.8-live-extended-thinking",
    "gemini-3.1-flash-live-preview", "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-robotics-er-2-streaming-preview",
})


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
        "evidence_date": MEDIA_EVIDENCE_DATE,
        "reason": MODEL_NOTES.get(
            model,
            "Distinct media-output/Live route not qualified by this text adapter; not vendor image incapability."
            if model in SPECIALIZED_IMAGE_MODELS
            else "Model/position not catalogued for this route; absence of evidence is not vendor incapability."
            if model not in USER_IMAGE_MODELS else None,
        ),
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

