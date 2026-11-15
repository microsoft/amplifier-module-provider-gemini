"""Provider-owned media wire and diagnostic projections (no storage owner imports)."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
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


_HTTP_DATA = frozenset({
    "_auth", "_base_url", "_cookies", "_default_encoding", "_event_hooks",
    "_headers", "_mounts", "_params", "_state", "_timeout", "_transport",
    "_trust_env", "follow_redirects", "max_redirects",
})
_HTTP_EXECUTION = frozenset(
    name for base in httpx.AsyncClient.__mro__
    for name, value in vars(base).items() if callable(value)
)


def _http_state(client):
    """Qualified HTTPX state; no secrets or URLs are emitted."""
    if _HTTP_EXECUTION.intersection(vars(client)):
        raise ValueError("Unqualified media HTTP execution override")
    if client.auth is not None:
        raise ValueError("Unqualified media HTTP authentication flow")
    if client.event_hooks.get("request") or client.event_hooks.get("response"):
        raise ValueError("Unqualified media HTTP request hooks")
    if (
        client.params or client.cookies
        or client.base_url.userinfo or client.base_url.query or client.base_url.fragment
        or any(name in client.headers for name in ("authorization", "proxy-authorization", "cookie"))
    ):
        raise ValueError("Unqualified media HTTP credential defaults")
    for transport in [client._transport, *client._mounts.values()]:
        if transport is None or isinstance(transport, httpx.MockTransport):
            continue
        if (
            type(transport) is not httpx.AsyncHTTPTransport
            or getattr(getattr(transport, "_pool", None), "_retries", None) != 0
        ):
            raise ValueError("Unqualified retrying media HTTP transport")
    return (
        tuple(client.headers.multi_items()), str(client.base_url),
        tuple(client.params.multi_items()),
        tuple((c.name, c.value, c.domain, c.path) for c in client.cookies.jar),
        id(client._transport),
        tuple((id(pattern), id(transport)) for pattern, transport in client._mounts.items()),
    )


def validate_wire(request, *, base, key, model, purpose):
    """Validate the concrete, merged SDK/HTTP request, not an API-key property."""
    path = f"models/{model.removeprefix('models/')}"
    root = str(base).rstrip("/") + "/v1beta/" + path
    urls = (
        [root + ":countTokens"] if purpose == "count" else
        [root + ":generateContent", root + ":streamGenerateContent?alt=sse"]
    )
    authority = request.url.netloc.decode("ascii")
    if (
        request.method != "POST"
        or request.url not in [httpx.URL(url) for url in urls]
        or request.url.userinfo or request.url.fragment
        or request.headers.get_list("x-goog-api-key") != [key]
        or request.headers.get_list("host") != [authority]
        or set(request.extensions) - {"timeout"}
        or any(name in request.headers for name in ("authorization", "proxy-authorization", "cookie"))
    ):
        raise ValueError("Gemini media effective HTTP request refused")


class MediaConnection:
    """Attest a selected static Developer connection before freezing a plan."""

    def __init__(self, client, *, api_key, base_url):
        from google.genai._api_client import BaseApiClient
        api = client._api_client
        if type(api) is not BaseApiClient:
            raise ValueError("Unqualified media SDK connection")
        self.client, self.api = client, api
        self.sdk_execution = frozenset(
            name for base in BaseApiClient.__mro__
            for name, value in vars(base).items() if callable(value)
        )
        self.options, self.http = api._http_options, api._async_httpx_client
        self.key = api_key
        self.base = httpx.URL(base_url)
        if (
            not isinstance(self.key, str) or not self.key or self.key != self.key.strip()
            or self.base.scheme != "https" or self.base.userinfo
            or self.base.query or self.base.fragment
        ):
            raise ValueError("Unqualified media static connection")
        self.headers = deepcopy(self.options.headers)
        self.http_state = _http_state(self.http)
        self.routing = self._routing()
        self.validate()

    def _routing(self):
        api = self.api
        return (
            api.custom_base_url, api.project, api.location,
            getattr(api, "_async_httpx_use_mtls_endpoint", False),
        )

    def validate(self):
        api, options = self.api, self.options
        headers = httpx.Headers(options.headers or {})
        if (
            self.sdk_execution.intersection(vars(api))
            or self.client._api_client is not api
            or api._http_options is not options
            or api._async_httpx_client is not self.http
            or api.api_key != self.key
            or api.vertexai or getattr(api, "enterprise", False)
            or getattr(api, "_credentials", None) is not None
            or httpx.URL(options.base_url) != self.base
            or options.api_version != "v1beta"
            or options.extra_body
            or getattr(options.base_url_resource_scope, "value", options.base_url_resource_scope)
            not in (None, "DEFAULT")
            or headers.get_list("x-goog-api-key") != [self.key]
            or any(name in headers for name in ("authorization", "proxy-authorization", "cookie"))
            or options.headers != self.headers
            or self._routing() != self.routing or self.routing[-1]
            or _http_state(self.http) != self.http_state
        ):
            raise ValueError("Gemini media effective connection changed")

    def wire(self, request, *, model, purpose):
        self.validate()
        validate_wire(request, base=self.base, key=self.key, model=model, purpose=purpose)

    def isolated_options(self):
        # No deepcopy of transports/locks: only isolate mutable wire options.
        return self.options.model_copy(update={
            "headers": deepcopy(self.headers),
            "retry_options": deepcopy(self.options.retry_options),
        })


class AttemptHTTP(httpx.AsyncClient):
    """Request-local HTTPX state, borrowing transports but never their cleanup."""

    def __init__(self, original, validate, wire=None):
        state = _http_state(original)
        # No constructor/pool allocation. Do not invoke the owner's request/send
        # extension points: those can add auth/hooks after our earlier check.
        # Only qualified HTTPX DATA state. Never copy instance methods,
        # descriptors or extension attributes that could shadow this guard.
        self.__dict__ = {name: original.__dict__[name] for name in _HTTP_DATA}
        self._headers = httpx.Headers(original.headers)
        self._params = httpx.QueryParams(original.params)
        self._cookies = httpx.Cookies(original.cookies)
        self._mounts = original._mounts.copy()
        self._event_hooks = {"request": [], "response": []}
        self.original, self.validate, self.wire = original, validate, wire
        self.owner_state = state

    async def request(self, *args, **kwargs):
        kwargs["follow_redirects"] = False
        return await super().request(*args, **kwargs)

    async def send(self, request, **kwargs):
        self._validate(request)
        if kwargs.get("auth") not in (None, httpx.USE_CLIENT_DEFAULT):
            raise ValueError("Unqualified media HTTP authentication flow")
        kwargs["auth"] = None
        kwargs["follow_redirects"] = False
        return await super().send(request, **kwargs)

    def _validate(self, request):
        self.validate()
        if (
            _http_state(self.original) != self.owner_state
            or _http_state(self) != self.owner_state
        ):
            raise ValueError("Gemini media HTTP connection changed")
        if self.wire is not None:
            self.wire(request)

    async def _send_single_request(self, request):
        # Beneath HTTPX auth/hooks, directly before the selected transport.
        self._validate(request)
        return await super()._send_single_request(request)

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

