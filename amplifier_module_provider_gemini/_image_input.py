"""Narrow GenerateContent image projection; no URL fetching or storage lifecycle."""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel

# Exact documented text-model endpoints, not a family-name vision heuristic.
# https://aistudio.google.com/docs/models
# https://aistudio.google.com/docs/function-calling?generate-content=true#multimodal
# Model/position evidence and this SDK's offline wire proof are not entitlement
# or live inference qualification. Unknown/specialized/Vertex routes stay explicit.
USER_IMAGE_MODELS = frozenset(
    {
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
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-robotics-er-2-preview",
    }
)
TOOL_IMAGE_MODELS = frozenset(
    {
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
        "gemini-4-hydrogen",
    }
)
IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    }
)
UNAVAILABLE = (
    "[Image unavailable: model, route or original position is not qualified; "
    "no pixels supplied. Supply a supported user image or a text description.]"
)
URL_UNAVAILABLE = (
    "[Image unavailable: URL input is not qualified by this adapter; no URL "
    "was fetched or pixels supplied. Supply inline base64 image data instead.]"
)
OMITTED = {"media_status": "not-retained", "reason": "diagnostic-image-omitted"}
IMAGE_ERROR = "Gemini image request failed; provider error content omitted."


def has_images(value: Any) -> bool:
    """Detect canonical and SDK media without parsing arbitrary text."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        return value.get("type") == "image" or any(
            key in {"inline_data", "inlineData", "file_data", "fileData"}
            or has_images(item)
            for key, item in value.items()
        )
    return isinstance(value, (list, tuple)) and any(map(has_images, value))


def media_echoes(value: Any) -> frozenset[str]:
    """Exact request-scoped source strings only, not a generic URL/text filter."""
    found: set[str] = set()
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("source"), dict):
            source = value["source"]
            for key in ("data", "url", "file_id"):
                text = source.get(key)
                if isinstance(text, str) and text:
                    found.add(text)
            encoded = source.get("data")
            if source.get("type") == "base64" and isinstance(encoded, str):
                try:
                    found.add(
                        base64.urlsafe_b64encode(
                            base64.b64decode(encoded, validate=True)
                        ).decode("ascii")
                    )
                except (TypeError, ValueError):
                    pass
        for item in value.values():
            found.update(media_echoes(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(media_echoes(item))
    return frozenset(found - {""})


def diagnostic_copy(value: Any, echoes: frozenset[str] = frozenset()) -> Any:
    """Omit typed media before the first hook; never stringify unknown objects."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        if value.get("type") == "image":
            return dict(OMITTED)
        return {
            key: dict(OMITTED)
            if key in {"inline_data", "inlineData", "file_data", "fileData"}
            else diagnostic_copy(item, echoes)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(item, echoes) for item in value]
    if isinstance(value, bytes):
        return dict(OMITTED)
    if isinstance(value, str):
        # Omit the containing diagnostic string: never risk a partial URL/token
        # substitution and never modify the canonical SDK or response payload.
        if any(echo in value for echo in echoes):
            return "[diagnostic-image-echo-omitted]"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {"diagnostic_status": "not-retained"}


def image_part(block: dict[str, Any], *, supported: bool) -> dict[str, Any]:
    source = block.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("Invalid image source")
    if source.get("type") == "url":
        return {"text": URL_UNAVAILABLE}
    if not supported:
        return {"text": UNAVAILABLE}
    if (
        source.get("type") != "base64"
        or source.get("media_type") not in IMAGE_MIME_TYPES
    ):
        return {
            "text": (
                "[Image unavailable: unsupported source or MIME type; no pixels "
                "supplied. Supply a supported inline base64 image instead.]"
            )
        }
    try:
        encoded = source["data"]
        if not isinstance(encoded, str) or not encoded:
            raise ValueError
        data = base64.b64decode(encoded, validate=True)
        if not data:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid inline image encoding") from None
    # Preserve the established user dictionary representation. The nested tool
    # path below uses decoded bytes and separately proves exact SDK wire output.
    return {"inline_data": {"mime_type": source["media_type"], "data": encoded}}


def result_ids(content: Any) -> set[str]:
    """Only Core's known ToolResultBlock envelope supplies call identity."""
    if not isinstance(content, list):
        return set()
    return {
        block["tool_call_id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("tool_call_id"), str)
        and block["tool_call_id"]
    }


def tool_has_images(content: Any) -> bool:
    """Only canonical block lists and known output envelopes are media input."""
    if isinstance(content, list):
        return any(
            isinstance(block, dict)
            and block.get("type") == "image"
            and isinstance(block.get("source"), dict)
            for block in content
        )
    if (
        isinstance(content, dict)
        and set(content) <= {"success", "error", "output"}
        and ("success" in content or "error" in content)
    ):
        return tool_has_images(content.get("output"))
    return False


def normalize_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unwrap known Core envelopes, retaining their original message/block slot."""
    normalized = []
    for message in messages:
        content = message.get("content")
        if message.get("role") not in {"user", "tool"} or not result_ids(content):
            normalized.append(message)
            continue
        pending = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if pending:
                    normalized.append({**message, "content": pending})
                    pending = []
                call_id = block.get("tool_call_id")
                if message.get("tool_call_id") not in (None, call_id):
                    raise ValueError("Tool result call identity mismatch")
                normalized.append(
                    {
                        **message,
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": block.get("output"),
                        **(
                            {"is_error": block["is_error"]}
                            if "is_error" in block
                            else {}
                        ),
                    }
                )
            else:
                pending.append(block)
        if pending:
            normalized.append({**message, "content": pending})
    return normalized


def tool_response(
    content: Any,
    *,
    name: str,
    call_id: str,
    supported: bool,
    is_error: bool | None = None,
) -> dict[str, Any]:
    """Native call-bound images; result JSON retains text/status and image order."""
    images: list[dict[str, Any]] = []

    def project_block(block: Any) -> Any:
        if (
            not isinstance(block, dict)
            or block.get("type") != "image"
            or not isinstance(block.get("source"), dict)
        ):
            return block
        source = block["source"]
        # Multimodal function responses document PNG/JPEG/WebP, not the wider
        # ordinary user-image MIME set.
        qualified = supported and source.get("media_type") in {
            "image/png",
            "image/jpeg",
            "image/webp",
        }
        part = image_part(block, supported=qualified)
        if "inline_data" not in part:
            return part
        blob = part["inline_data"]
        index = len(images)
        label = f"tool-image-{index}"
        images.append(
            {
                "inline_data": {
                    **blob,
                    "data": base64.b64decode(blob["data"], validate=True),
                    "display_name": label,
                }
            }
        )
        # FunctionResponse.parts is media-only, not arbitrary text/image
        # interleaving. Use the documented unique displayName/$ref association
        # in the original result slot, referencing each native media part once.
        return {"image": {"$ref": label}}

    def project_result(value: Any) -> Any:
        if isinstance(value, list):
            return [project_block(block) for block in value]
        if tool_has_images(value):
            return {**value, "output": project_result(value["output"])}
        return value

    response = {
        "id": call_id,
        "name": name,
        "response": {
            "error"
            if is_error is True
            else "output"
            if is_error is False
            else "result": project_result(content)
        },
    }
    if images:
        response["parts"] = images
    return response
