"""Native Gemini image backend for the portable image tool.

Wire support follows Google's generateContent image documentation:
https://ai.google.dev/gemini-api/docs/generate-content/image-generation
Catalog visibility is never treated as generation entitlement.
"""

from __future__ import annotations

import math
from io import BytesIO

# Maintained, qualified wire families, ordered by the documented default path.
# Unknown future models are not silently assumed to share these capabilities.
_MODELS = (
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image",
    "gemini-nano-banana-2.1",
)
_PREVIEW = ("gemini-3-pro-image-preview",)


def image_generation_info():
    return {
        "schemaVersion": 1,
        "configKey": "image_generation",
        "defaultBackendId": "gemini",
        "operations": ["generate", "edit"],
        "outputFormats": ["png"],
        "modelDiscovery": True,
        "automaticModel": "auto",
        "entitlement": "unverified",
    }


async def list_image_models(provider):
    from google.genai import types

    config = types.ListModelsConfig(
        http_options=types.HttpOptions(
            timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)
        )
    )
    pager = await provider.client.aio.models.list(config=config)
    models = set()
    async for row in pager:
        identity = (row.name or "").removeprefix("models/")
        if identity in (*_MODELS, *_PREVIEW) and "generateContent" in (
            row.supported_actions or []
        ):
            models.add(identity)
    return [
        {"id": model, "display_name": model, "entitlement": "unverified"}
        for model in sorted(models)
    ]


def latest_image_model(rows):
    available = {row["id"] for row in rows}
    for model in reversed(_MODELS):
        if model in available:
            return model
    raise ValueError(
        "No supported stable Gemini image model is available. Choose an image model explicitly."
    )


class GeminiImageBackend:
    def __init__(self, provider, config):
        self.provider = provider
        self.model = config.get("model")
        self.timeout = config.get("timeout")
        if self.timeout is not None:
            try:
                seconds = float(self.timeout)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "image_generation.timeout must be positive finite seconds or None"
                ) from None
            if (
                isinstance(self.timeout, bool)
                or not math.isfinite(seconds)
                or seconds <= 0
            ):
                raise ValueError(
                    "image_generation.timeout must be positive finite seconds or None"
                )
            self.timeout = seconds

    def describe(self):
        configured = self.model in (*_MODELS, *_PREVIEW, "auto")
        return {
            "provider": "gemini",
            "model": self.model,
            "modelSelection": "automatic" if self.model == "auto" else "explicit",
            "configured": configured,
            "missing": [] if configured else ["supported_image_model"],
            "operations": ["generate", "edit"],
            "outputFormats": ["png"],
            "defaults": {"size": "auto", "quality": "auto", "background": "auto"},
            "options": {"size": ["auto"], "quality": ["auto"], "background": ["auto"]},
            "entitlement": "unverified",
            "automaticRetries": False,
        }

    async def generate(self, *, action, prompt, images, size, quality, background):
        from google.genai import types

        if not self.describe()["configured"]:
            raise ValueError("Choose a supported Gemini image_generation.model.")
        if action not in {"generate", "edit"} or (action == "edit") != bool(images):
            raise ValueError(
                "Generation takes no image inputs; edit needs image inputs."
            )
        if size != "auto" or quality != "auto" or background != "auto":
            raise ValueError(
                "Gemini currently supports automatic size, quality and background only; explicit image controls are not silently ignored."
            )
        model = self.model
        if model == "auto":
            model = latest_image_model(await list_image_models(self.provider))
        if model == "gemini-3.1-flash-lite-image" and len(images) > 1:
            raise ValueError(
                "This Gemini image model supports only one reference in this adapter."
            )
        parts = [types.Part.from_text(text=prompt)]
        for image in images:
            if image["mimeType"] not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Unsupported Gemini image reference type.")
            parts.append(
                types.Part.from_bytes(data=image["data"], mime_type=image["mimeType"])
            )
        from .image_operation import _BoundHTTP, _compatibility_client
        client = self.provider.client
        params = {
            "model": model,
            "contents": [types.Content(role="user", parts=parts)],
            "config": types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                candidate_count=1,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                http_options=types.HttpOptions(
                    timeout=None if self.timeout is None else self.timeout * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ),
        }
        if hasattr(client, "_api_client"):
            async with _BoundHTTP(client._api_client._async_httpx_client) as transport:
                response = await _compatibility_client(client, transport).generate_content(**params)
        else:
            # Explicit non-SDK backends retain their declared transport contract.
            response = await client.aio.models.generate_content(**params)
        outputs, text = [], []
        complete = len(response.candidates or []) == 1 and not (
            response.prompt_feedback and response.prompt_feedback.block_reason
        )
        for candidate in response.candidates or []:
            complete = complete and candidate.finish_reason == types.FinishReason.STOP
            for part in (candidate.content.parts if candidate.content else []) or []:
                if part.thought:
                    continue
                if part.inline_data:
                    blob = part.inline_data
                    if blob.mime_type not in {
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                    } or not isinstance(blob.data, bytes):
                        raise ValueError("Gemini returned an unsupported image format.")
                    if not blob.data or len(blob.data) > 8 * 1024 * 1024:
                        raise ValueError(
                            "Gemini image exceeds the bounded artifact limit or is empty."
                        )
                    outputs.append(normalize_image(blob.data, blob.mime_type))
                elif part.text:
                    text.append(part.text)
        if len(outputs) != 1:
            raise ValueError(
                "Gemini did not return exactly one final image; no automatic retry was made."
            )
        return {
            "status": "completed" if complete else "failed",
            **({"reason": "incomplete_image_result"} if not complete else {}),
            "data": outputs[0],
            "model": response.model_version or model,
            "request_id": response.response_id,
            "usage": response.usage_metadata.model_dump(mode="json")
            if response.usage_metadata
            else None,
            "text": "\n".join(text),
        }


def normalize_image(data, mime_type):
    """Validate provider bytes and normalize supported raster output to PNG."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
    try:
        decoded = Image.open(BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Gemini returned invalid image bytes.") from exc
    with decoded as image:
        if image.format != formats[mime_type] or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Gemini image bytes do not match the declared format.")
        if image.width * image.height > 16_777_216:
            raise ValueError("Gemini image dimensions exceed the artifact limit.")
        image.load()
        # Preserve original PNG bytes, including provenance metadata, where possible.
        if image.format == "PNG" and image.mode in {"RGB", "RGBA"}:
            return data
        output = BytesIO()
        ImageOps.exif_transpose(image).convert(
            "RGBA" if "A" in image.getbands() else "RGB"
        ).save(output, format="PNG")
        result = output.getvalue()
        if len(result) > 8 * 1024 * 1024:
            raise ValueError("Normalized Gemini image exceeds the artifact limit.")
        return result


def register_image_backend(coordinator, provider, config):
    settings = config.get("image_generation") or {}
    if settings.get("enabled") is not True:
        return lambda: None
    identity = settings.get("id", "gemini")
    if not isinstance(identity, str) or not identity or len(identity) > 100:
        raise ValueError("image_generation.id must be a nonempty backend identifier")
    backends = dict(coordinator.get_capability("image.backends") or {})
    if identity in backends:
        raise ValueError("An image backend with this id is already registered")
    backend = GeminiImageBackend(provider, settings)
    backends[identity] = backend
    coordinator.register_capability("image.backends", backends)

    def cleanup():
        current = dict(coordinator.get_capability("image.backends") or {})
        if current.get(identity) is backend:
            del current[identity]
            coordinator.register_capability("image.backends", current)

    return cleanup
