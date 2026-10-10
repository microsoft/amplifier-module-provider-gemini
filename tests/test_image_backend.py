from io import BytesIO
from PIL import Image
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types

from amplifier_module_provider_gemini.images import (
    GeminiImageBackend,
    image_generation_info,
    latest_image_model,
    list_image_models,
    register_image_backend,
)


def picture(fmt="PNG"):
    output = BytesIO()
    Image.new("RGB", (4, 4), "blue").save(output, format=fmt)
    return output.getvalue()


def provider(response=None):
    return SimpleNamespace(
        client=SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(return_value=response),
                    list=AsyncMock(),
                )
            )
        )
    )


def response(parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=parts))],
        model_version="gemini-3.1-flash-image",
        response_id="receipt-test",
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=7),
    )


async def generate(backend, **changes):
    args = dict(
        action="generate",
        prompt="A blue square",
        images=[],
        size="auto",
        quality="auto",
        background="auto",
    )
    return await backend.generate(**(args | changes))


@pytest.mark.asyncio
async def test_actual_sdk_request_and_mixed_response():
    p = provider(
        response(
            [
                types.Part(text="A blue square"),
                types.Part(
                    inline_data=types.Blob(data=picture(), mime_type="image/png")
                ),
            ]
        )
    )
    b = GeminiImageBackend(p, {"model": "gemini-3.1-flash-image"})
    result = await generate(b)
    assert result["data"] == picture() and result["text"] == "A blue square"
    assert (
        result["model"] == "gemini-3.1-flash-image"
        and result["request_id"] == "receipt-test"
    )
    config = p.client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.http_options.retry_options.attempts == 1
    assert config.response_modalities == ["TEXT", "IMAGE"]
    assert result["usage"]["prompt_token_count"] == 7


@pytest.mark.asyncio
async def test_edit_reference_and_thought_images():
    p = provider(
        response(
            [
                types.Part(
                    thought=True,
                    inline_data=types.Blob(data=b"thought", mime_type="image/png"),
                ),
                types.Part(
                    inline_data=types.Blob(data=picture(), mime_type="image/png")
                ),
            ]
        )
    )
    b = GeminiImageBackend(p, {"model": "gemini-3.1-flash-image"})
    result = await generate(
        b, action="edit", images=[{"data": b"original", "mimeType": "image/png"}]
    )
    assert result["data"] == picture()
    parts = p.client.aio.models.generate_content.call_args.kwargs["contents"][0].parts
    assert parts[1].inline_data.data == b"original"


@pytest.mark.asyncio
async def test_catalog_is_account_scoped_and_no_inference():
    p = provider()

    async def rows():
        for name in [
            "gemini-3.1-flash-image",
            "gemini-nano-banana-2.1",
            "gemini-3-pro-image-preview",
            "gemini-3.7-flash",
            "imagen-4.0-generate-001",
        ]:
            yield types.Model(
                name="models/" + name, supported_actions=["generateContent"]
            )

    p.client.aio.models.list.return_value = rows()
    catalog = await list_image_models(p)
    assert len(catalog) == 3
    assert latest_image_model(catalog) == "gemini-nano-banana-2.1"
    p.client.aio.models.generate_content.assert_not_called()
    assert image_generation_info()["automaticModel"] == "auto"


@pytest.mark.asyncio
async def test_auto_uses_selected_catalog_and_records_actual_model():
    p = provider(
        response(
            [types.Part(inline_data=types.Blob(data=picture(), mime_type="image/png"))]
        )
    )

    async def rows():
        yield types.Model(
            name="models/gemini-3.1-flash-image", supported_actions=["generateContent"]
        )

    p.client.aio.models.list.return_value = rows()
    result = await generate(GeminiImageBackend(p, {"model": "auto"}))
    assert result["model"] == "gemini-3.1-flash-image"
    assert (
        p.client.aio.models.generate_content.call_args.kwargs["model"]
        == "gemini-3.1-flash-image"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts",
    [
        [],
        [types.Part(text="Cannot generate")],
        [types.Part(inline_data=types.Blob(data=b"jpeg", mime_type="image/jpeg"))],
        [
            types.Part(inline_data=types.Blob(data=b"a", mime_type="image/png")),
            types.Part(inline_data=types.Blob(data=b"b", mime_type="image/png")),
        ],
    ],
)
async def test_invalid_response_never_retries(parts):
    p = provider(response(parts))
    with pytest.raises(ValueError):
        await generate(GeminiImageBackend(p, {"model": "gemini-3.1-flash-image"}))
    assert p.client.aio.models.generate_content.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"quality": "high"},
        {"size": "1024x1024"},
        {"background": "transparent"},
        {"action": "edit"},
        {"images": [{"data": b"x", "mimeType": "image/png"}]},
    ],
)
async def test_unsupported_controls_rejected_before_request(change):
    p = provider()
    with pytest.raises(ValueError):
        await generate(
            GeminiImageBackend(p, {"model": "gemini-3.1-flash-image"}), **change
        )
    p.client.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_network_failure_one_attempt():
    p = provider()
    p.client.aio.models.generate_content.side_effect = TimeoutError("unknown")
    with pytest.raises(TimeoutError):
        await generate(GeminiImageBackend(p, {"model": "gemini-3.1-flash-image"}))
    assert p.client.aio.models.generate_content.await_count == 1


def test_opt_out_duplicate_and_owned_cleanup():
    registry = {}
    c = SimpleNamespace(
        get_capability=lambda name: registry.get(name),
        register_capability=lambda name, value: registry.__setitem__(name, value),
    )
    register_image_backend(c, provider(), {})()
    assert registry == {}
    cleanup = register_image_backend(
        c,
        provider(),
        {"image_generation": {"enabled": True, "id": "images", "model": "auto"}},
    )
    with pytest.raises(ValueError):
        register_image_backend(
            c, provider(), {"image_generation": {"enabled": True, "id": "images"}}
        )
    registry["image.backends"]["other"] = object()
    cleanup()
    assert set(registry["image.backends"]) == {"other"}


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        GeminiImageBackend(provider(), {"model": "auto", "timeout": timeout})


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt,mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
async def test_native_raster_output_normalized_without_new_request(fmt, mime):
    p = provider(
        response(
            [types.Part(inline_data=types.Blob(data=picture(fmt), mime_type=mime))]
        )
    )
    result = await generate(GeminiImageBackend(p, {"model": "gemini-3.1-flash-image"}))
    with Image.open(BytesIO(result["data"])) as image:
        assert image.format == "PNG" and image.size == (4, 4)
        assert image.getpixel((2, 2))[2] > 240
    assert p.client.aio.models.generate_content.await_count == 1
