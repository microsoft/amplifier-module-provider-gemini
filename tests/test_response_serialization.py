"""Provider response metadata must survive JSON checkpoint persistence."""

import base64
import json
from types import SimpleNamespace

from google.genai import types

from amplifier_module_provider_gemini import GeminiProvider


def test_sdk_response_metadata_is_json_safe_and_detached():
    signature = b"\xff\xfe\x00"
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="armed", thought_signature=signature)]
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        model_version="gemini-3.7-flash",
    )
    provider = GeminiProvider(
        api_key="synthetic-key", config={"default_model": "gemini-3.7-flash"}
    )

    converted = provider._convert_to_chat_response(response)
    checkpoint = json.loads(json.dumps(converted.metadata))
    json.dumps(converted.model_dump(mode="json"))

    raw = checkpoint["raw_response"]
    assert raw["model_version"] == "gemini-3.7-flash"
    assert raw["candidates"][0]["finish_reason"] == "STOP"
    part = raw["candidates"][0]["content"]["parts"][0]
    assert part["text"] == "armed"
    assert base64.urlsafe_b64decode(part["thought_signature"]) == signature
    response.candidates[0].content.parts[0].text = "changed after conversion"
    assert converted.metadata == checkpoint


def test_stream_assembly_metadata_is_json_safe():
    signature = b"\xff\xfe\x00"
    # This is the provider's streaming aggregation shape, not a native SDK model.
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="armed", thought_signature=signature),
                        SimpleNamespace(
                            function_call=types.FunctionCall(
                                name="probe", args={"nested": [1, True, None]}
                            ),
                            thought_signature=signature,
                        ),
                    ]
                )
            )
        ],
        usage_metadata=None,
    )
    provider = GeminiProvider(api_key="synthetic-key", config={})

    converted = provider._convert_to_chat_response(response)
    checkpoint = json.loads(json.dumps(converted.metadata))
    json.dumps(converted.model_dump(mode="json"))

    parts = checkpoint["raw_response"]["candidates"][0]["content"]["parts"]
    assert parts[0]["text"] == "armed"
    assert parts[1]["function_call"]["args"] == {"nested": [1, True, None]}
    assert base64.urlsafe_b64decode(parts[1]["thought_signature"]) == signature
    assert converted.tool_calls[0].name == "probe"
