"""A tool response must survive the transcript/checkpoint JSON boundary."""
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_gemini import GeminiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_tool_response_metadata_is_json_safe_and_signature_round_trips(stream):
    signature = b'\xff\xfe\x80\x00opaque-signature'
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role='model', parts=[
            types.Part(function_call=types.FunctionCall(name='image_generate', args={'action': 'capabilities'}),
                       thought_signature=signature),
        ]), finish_reason='STOP')],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15),
    )
    async def chunks():
        yield response
    provider = GeminiProvider(api_key='synthetic-key', config={'max_retries': 0})
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=AsyncMock(return_value=response),
        generate_content_stream=AsyncMock(return_value=chunks()),
    )))
    result = await provider.complete(ChatRequest(
        messages=[Message(role='user', content='Inspect image capabilities only')], metadata={'stream': stream},
    ))
    # The orchestrator retains provider metadata alongside normalized blocks.
    # This is also the shape the host hashes before fitting the next request.
    message = {'role': 'assistant', 'content': [b.model_dump() for b in result.content],
               'tool_calls': [t.model_dump() for t in result.tool_calls], 'metadata': result.metadata}
    restored = json.loads(json.dumps(message, sort_keys=True, ensure_ascii=False))
    assert 'raw_response' not in restored['metadata']
    assert restored['metadata']['gemini_finish_reason'] == 'STOP'
    assert result.finish_reason == 'tool_calls'
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert json.loads(result.model_dump_json())['tool_calls'][0]['name'] == 'image_generate'
    _, contents = provider._convert_messages([restored])
    call = next(p for p in contents[0]['parts'] if 'function_call' in p)
    assert call['function_call'] == {'name': 'image_generate', 'args': {'action': 'capabilities'}}
    assert call['thought_signature'] == base64.b64encode(signature).decode('ascii')
