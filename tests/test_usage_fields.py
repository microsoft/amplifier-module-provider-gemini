"""Tests for Gemini usage field extraction (Phase 2).

Verifies that reasoning_tokens and cache_read_tokens are extracted
from Gemini's usage_metadata and passed to the Usage constructor.
"""

from types import SimpleNamespace

from amplifier_module_provider_gemini import GeminiProvider


def _make_response(
    thoughts_token_count=None,
    cached_content_token_count=None,
    prompt_token_count=100,
    candidates_token_count=50,
    total_token_count=150,
):
    """Create a mock Gemini response with configurable usage_metadata."""
    part = SimpleNamespace(text="Hello", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)

    usage_kwargs = {
        "prompt_token_count": prompt_token_count,
        "candidates_token_count": candidates_token_count,
        "total_token_count": total_token_count,
    }
    if thoughts_token_count is not None:
        usage_kwargs["thoughts_token_count"] = thoughts_token_count
    if cached_content_token_count is not None:
        usage_kwargs["cached_content_token_count"] = cached_content_token_count

    usage = SimpleNamespace(**usage_kwargs)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def test_reasoning_tokens_extracted():
    """thoughts_token_count should map to reasoning_tokens."""
    provider = GeminiProvider(api_key="test-key")
    response = _make_response(thoughts_token_count=200)

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.reasoning_tokens == 200


def test_cache_read_tokens_extracted():
    """cached_content_token_count should map to cache_read_tokens."""
    provider = GeminiProvider(api_key="test-key")
    response = _make_response(cached_content_token_count=1000)

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.cache_read_tokens == 1000


def test_both_new_fields_extracted():
    """Both reasoning_tokens and cache_read_tokens populated together."""
    provider = GeminiProvider(api_key="test-key")
    response = _make_response(thoughts_token_count=300, cached_content_token_count=500)

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.reasoning_tokens == 300
    assert chat_response.usage.cache_read_tokens == 500


def test_missing_fields_are_none():
    """When usage_metadata lacks the new fields, they should be None."""
    provider = GeminiProvider(api_key="test-key")
    response = _make_response()  # No thoughts or cached fields

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.reasoning_tokens is None
    assert chat_response.usage.cache_read_tokens is None


def test_zero_values_preserved():
    """Zero values are preserved — 0 means 'measured, none used', not absent.

    Consistent with OpenAI/vLLM providers.  None means the field was not
    reported by the API at all (tested in test_missing_fields_are_none).
    """
    provider = GeminiProvider(api_key="test-key")
    response = _make_response(thoughts_token_count=0, cached_content_token_count=0)

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.reasoning_tokens == 0
    assert chat_response.usage.cache_read_tokens == 0


def test_standard_usage_fields_still_work():
    """input_tokens, output_tokens, total_tokens should still be populated."""
    provider = GeminiProvider(api_key="test-key")
    response = _make_response(
        prompt_token_count=100,
        candidates_token_count=50,
        total_token_count=150,
    )

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.input_tokens == 100
    assert chat_response.usage.output_tokens == 50
    assert chat_response.usage.total_tokens == 150
