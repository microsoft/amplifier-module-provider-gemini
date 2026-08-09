"""Regression guard: Gemini must never report a fabricated cache_write_tokens.

Investigation finding (verified against the installed google-genai SDK's
actual response type, not just documentation): `GenerateContentResponse
.usage_metadata` (`google.genai.types.GenerateContentResponseUsageMetadata`,
SDK 1.46.0) has no field for tokens written to cache. Its full field set is:
cache_tokens_details, cached_content_token_count, candidates_token_count,
candidates_tokens_details, prompt_token_count, prompt_tokens_details,
thoughts_token_count, tool_use_prompt_token_count,
tool_use_prompt_tokens_details, total_token_count, traffic_type.

This provider also does not use Gemini's *explicit* caching API
(`client.caches.create` / `CachedContent` / `cached_content=`) -- confirmed
absent by grep across the module. It relies solely on *implicit* caching,
which Google documents as automatic and billed like ordinary cache hit/miss
input tokens, with no separate "cache write" count exposed per
generateContent call (unlike Anthropic's cache_creation_input_tokens, or
OpenAI's GPT-5.6 cache_write_tokens under *explicit* prompt_cache_options).

Per the no-fabrication rule, `Usage.cache_write_tokens` must therefore stay
`None` for Gemini -- never estimated, derived, or defaulted to 0 as a stand-in
for "not applicable". This test locks that in so a future change cannot
quietly start synthesizing a value for a metric the API does not provide.
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


def test_cache_write_tokens_is_none_even_with_cache_read_present():
    """A response reporting cache reads must still leave cache_write_tokens
    unset -- Gemini's API has no equivalent field to read it from, and it
    must not be synthesized from cached_content_token_count or any other
    value.
    """
    provider = GeminiProvider(api_key="test-key")
    response = _make_response(cached_content_token_count=1000)

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.cache_read_tokens == 1000
    assert chat_response.usage.cache_write_tokens is None, (
        "cache_write_tokens must stay None -- Gemini's usage_metadata has no "
        "field for it (verified against the installed SDK's response type). "
        "A non-None value here would be a fabricated metric."
    )


def test_cache_write_tokens_is_none_with_no_usage_metadata_fields():
    """Baseline: with no optional fields reported, cache_write_tokens is None
    (same as every other unreported optional field), not a synthesized 0.
    """
    provider = GeminiProvider(api_key="test-key")
    response = _make_response()

    chat_response = provider._convert_to_chat_response(response)

    assert chat_response.usage is not None
    assert chat_response.usage.cache_write_tokens is None


def test_usage_metadata_type_has_no_cache_write_field():
    """Source-level confirmation (not just behavioral): the installed
    google-genai SDK's actual usage_metadata response model exposes no
    cache-write-equivalent field. This pins the investigation finding itself,
    independent of how this provider's code reads it, so an SDK upgrade that
    *adds* such a field is caught here rather than silently continuing to
    report None forever after the API started providing real data.
    """
    from google.genai import types as genai_types

    field_names = set(genai_types.GenerateContentResponseUsageMetadata.model_fields)

    cache_write_like_fields = {
        name
        for name in field_names
        if "cache" in name.lower() and "write" in name.lower()
    }
    assert cache_write_like_fields == set(), (
        "google-genai now exposes a cache-write-like field on "
        f"GenerateContentResponseUsageMetadata: {cache_write_like_fields}. "
        "The 'Gemini reports no cache-write metric' finding in "
        "amplifier_module_provider_gemini/__init__.py needs re-investigation "
        "-- if this is a real, billed metric, it should be surfaced as "
        "Usage.cache_write_tokens instead of left as a documented absence."
    )
