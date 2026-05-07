"""Tests for _cost.py: compute_cost() and _RATES.

Covers:
  (a) Flash flat rate: 1M input → $0.30
  (b) Flash lite flat rate: 1M output → $0.40
  (c) Pro low tier (≤200K prompt): 1M input @ $1.25
  (d) Pro high tier (>200K prompt): 1M input @ $2.50
  (e) Pro-preview alias: same rates as 2.5-pro
  (f) Gemini-3-pro-preview tiered: low/high rates
  (g) Gemini-3.1-pro-preview alias: same rates as 3-pro-preview
  (h) Fresh-input subtraction: cost uses (prompt - cached) for input charge
  (i) Cached double-charge: all-cached request only charges cache_read
  (j) Unknown model returns None
  (k) None != Decimal('0')
  (l) Result type is always Decimal, never float

Integration tests (m–o): _convert_to_chat_response stamps cost_usd on Usage
  (m) Known model + tokens → cost_usd is Decimal > 0
  (n) Fully cached request → cost_usd stamped correctly
  (o) Unknown model → cost_usd is None
"""

from decimal import Decimal
from unittest.mock import MagicMock


from amplifier_module_provider_gemini._cost import compute_cost


# ---------------------------------------------------------------------------
# (a) Flash flat rate: 1M input → $0.30
# ---------------------------------------------------------------------------
def test_flash_input_cost():
    """gemini-2.5-flash: 1M fresh input → $0.30"""
    result = compute_cost(
        "gemini-2.5-flash", prompt_token_count=1_000_000, candidates_token_count=0
    )
    assert result == Decimal("0.30"), f"Expected Decimal('0.30'), got {result!r}"


# ---------------------------------------------------------------------------
# (b) Flash lite flat rate: 1M output → $0.40
# ---------------------------------------------------------------------------
def test_flash_lite_output_cost():
    """gemini-2.5-flash-lite: 1M output → $0.40"""
    result = compute_cost(
        "gemini-2.5-flash-lite", prompt_token_count=0, candidates_token_count=1_000_000
    )
    assert result == Decimal("0.40"), f"Expected Decimal('0.40'), got {result!r}"


# ---------------------------------------------------------------------------
# (c) Pro low tier (≤200K total prompt): 1M input @ $1.25/M
# ---------------------------------------------------------------------------
def test_pro_low_tier_input_cost():
    """gemini-2.5-pro low tier (100K total prompt): 100K input → $0.125"""
    # 100K tokens × $1.25/M = $0.125
    result = compute_cost(
        "gemini-2.5-pro", prompt_token_count=100_000, candidates_token_count=0
    )
    assert result == Decimal("0.125"), f"Expected Decimal('0.125'), got {result!r}"


# ---------------------------------------------------------------------------
# (d) Pro high tier (>200K total prompt): 1M input @ $2.50/M
# ---------------------------------------------------------------------------
def test_pro_high_tier_input_cost():
    """gemini-2.5-pro high tier (500K total prompt): 500K input → $1.25"""
    # 500K tokens × $2.50/M = $1.25
    result = compute_cost(
        "gemini-2.5-pro", prompt_token_count=500_000, candidates_token_count=0
    )
    assert result == Decimal("1.25"), f"Expected Decimal('1.25'), got {result!r}"


# ---------------------------------------------------------------------------
# (e) Pro-preview alias: same rates as gemini-2.5-pro
# ---------------------------------------------------------------------------
def test_pro_preview_alias_low_tier():
    """gemini-2.5-pro-preview: same low-tier rates as gemini-2.5-pro"""
    result_pro = compute_cost(
        "gemini-2.5-pro", prompt_token_count=100_000, candidates_token_count=0
    )
    result_preview = compute_cost(
        "gemini-2.5-pro-preview", prompt_token_count=100_000, candidates_token_count=0
    )
    assert result_pro == result_preview


# ---------------------------------------------------------------------------
# (f) Gemini-3-pro-preview tiered rates
# ---------------------------------------------------------------------------
def test_gemini3_pro_preview_low_tier():
    """gemini-3-pro-preview low tier: 100K input → $0.20"""
    # 100K × $2.00/M = $0.20
    result = compute_cost(
        "gemini-3-pro-preview", prompt_token_count=100_000, candidates_token_count=0
    )
    assert result == Decimal("0.20"), f"Expected Decimal('0.20'), got {result!r}"


def test_gemini3_pro_preview_high_tier():
    """gemini-3-pro-preview high tier (>200K): 500K input → $2.00"""
    # 500K × $4.00/M = $2.00
    result = compute_cost(
        "gemini-3-pro-preview", prompt_token_count=500_000, candidates_token_count=0
    )
    assert result == Decimal("2.00"), f"Expected Decimal('2.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (g) Gemini-3.1-pro-preview alias
# ---------------------------------------------------------------------------
def test_gemini31_pro_preview_alias():
    """gemini-3.1-pro-preview: same rates as gemini-3-pro-preview"""
    result_3 = compute_cost(
        "gemini-3-pro-preview", prompt_token_count=100_000, candidates_token_count=0
    )
    result_31 = compute_cost(
        "gemini-3.1-pro-preview", prompt_token_count=100_000, candidates_token_count=0
    )
    assert result_3 == result_31


# ---------------------------------------------------------------------------
# (h) Fresh-input subtraction: cost uses (prompt - cached) for input charge
# ---------------------------------------------------------------------------
def test_fresh_input_subtraction():
    """Fresh input = prompt_token_count - cached_content_token_count.

    With 50K cached tokens the cost is:
      - fresh_input (150K) × $0.30/M  = $0.045000
      - cache_read  (50K)  × $0.03/M  = $0.001500
      - total                          = $0.046500
    """
    # 200K total, 50K cached → 150K fresh input, 50K cache_read
    # gemini-2.5-flash (flat rate, no tier)
    result = compute_cost(
        "gemini-2.5-flash",
        prompt_token_count=200_000,
        candidates_token_count=0,
        cached_content_token_count=50_000,
    )
    fresh_cost = Decimal("150000") * Decimal("0.30") / Decimal("1000000")
    cache_cost = Decimal("50000") * Decimal("0.03") / Decimal("1000000")
    expected = fresh_cost + cache_cost
    assert result == expected, f"Expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# (i) Cached double-charge: 1M total, all cached → fresh_input=0
# Test from spec: preserve assertion exactly as specified in the plan
# ---------------------------------------------------------------------------
def test_cached_request_does_not_double_charge():
    # 1M total prompt, all cached → fresh_input = 0 → only cache_read_rate applies.
    # 1M > 200K threshold → HIGH tier: high_cache_read_per_m = $0.25/MTok
    # Expected: 1_000_000 × $0.25 / 1_000_000 = $0.25
    result = compute_cost(
        "gemini-2.5-pro",
        prompt_token_count=1_000_000,
        candidates_token_count=0,
        cached_content_token_count=1_000_000,
    )
    assert result == Decimal("0.25")  # high-tier cache_read only (1M > 200K threshold)


# ---------------------------------------------------------------------------
# (j) Unknown model returns None
# ---------------------------------------------------------------------------
def test_unknown_model_returns_none():
    """Unrecognised model name must return None (not 0, not raise)."""
    result = compute_cost("gemini-does-not-exist-9999", prompt_token_count=1_000_000)
    assert result is None, f"Expected None for unknown model, got {result!r}"


# ---------------------------------------------------------------------------
# (k) None != Decimal('0'): unknown is distinct from free
# ---------------------------------------------------------------------------
def test_unknown_distinct_from_zero():
    """None returned for unknown model must not equal Decimal('0')."""
    result = compute_cost("no-such-model", prompt_token_count=0)
    assert result is None
    assert result != Decimal("0")


# ---------------------------------------------------------------------------
# (l) Result type is always Decimal, never float
# ---------------------------------------------------------------------------
def test_result_type_is_decimal():
    """compute_cost must return a Decimal, not a float."""
    result = compute_cost("gemini-2.5-flash", prompt_token_count=1_000)
    assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
    assert not isinstance(result, float), "Result must not be a float"


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def _make_gemini_provider():
    """Create a minimal GeminiProvider for direct method testing (no API key needed)."""
    from amplifier_module_provider_gemini import GeminiProvider

    return GeminiProvider(api_key="test-key", config={})


def _make_response(
    model: str,
    prompt_token_count: int,
    candidates_token_count: int,
    cached_content_token_count: int = 0,
):
    """Build a fake Gemini API response for testing _convert_to_chat_response."""
    part = MagicMock()
    part.text = "Hello"
    part.thought = False
    part.function_call = None

    candidate = MagicMock()
    candidate.content.parts = [part]

    response = MagicMock()
    response.candidates = [candidate]
    response.model = model
    response.usage_metadata.prompt_token_count = prompt_token_count
    response.usage_metadata.candidates_token_count = candidates_token_count
    response.usage_metadata.cached_content_token_count = cached_content_token_count
    response.usage_metadata.total_token_count = (
        prompt_token_count + candidates_token_count
    )
    response.usage_metadata.thoughts_token_count = None
    return response


# ---------------------------------------------------------------------------
# (m) Integration: _convert_to_chat_response stamps cost_usd for known model
# ---------------------------------------------------------------------------
def test_convert_stamps_cost_on_usage():
    """Known model + tokens → result.usage.cost_usd is not None, Decimal, > 0."""
    provider = _make_gemini_provider()
    response = _make_response(
        model="gemini-2.5-flash",
        prompt_token_count=1_000,
        candidates_token_count=500,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd is not None, (
        "cost_usd should be stamped for known model"
    )
    assert isinstance(result.usage.cost_usd, Decimal), (
        f"cost_usd should be Decimal, got {type(result.usage.cost_usd)}"
    )
    assert result.usage.cost_usd > 0, (
        f"cost_usd should be > 0, got {result.usage.cost_usd}"
    )


# ---------------------------------------------------------------------------
# (n) Integration: _convert_to_chat_response handles cached tokens in cost
# ---------------------------------------------------------------------------
def test_convert_stamps_cost_with_cache():
    """Cached tokens are charged at cache_read_per_m; fresh input at input_per_m."""
    provider = _make_gemini_provider()
    response = _make_response(
        model="gemini-2.5-flash",
        prompt_token_count=1_000_000,
        candidates_token_count=0,
        cached_content_token_count=1_000_000,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    # fresh_input=0, cached=1M: cost = 0 * input + 0 * output + 1M * 0.03/1M = 0.03
    assert result.usage.cost_usd == Decimal("0.03"), (
        f"Expected Decimal('0.03') for all-cached flash request, got {result.usage.cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# (o) Integration: _convert_to_chat_response leaves cost_usd=None for unknown model
# ---------------------------------------------------------------------------
def test_convert_leaves_cost_none_for_unknown_model():
    """Unknown model → result.usage.cost_usd is None."""
    provider = _make_gemini_provider()
    response = _make_response(
        model="gemini-unknown-model-9999",
        prompt_token_count=1_000,
        candidates_token_count=500,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd is None, (
        f"cost_usd should be None for unknown model, got {result.usage.cost_usd!r}"
    )
