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

Defect B regression tests (p–s): thinking tokens billed at output rate
  (p) Measured live scenario: gemini-3-flash-preview, 10,101 in / 4 out /
      228 thoughts → cost matches the corrected ≈$0.0057465 (not the
      previously-reported $0.0050625, an ~13.5% under-report).
  (q) thoughts_token_count defaults to 0 → identical to omitting it entirely
      (backward compatible for non-thinking calls).
  (r) Thinking tokens billed at the SAME rate as candidates (tiered model).
  (s) thoughts_token_count=0 explicitly behaves like omission.

Defect D regression tests (t–w): new-model rates + still-missing model
  (t) gemini-3.5-flash flat rate: $1.50 / $9.00 / $0.15 per 1M
  (u) gemini-3.6-flash flat rate: $1.50 / $7.50 / $0.15 per 1M
  (v) gemini-3.1-flash-lite flat rate: $0.25 / $1.50 / $0.025 per 1M
  (w) gemini-3.1-flash-image-preview remains unpriced (no confirmable
      official rate) -- returns None, not a guessed number.
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
    fresh_cost = Decimal(150000) * Decimal("0.30") / Decimal(1000000)
    cache_cost = Decimal(50000) * Decimal("0.03") / Decimal(1000000)
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
    assert result != Decimal(0)


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
    result = provider._convert_to_chat_response(response, model="gemini-2.5-flash")
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
    result = provider._convert_to_chat_response(response, model="gemini-2.5-flash")
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
    result = provider._convert_to_chat_response(
        response, model="gemini-unknown-model-9999"
    )
    assert result.usage is not None
    assert result.usage.cost_usd is None, (
        f"cost_usd should be None for unknown model, got {result.usage.cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# Defect B: thinking tokens billed at output rate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (p) Measured live scenario (bat-and-ball prompt, gemini-3-flash-preview):
#     input_tokens=10101, output_tokens=4, reasoning_tokens=228.
#     Previously reported cost_usd = "0.0050625" (thinking tokens excluded).
#     Corrected cost must be ≈$0.0057465 -- an ~13.5% under-report before
#     this fix (0.0006840 / 0.0050625 == 0.1351...).
# ---------------------------------------------------------------------------
def test_measured_defect_b_scenario_bills_thinking_tokens():
    """Reproduces the exact measured defect-B scenario from the bug report."""
    old_wrong_cost = compute_cost(
        "gemini-3-flash-preview",
        prompt_token_count=10_101,
        candidates_token_count=4,
        # thoughts_token_count omitted -> old (buggy) behaviour
    )
    assert old_wrong_cost is not None
    assert old_wrong_cost == Decimal("0.0050625"), (
        f"Sanity check on old behaviour failed: got {old_wrong_cost!r}"
    )

    corrected_cost = compute_cost(
        "gemini-3-flash-preview",
        prompt_token_count=10_101,
        candidates_token_count=4,
        thoughts_token_count=228,
    )
    assert corrected_cost is not None
    assert corrected_cost == Decimal("0.0057465"), (
        f"Expected Decimal('0.0057465'), got {corrected_cost!r}"
    )

    # The old value under-reports true cost by ~13.5%.
    under_report_fraction = (corrected_cost - old_wrong_cost) / old_wrong_cost
    assert abs(under_report_fraction - Decimal("0.135")) < Decimal("0.001"), (
        f"Expected ~13.5% under-report, got {under_report_fraction * 100:.2f}%"
    )


# ---------------------------------------------------------------------------
# (q) thoughts_token_count defaults to 0 -> identical to omitting it
# ---------------------------------------------------------------------------
def test_thoughts_token_count_defaults_to_zero():
    """Omitting thoughts_token_count must be identical to passing 0 explicitly."""
    omitted = compute_cost(
        "gemini-2.5-flash", prompt_token_count=1_000, candidates_token_count=200
    )
    explicit_zero = compute_cost(
        "gemini-2.5-flash",
        prompt_token_count=1_000,
        candidates_token_count=200,
        thoughts_token_count=0,
    )
    assert omitted == explicit_zero


# ---------------------------------------------------------------------------
# (r) Thinking tokens billed at the SAME rate as candidates (flat-rate model)
# ---------------------------------------------------------------------------
def test_thinking_tokens_billed_at_output_rate_flat_model():
    """1M thinking tokens alone must cost the same as 1M candidate tokens alone."""
    from_candidates = compute_cost(
        "gemini-2.5-flash", prompt_token_count=0, candidates_token_count=1_000_000
    )
    from_thoughts = compute_cost(
        "gemini-2.5-flash",
        prompt_token_count=0,
        candidates_token_count=0,
        thoughts_token_count=1_000_000,
    )
    assert from_candidates == from_thoughts == Decimal("2.50")


# ---------------------------------------------------------------------------
# (s) thoughts_token_count adds on top of candidates, not replaces them
# ---------------------------------------------------------------------------
def test_thinking_and_candidate_tokens_both_billed():
    """candidates + thoughts must both contribute to the output charge."""
    result = compute_cost(
        "gemini-2.5-flash-lite",
        prompt_token_count=0,
        candidates_token_count=500_000,
        thoughts_token_count=500_000,
    )
    # 1M combined output tokens x $0.40/M = $0.40
    assert result == Decimal("0.40"), f"Expected Decimal('0.40'), got {result!r}"


# ---------------------------------------------------------------------------
# Defect D: new-model rates + still-unpriced model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (t) gemini-3.5-flash flat rate: $1.50 / $9.00 / $0.15 per 1M
# ---------------------------------------------------------------------------
def test_gemini_3_5_flash_rates():
    input_cost = compute_cost(
        "gemini-3.5-flash", prompt_token_count=1_000_000, candidates_token_count=0
    )
    output_cost = compute_cost(
        "gemini-3.5-flash", prompt_token_count=0, candidates_token_count=1_000_000
    )
    cache_cost = compute_cost(
        "gemini-3.5-flash",
        prompt_token_count=1_000_000,
        candidates_token_count=0,
        cached_content_token_count=1_000_000,
    )
    assert input_cost == Decimal("1.50")
    assert output_cost == Decimal("9.00")
    assert cache_cost == Decimal("0.15")  # fresh_input=0 -> only cache_read applies


# ---------------------------------------------------------------------------
# (u) gemini-3.6-flash flat rate: $1.50 / $7.50 / $0.15 per 1M
# ---------------------------------------------------------------------------
def test_gemini_3_6_flash_rates():
    input_cost = compute_cost(
        "gemini-3.6-flash", prompt_token_count=1_000_000, candidates_token_count=0
    )
    output_cost = compute_cost(
        "gemini-3.6-flash", prompt_token_count=0, candidates_token_count=1_000_000
    )
    assert input_cost == Decimal("1.50")
    assert output_cost == Decimal("7.50")


# ---------------------------------------------------------------------------
# (v) gemini-3.1-flash-lite flat rate: $0.25 / $1.50 / $0.025 per 1M
# ---------------------------------------------------------------------------
def test_gemini_3_1_flash_lite_rates():
    input_cost = compute_cost(
        "gemini-3.1-flash-lite", prompt_token_count=1_000_000, candidates_token_count=0
    )
    output_cost = compute_cost(
        "gemini-3.1-flash-lite", prompt_token_count=0, candidates_token_count=1_000_000
    )
    assert input_cost == Decimal("0.25")
    assert output_cost == Decimal("1.50")


# ---------------------------------------------------------------------------
# (w) gemini-3.1-flash-image-preview remains unpriced -- no official rate
#     could be confirmed. This must NOT silently become Decimal('0').
# ---------------------------------------------------------------------------
def test_gemini_3_1_flash_image_preview_still_unpriced():
    """Deliberately excluded: no confirmable official Google rate exists.

    Do not add a rate for this model without a citation to
    https://ai.google.dev/gemini-api/docs/pricing (or successor) actually
    listing it -- third-party aggregators disagree with each other and are
    not an authoritative source.
    """
    result = compute_cost(
        "gemini-3.1-flash-image-preview",
        prompt_token_count=1_000,
        candidates_token_count=500,
    )
    assert result is None
    assert result != Decimal(0)
