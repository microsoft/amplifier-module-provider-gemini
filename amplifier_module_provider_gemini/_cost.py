"""Gemini pricing rates and cost computation.

Verification date: 2026-08-12 (thinking-token billing + new-model rates added;
original rates verified 2026-05-06 remain unchanged)
Source: https://ai.google.dev/gemini-api/docs/pricing

Usage
-----
    from amplifier_module_provider_gemini._cost import compute_cost
    from decimal import Decimal

    cost = compute_cost(
        "gemini-2.5-flash",
        prompt_token_count=1_000,
        candidates_token_count=200,
        thoughts_token_count=0,
    )
    # Returns Decimal or None if the model is not recognised.

Thinking tokens are billed at the OUTPUT rate
----------------------------------------------
Google's own pricing page labels every thinking-capable model's output price
"(including thinking tokens)" -- e.g. Gemini 3.6 Flash's "Output price
(including thinking tokens): $7.50". The Gemini API reports thinking tokens
separately from candidate (visible) tokens as
``usage_metadata.thoughts_token_count`` -- it is NOT already folded into
``candidates_token_count`` (confirmed empirically: a measured call showed
``candidates_token_count=4`` and ``thoughts_token_count=228`` as disjoint
counts, and reconciling the reported cost against gemini-3-flash-preview's
published rates only matches when both counts are billed together at the
output rate). ``compute_cost`` therefore bills
``candidates_token_count + thoughts_token_count`` at ``output_per_m``.
This mirrors Anthropic/OpenAI, whose ``output_tokens`` field already
includes reasoning tokens at the source -- Gemini is the outlier in
reporting them as a separate field, not in how they're billed.
"""

from __future__ import annotations

from decimal import Decimal

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PER_M = Decimal(1_000_000)

# _RATES maps model-id → {
#   "input_per_m":             Decimal,   # fresh input tokens, per 1M
#   "output_per_m":            Decimal,   # output tokens, per 1M
#   "cache_read_per_m":        Decimal,   # cache-read input tokens, per 1M
#
#   For tiered models only:
#   "tier_threshold":          int,       # total prompt_token_count boundary
#   "high_input_per_m":        Decimal,   # input rate above threshold
#   "high_output_per_m":       Decimal,   # output rate above threshold
#   "high_cache_read_per_m":   Decimal,   # cache-read rate above threshold
# }
#
# Rates are in USD.
_RATES: dict[str, dict] = {
    # ------------------------------------------------------------------
    # gemini-2.5-flash  (flat rate: $0.30 / $2.50 / $0.03 per 1M)
    # ------------------------------------------------------------------
    "gemini-2.5-flash": {
        "input_per_m": Decimal("0.30"),
        "output_per_m": Decimal("2.50"),
        "cache_read_per_m": Decimal("0.03"),
    },
    # ------------------------------------------------------------------
    # gemini-2.5-flash-lite  (flat rate: $0.10 / $0.40 / $0.01 per 1M)
    # ------------------------------------------------------------------
    "gemini-2.5-flash-lite": {
        "input_per_m": Decimal("0.10"),
        "output_per_m": Decimal("0.40"),
        "cache_read_per_m": Decimal("0.01"),
    },
    # ------------------------------------------------------------------
    # gemini-2.5-pro  (tiered at 200K total prompt tokens)
    #   ≤200K: $1.25 / $10.00 / $0.125 per 1M
    #   >200K: $2.50 / $15.00 / $0.25  per 1M
    # ------------------------------------------------------------------
    "gemini-2.5-pro": {
        "input_per_m": Decimal("1.25"),
        "output_per_m": Decimal("10.00"),
        "cache_read_per_m": Decimal("0.125"),
        "tier_threshold": 200_000,
        "high_input_per_m": Decimal("2.50"),
        "high_output_per_m": Decimal("15.00"),
        "high_cache_read_per_m": Decimal("0.25"),
    },
    # ------------------------------------------------------------------
    # gemini-2.5-pro-preview  (alias — same rates as gemini-2.5-pro)
    # ------------------------------------------------------------------
    "gemini-2.5-pro-preview": {
        "input_per_m": Decimal("1.25"),
        "output_per_m": Decimal("10.00"),
        "cache_read_per_m": Decimal("0.125"),
        "tier_threshold": 200_000,
        "high_input_per_m": Decimal("2.50"),
        "high_output_per_m": Decimal("15.00"),
        "high_cache_read_per_m": Decimal("0.25"),
    },
    # ------------------------------------------------------------------
    # gemini-3-flash-preview  ($0.50 / $3.00 / $0.05 per 1M — flat rate, no tier)
    # Used by routing matrix 'vision' and 'fast' roles (gemini-*-flash-preview glob).
    # ------------------------------------------------------------------
    "gemini-3-flash-preview": {
        "input_per_m": Decimal("0.50"),
        "output_per_m": Decimal("3.00"),
        "cache_read_per_m": Decimal("0.05"),
    },
    # ------------------------------------------------------------------
    # gemini-3-pro-preview  (tiered at 200K total prompt tokens)
    #   ≤200K: $2.00 / $12.00 / $0.20 per 1M
    #   >200K: $4.00 / $18.00 / $0.40 per 1M
    # ------------------------------------------------------------------
    "gemini-3-pro-preview": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("12.00"),
        "cache_read_per_m": Decimal("0.20"),
        "tier_threshold": 200_000,
        "high_input_per_m": Decimal("4.00"),
        "high_output_per_m": Decimal("18.00"),
        "high_cache_read_per_m": Decimal("0.40"),
    },
    # ------------------------------------------------------------------
    # gemini-3.1-pro-preview  (alternate API ID — same rates as 3-pro-preview)
    # ------------------------------------------------------------------
    "gemini-3.1-pro-preview": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("12.00"),
        "cache_read_per_m": Decimal("0.20"),
        "tier_threshold": 200_000,
        "high_input_per_m": Decimal("4.00"),
        "high_output_per_m": Decimal("18.00"),
        "high_cache_read_per_m": Decimal("0.40"),
    },
    # ------------------------------------------------------------------
    # gemini-3.5-flash  (flat rate: $1.50 / $9.00 / $0.15 per 1M)
    # Source: https://ai.google.dev/gemini-api/docs/pricing, "Gemini 3.5
    # Flash" section, Standard tier (verified 2026-08-12; page states
    # "Last updated 2026-08-11 UTC"). Output price is explicitly labelled
    # "(including thinking tokens)" on the page.
    # ------------------------------------------------------------------
    "gemini-3.5-flash": {
        "input_per_m": Decimal("1.50"),
        "output_per_m": Decimal("9.00"),
        "cache_read_per_m": Decimal("0.15"),
    },
    # ------------------------------------------------------------------
    # gemini-3.6-flash  (flat rate: $1.50 / $7.50 / $0.15 per 1M)
    # Source: https://ai.google.dev/gemini-api/docs/pricing, "Gemini 3.6
    # Flash" section, Standard tier (verified 2026-08-12).
    # ------------------------------------------------------------------
    "gemini-3.6-flash": {
        "input_per_m": Decimal("1.50"),
        "output_per_m": Decimal("7.50"),
        "cache_read_per_m": Decimal("0.15"),
    },
    # ------------------------------------------------------------------
    # gemini-3.1-flash-lite  (flat rate: $0.25 / $1.50 / $0.025 per 1M)
    # Source: https://ai.google.dev/gemini-api/docs/pricing, "Gemini 3.1
    # Flash-Lite" section, Standard tier (verified 2026-08-12). The page
    # quotes a higher rate for audio input/caching ($0.50 / $0.05) that
    # this table does not model -- consistent with how gemini-2.5-flash-lite
    # above already only models the text/image/video rate, not audio.
    # ------------------------------------------------------------------
    "gemini-3.1-flash-lite": {
        "input_per_m": Decimal("0.25"),
        "output_per_m": Decimal("1.50"),
        "cache_read_per_m": Decimal("0.025"),
    },
    # ------------------------------------------------------------------
    # gemini-3.1-flash-image-preview -- INTENTIONALLY NOT PRICED.
    #
    # This model is live in the API (confirmed present in production
    # settings.yaml configs) but Google's own pricing page
    # (https://ai.google.dev/gemini-api/docs/pricing, checked 2026-08-12)
    # does not list it at all -- only "Gemini 2.5 Flash Image (Nano Banana)"
    # (gemini-2.5-flash-image) appears there. Third-party aggregators quote
    # mutually-inconsistent numbers for this model ($0.25/$1.50/M tokens vs
    # $0.50/$3.00/M tokens vs an image-token rate implying ~$60/M output
    # tokens), and none of them is Google's own published rate.
    #
    # Per the no-fabrication rule (a wrong rate is worse than a missing
    # one), no entry is added here. compute_cost() returns None for this
    # model as a result -- callers must not treat that as "free"; see
    # GeminiProvider._convert_to_chat_response's handling of cost=None
    # (stamps Usage.cost_unpriced_model and reports it via session.cost /
    # llm:response) which makes the gap visible instead of silent.
    #
    # Add a rate here only once Google publishes an authoritative number
    # for this model on the pricing page above.
    # ------------------------------------------------------------------
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cost(
    model: str,
    *,
    prompt_token_count: int = 0,
    candidates_token_count: int = 0,
    cached_content_token_count: int = 0,
    thoughts_token_count: int = 0,
) -> Decimal | None:
    """Return the USD cost for a Gemini API call as a :class:`~decimal.Decimal`.

    Parameters
    ----------
    model:
        Gemini model identifier (e.g. ``"gemini-2.5-flash"``).
    prompt_token_count:
        TOTAL input tokens (``usage_metadata.prompt_token_count``).
        Includes cached tokens — fresh input is derived by subtraction.
    candidates_token_count:
        Visible output tokens generated (``usage_metadata.candidates_token_count``).
    cached_content_token_count:
        Tokens served from the context cache
        (``usage_metadata.cached_content_token_count``).
    thoughts_token_count:
        Reasoning/thinking tokens generated (``usage_metadata.thoughts_token_count``).
        Google bills these at the SAME rate as ``candidates_token_count`` --
        every thinking-capable model's "Output price" on Google's pricing page
        is explicitly labelled "(including thinking tokens)". Defaults to 0 so
        callers with no thinking activity (or providers/tests that don't pass
        it) are unaffected.

    Returns
    -------
    Decimal | None
        The computed cost in USD, or ``None`` if *model* is not recognised.
        ``None`` is semantically distinct from ``Decimal('0')`` (a free call).
    """
    rates = _RATES.get(model)
    if rates is None:
        return None

    fresh_input = max(0, prompt_token_count - cached_content_token_count)

    # Tier selection uses TOTAL prompt_token_count (including cached)
    if rates.get("tier_threshold") and prompt_token_count > rates["tier_threshold"]:
        input_rate = rates.get("high_input_per_m", rates["input_per_m"])
        output_rate = rates.get("high_output_per_m", rates["output_per_m"])
        cache_read_rate = rates.get("high_cache_read_per_m", rates["cache_read_per_m"])
    else:
        input_rate = rates["input_per_m"]
        output_rate = rates["output_per_m"]
        cache_read_rate = rates["cache_read_per_m"]

    # Billed output = visible (candidate) tokens + thinking tokens. Google does
    # not bill thinking tokens as a separate line item -- both are charged at
    # output_rate. See module docstring for the empirical confirmation.
    billed_output_tokens = candidates_token_count + thoughts_token_count

    cost = Decimal(fresh_input) * input_rate / _PER_M
    cost += Decimal(billed_output_tokens) * output_rate / _PER_M
    if cached_content_token_count:
        cost += Decimal(cached_content_token_count) * cache_read_rate / _PER_M

    return cost
