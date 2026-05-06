"""Gemini pricing rates and cost computation.

Verification date: 2026-05-06
Source: https://ai.google.dev/gemini-api/docs/pricing

Usage
-----
    from amplifier_module_provider_gemini._cost import compute_cost
    from decimal import Decimal

    cost = compute_cost(
        "gemini-2.5-flash",
        prompt_token_count=1_000,
        candidates_token_count=200,
    )
    # Returns Decimal or None if the model is not recognised.
"""

from __future__ import annotations

from decimal import Decimal

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PER_M = Decimal("1_000_000")

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
        Output tokens generated (``usage_metadata.candidates_token_count``).
    cached_content_token_count:
        Tokens served from the context cache
        (``usage_metadata.cached_content_token_count``).

    Returns
    -------
    Decimal | None
        The computed cost in USD, or ``None`` if *model* is not recognised.
        ``None`` is semantically distinct from ``Decimal('0')`` (a free call).
    """
    rates = _RATES.get(model)
    if rates is None:
        return None

    fresh_input = prompt_token_count - cached_content_token_count

    # Tier selection uses TOTAL prompt_token_count (including cached)
    if rates.get("tier_threshold") and prompt_token_count > rates["tier_threshold"]:
        input_rate = rates.get("high_input_per_m", rates["input_per_m"])
        output_rate = rates.get("high_output_per_m", rates["output_per_m"])
        cache_read_rate = rates.get("high_cache_read_per_m", rates["cache_read_per_m"])
    else:
        input_rate = rates["input_per_m"]
        output_rate = rates["output_per_m"]
        cache_read_rate = rates["cache_read_per_m"]

    cost = Decimal(fresh_input) * input_rate / _PER_M
    cost += Decimal(candidates_token_count) * output_rate / _PER_M
    if cached_content_token_count:
        cost += Decimal(cached_content_token_count) * cache_read_rate / _PER_M

    return cost
