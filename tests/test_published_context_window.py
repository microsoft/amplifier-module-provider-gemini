"""`get_info()` must publish a context window, and the RIGHT one.

Why this file exists
--------------------
`amplifier-module-context-simple._calculate_budget` sizes a session by reading
`provider.get_info().defaults["context_window"]` and `["max_output_tokens"]`,
in that order of preference, and falls back to a conservative constant when a
provider publishes neither.

This provider published neither. So every Gemini session ran on that fallback
-- 200,000 tokens by default, 300,000 where a bundle configured it -- against
models that actually accept 1,048,576. Roughly 70% of the window went unused,
silently: nothing errors when a budget is too small, it just compacts early.

The window was never unknown. `list_models()` has always read
`input_token_limit` straight off the live API; the number simply never reached
the budget path.

These tests pin the contract at the seam that consumes it: the exact pair of
keys, on the configured model, with limits that are current rather than
generations stale.
"""

from __future__ import annotations

import pytest

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini._capabilities import (
    DEFAULT_LIMITS,
    get_limits,
)

#: Every current Gemini text model, verified 2026-09-15 against
#: https://ai.google.dev/gemini-api/docs/models/<id> ("Token limits").
CURRENT_CONTEXT_WINDOW = 1_048_576
CURRENT_MAX_OUTPUT = 65_536

#: What context-simple falls back to when a provider publishes no window.
#: Named here so the assertion below says WHY the number matters.
CONTEXT_SIMPLE_FALLBACK = 200_000


def _provider(**config) -> GeminiProvider:
    return GeminiProvider(api_key="test-key", config=config)


# ---------------------------------------------------------------------------
# 1. The keys context-simple actually reads.
# ---------------------------------------------------------------------------


def test_get_info_publishes_both_keys_context_simple_reads():
    """Both, or the budget path ignores them: it gates on `if cw and mot`."""
    defaults = _provider().get_info().defaults

    assert defaults["context_window"] == CURRENT_CONTEXT_WINDOW
    assert defaults["max_output_tokens"] == CURRENT_MAX_OUTPUT


def test_published_window_is_far_above_the_fallback_it_replaces():
    """The whole point: stop running a 1M model on a 200K budget."""
    published = _provider().get_info().defaults["context_window"]

    assert published > CONTEXT_SIMPLE_FALLBACK * 5, (
        f"Gemini sessions ran on context-simple's "
        f"{CONTEXT_SIMPLE_FALLBACK:,}-token fallback because this provider "
        f"published no window; it must now publish its real "
        f"{CURRENT_CONTEXT_WINDOW:,}."
    )


# ---------------------------------------------------------------------------
# 2. The CONFIGURED model, not a hardcoded one.
# ---------------------------------------------------------------------------


def test_get_info_reports_the_configured_model_not_a_hardcoded_id():
    """`defaults["model"]` was pinned to gemini-3.7-flash regardless of config."""
    provider = _provider(default_model="gemini-2.5-pro")

    assert provider.get_info().defaults["model"] == "gemini-2.5-pro"


def test_limits_follow_the_configured_model():
    provider = _provider(default_model="gemini-2.5-flash-lite")
    defaults = provider.get_info().defaults

    limits = get_limits("gemini-2.5-flash-lite")
    assert defaults["context_window"] == limits.context_window
    assert defaults["max_output_tokens"] == limits.max_output_tokens


# ---------------------------------------------------------------------------
# 3. The stale 8,192 output cap is gone.
# ---------------------------------------------------------------------------


def test_default_output_cap_is_not_the_stale_8192():
    """8,192 predates every model this provider serves."""
    provider = _provider()

    assert provider.max_tokens == CURRENT_MAX_OUTPUT
    assert provider.max_tokens != 8192


def test_get_info_max_tokens_default_is_not_the_stale_8192():
    assert _provider().get_info().defaults["max_tokens"] == CURRENT_MAX_OUTPUT


def test_explicit_config_still_wins_over_the_model_default():
    """Publishing a model default must not override an operator's choice."""
    assert _provider(max_output_tokens=4_096).max_tokens == 4_096


# ---------------------------------------------------------------------------
# 4. Unknown and suffixed model ids.
# ---------------------------------------------------------------------------


def test_unknown_model_gets_the_documented_default():
    limits = get_limits("gemini-9.9-something-unreleased")

    assert limits == DEFAULT_LIMITS
    assert limits.context_window == CURRENT_CONTEXT_WINDOW


def test_dated_alias_resolves_to_its_base_model():
    """A suffixed id must not silently fall to the unknown-model branch."""
    assert get_limits("gemini-3.7-flash-preview-09-2026") == get_limits(
        "gemini-3.7-flash"
    )


def test_longest_prefix_wins_over_a_shorter_one():
    """`gemini-3.5-flash-lite` must not resolve via `gemini-3.5-flash`."""
    assert get_limits("gemini-3.5-flash-lite") == get_limits("gemini-3.5-flash-lite")
    assert get_limits("gemini-3.5-flash-lite-2026-07") == get_limits(
        "gemini-3.5-flash-lite"
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
)
def test_every_tabled_model_carries_current_limits(model_id: str):
    """Uniform today -- asserted per model so the next divergence breaks a row."""
    limits = get_limits(model_id)

    assert limits.context_window == CURRENT_CONTEXT_WINDOW
    assert limits.max_output_tokens == CURRENT_MAX_OUTPUT
