"""Per-model token limits -- the numbers `get_info()` publishes.

Why this file exists
--------------------
`get_info()` is synchronous and must not make a network call, but the context
manager needs a context window from it: `_calculate_budget` in
`amplifier-module-context-simple` reads `get_info().defaults["context_window"]`
and `["max_output_tokens"]`, and falls back to a conservative guess when a
provider publishes neither.

This provider published neither. Gemini sessions therefore ran on that
fallback -- 200,000-300,000 tokens -- against models that actually accept
1,048,576. The window was known (``list_models()`` reads ``input_token_limit``
straight off the live API) but never reached the budget path, so roughly 70% of
every Gemini context window went unused.

Verification date: 2026-09-15
Sources:
  https://ai.google.dev/gemini-api/docs/models        (page updated 2026-09-04)
  https://ai.google.dev/gemini-api/docs/models/<id>   ("Token limits" block)
  https://ai.google.dev/gemini-api/docs/deprecations

Every current Gemini text model reports the SAME limits -- 1,048,576 input /
65,536 output -- so the table below is uniform today. It is still written out
per model rather than collapsed to a constant, because that uniformity is an
observation about this moment, not a property of the API, and the next model
that breaks it should break one row rather than the rule.

`list_models()` remains the live source of truth and is not replaced by this
table: it reports whatever the API says for every model on the account,
including ones released after this file was last verified. This table answers
only the synchronous question `get_info()` has to answer without a network
round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelLimits", "get_limits", "DEFAULT_LIMITS"]


@dataclass(frozen=True)
class ModelLimits:
    """Token limits for one model."""

    context_window: int
    max_output_tokens: int


#: What an unrecognized model id gets.
#:
#: Matches the values `list_models()` already falls back to when the API omits
#: a limit, and matches every current model. An unknown id is far more likely
#: to be a model released after this file was verified -- which would share
#: these limits -- than a smaller legacy one, and the failure direction is
#: benign either way: an over-large context window surfaces as a loud API
#: error on an oversized request, not as silent truncation.
DEFAULT_LIMITS = ModelLimits(context_window=1_048_576, max_output_tokens=65_536)


_LIMITS: dict[str, ModelLimits] = {
    # Gemini 3.x -- current generation
    "gemini-3.8-flash": ModelLimits(1_048_576, 65_536),
    "gemini-3.7-flash": ModelLimits(1_048_576, 65_536),
    "gemini-3.6-flash": ModelLimits(1_048_576, 65_536),
    "gemini-3.5-flash": ModelLimits(1_048_576, 65_536),
    "gemini-3.5-flash-lite": ModelLimits(1_048_576, 65_536),
    "gemini-3.1-flash-lite": ModelLimits(1_048_576, 65_536),
    "gemini-3.1-pro-preview": ModelLimits(1_048_576, 65_536),
    "gemini-3-flash-preview": ModelLimits(1_048_576, 65_536),
    # Gemini 2.5 -- previous generation, still served
    "gemini-2.5-pro": ModelLimits(1_048_576, 65_536),
    "gemini-2.5-flash": ModelLimits(1_048_576, 65_536),
    "gemini-2.5-flash-lite": ModelLimits(1_048_576, 65_536),
}


def get_limits(model_id: str) -> ModelLimits:
    """Return token limits for *model_id*, falling back to `DEFAULT_LIMITS`.

    Matches on the exact id first, then on the longest known prefix, so dated
    or suffixed aliases (``gemini-3.7-flash-preview-09-2026``) resolve to their
    base model instead of silently taking the unknown-model default.
    """
    if model_id in _LIMITS:
        return _LIMITS[model_id]

    candidates = [known for known in _LIMITS if model_id.startswith(known)]
    if candidates:
        return _LIMITS[max(candidates, key=len)]

    return DEFAULT_LIMITS
