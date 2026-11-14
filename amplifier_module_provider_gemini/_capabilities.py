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

The originally qualified Gemini 3 / 2.5 counter models share 1,048,576 input /
65,536 output limits. Public model facts need not share that uniformity or imply
native-count qualification: the exact Robotics ER2 endpoint documents a smaller
input limit. Keep its publication-only fact separate from the counter's model set.

`list_models(filters={})` remains the live source of truth and is not replaced by this
table: it reports whatever the API says for every model on the account,
including ones released after this file was last verified. This table answers
only the synchronous question `get_info()` has to answer without a network
round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ModelLimits",
    "get_limits",
    "has_known_limits",
    "has_native_count_limits",
    "has_exact_model_limits",
    "DEFAULT_LIMITS",
]


@dataclass(frozen=True)
class ModelLimits:
    """Token limits for one model."""

    context_window: int
    max_output_tokens: int


#: What an unrecognized model id gets.
#:
#: Inference-only fallback; model listing uses zero for unknown API limits.
#: Legacy fallback retained for uncatalogued IDs; not authoritative evidence
#: about a future, smaller or unified-window model. Known publication-only
#: exceptions below must not inherit it or gain native-count qualification.
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

# Public model facts do not qualify a native-budget adapter. Exact IDs only:
# do not propagate this fact to the separate Live endpoint or guessed suffixes.
# Audited 2026-11-14, "Model endpoints" / standard ER2 table:
# https://aistudio.google.com/docs/robotics-overview?generate-content=true#model-versions
_PUBLICATION_ONLY_LIMITS = {
    "gemini-robotics-er-2-preview": ModelLimits(131_072, 65_536),
}


def get_limits(model_id: str) -> ModelLimits:
    """Return token limits for *model_id*, falling back to `DEFAULT_LIMITS`.

    Publication-only facts match exact IDs. Counter models match their exact id
    first, then the longest known prefix, so dated
    or suffixed aliases (``gemini-3.7-flash-preview-09-2026``) resolve to their
    base model instead of silently taking the unknown-model default.
    """
    if model_id in _PUBLICATION_ONLY_LIMITS:
        return _PUBLICATION_ONLY_LIMITS[model_id]
    if model_id in _LIMITS:
        return _LIMITS[model_id]

    candidates = [known for known in _LIMITS if model_id.startswith(known)]
    if candidates:
        return _LIMITS[max(candidates, key=len)]

    return DEFAULT_LIMITS


def has_known_limits(model_id: str) -> bool:
    """Whether a model has a documented exact/prefix token-limit entry."""
    return model_id in _PUBLICATION_ONLY_LIMITS or has_native_count_limits(model_id)


def has_native_count_limits(model_id: str) -> bool:
    """Whether the existing counter qualified this model's budget semantics."""
    return model_id in _LIMITS or any(model_id.startswith(known) for known in _LIMITS)


def has_exact_model_limits(model_id: str) -> bool:
    """Whether the literal model ID is documented, without trusting a suffix."""
    return model_id in _LIMITS or model_id in _PUBLICATION_ONLY_LIMITS
