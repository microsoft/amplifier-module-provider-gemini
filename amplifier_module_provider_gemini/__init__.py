"""
Gemini provider module for Amplifier.
Integrates with Google's Gemini API.
"""

__all__ = ["mount", "GeminiProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import base64
import difflib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
import json
import logging
import os
import time
from urllib.parse import quote
import uuid
from contextlib import suppress
from typing import Any
from typing import TYPE_CHECKING

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.events import PROVIDER_RETRY
from amplifier_core.llm_errors import AccessDeniedError
from amplifier_core.llm_errors import AuthenticationError
from amplifier_core.llm_errors import ContentFilterError
from amplifier_core.llm_errors import ContextLengthError
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.llm_errors import LLMError
from amplifier_core.llm_errors import LLMTimeoutError
from amplifier_core.llm_errors import NotFoundError
from amplifier_core.llm_errors import ProviderUnavailableError
from amplifier_core.llm_errors import RateLimitError
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from amplifier_core.utils import redact_secrets
from amplifier_core.message_models import ChatRequest
from ._capabilities import DEFAULT_LIMITS
from ._capabilities import get_limits
from ._capabilities import has_exact_model_limits
from ._capabilities import has_known_limits
from ._cost import compute_cost
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import ToolCall
from amplifier_core.message_models import Usage

# google.genai.errors provides the native exception hierarchy for the GenAI SDK.
# Guard the import so the module still loads in unusual environments.
try:
    from google.genai import errors as genai_errors
except ImportError:
    genai_errors = None  # type: ignore[assignment]

# google.api_core.exceptions may be available as a transitive dependency.
# Some environments install it alongside google-genai; others don't.
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from google import genai  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Teardown hard bound (see GeminiProvider.close)
# ---------------------------------------------------------------------------
# `google.genai.Client` holds TWO independent httpx transports, both built
# eagerly in `Client(...)` (verified against google-genai 1.56.0 -- this
# module's dependency floor -- and 2.22.0):
#
#   - a SYNC one, closed by the SYNCHRONOUS `Client.close()`
#     (`self._api_client.close()` -> `httpx.Client.close()`), and
#   - an ASYNC one, closed by `await Client.aio.aclose()`
#     (`self._api_client.aclose()` -> `httpx.AsyncClient.aclose()` plus any
#     aiohttp sessions).
#
# Neither has a deadline of its own, and `httpx.AsyncClient.aclose()` on a
# half-closed (CLOSE-WAIT) connection can block indefinitely. Session cleanup
# runs BEFORE a CLI command returns its result, so an unbounded close turns a
# finished run into a silent hang -- the failure mode measured on the sibling
# providers (recipes-8sr: 28 minutes). Overridable per instance via the
# `close_timeout` config key, matching anthropic/openai/vllm/azure-openai/
# chat-completions/github-copilot.
_DEFAULT_CLOSE_TIMEOUT: float = 5.0


class _UnsupportedCountRequest(ValueError):
    """The finalized request cannot be represented by Developer countTokens."""


# `request_budget()` returns None when a native count cannot be produced. That
# is a correct decision, but on its own it is indistinguishable from "this
# provider has no counter at all" -- the exact ambiguity a live session hit
# (two estimate-based truncations, no successful provider_budget event, and no
# way to tell which precondition failed). This event carries the REASON out,
# without carrying anything sensitive with it.
PROVIDER_REQUEST_BUDGET_UNAVAILABLE = "provider:request_budget_unavailable"

# Fixed, safe reason codes. These are the entire diagnostic vocabulary: no
# exception text, response body, request body, URL, credential, or environment
# value is ever placed in the payload or in the matching log line.
_BUDGET_UNAVAILABLE_UNSUPPORTED_MODEL = "unsupported_model"
_BUDGET_UNAVAILABLE_UNSUPPORTED_ROUTE = "unsupported_route"
_BUDGET_UNAVAILABLE_REQUEST_PLAN = "request_plan_unavailable"
_BUDGET_UNAVAILABLE_REQUEST_PROJECTION = "request_projection_unavailable"
_BUDGET_UNAVAILABLE_INVALID_OUTPUT_LIMIT = "invalid_output_limit"
_BUDGET_UNAVAILABLE_HTTP_ERROR = "http_error"
_BUDGET_UNAVAILABLE_INVALID_RESPONSE = "invalid_response"

def _safe_event_model(model: Any) -> str | None:
    """Publish only an exact documented ID, never caller-controlled suffixes."""
    # Limit lookup intentionally accepts suffixed aliases, but telemetry must
    # not expose unknown strings, even when they start with a known model ID.
    return model if type(model) is str and has_exact_model_limits(model) else None


def _sdk_environment_uses_vertexai() -> bool:
    """Match google-genai 2.23's environment route selection without loading it."""
    enterprise = os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE")
    vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    if enterprise is not None:
        return enterprise.lower() in ("true", "1")
    if vertexai is not None:
        return vertexai.lower() in ("true", "1")
    return False


def _sdk_environment_uses_canonical_developer_route() -> bool:
    """Whether SDK environment defaults select this counter's fixed endpoint."""
    return not (
        os.environ.get("GOOGLE_GEMINI_BASE_URL")
        or _sdk_environment_uses_vertexai()
    )


@dataclass(frozen=True)
class _GeminiRequestPlan:
    """The one provider-owned plan used for generation and native counting."""

    model: str
    contents: list[dict[str, Any]]
    config: Any
    count_unsupported_reason: str | None


# GenerateContentConfig has local controls that do not reach the Developer REST
# request.  The count projection intentionally omits only these known fields.
_COUNT_LOCAL_CONFIG_FIELDS = frozenset(
    {
        "automaticFunctionCalling",
        "httpOptions",
        "shouldReturnHttpResponse",
    }
)
_COUNT_TOP_LEVEL_CONFIG_FIELDS = frozenset(
    {
        "cachedContent",
        "safetySettings",
        "serviceTier",
        "systemInstruction",
        "toolConfig",
        "tools",
    }
)
_COUNT_GENERATION_CONFIG_FIELDS = frozenset(
    {
        "candidateCount",
        "enableEnhancedCivicAnswers",
        "frequencyPenalty",
        "logprobs",
        "maxOutputTokens",
        "mediaResolution",
        "presencePenalty",
        "responseLogprobs",
        "responseMimeType",
        "responseModalities",
        "seed",
        "stopSequences",
        "temperature",
        "thinkingConfig",
        "topK",
        "topP",
    }
)
_COUNT_UNSUPPORTED_CONFIG_FIELDS = frozenset(
    {
        "audioTimestamp",
        "audioTranscriptionConfig",
        "imageConfig",
        "labels",
        "modelArmorConfig",
        "modelSelectionConfig",
        "responseJsonSchema",
        "responseSchema",
        "routingConfig",
        "speechConfig",
    }
)


def _count_system_instruction_content(instruction: Any, types: Any) -> Any:
    """Normalize documented public system-instruction forms to ``Content``.

    ``GenerateContentConfig`` accepts strings and public ``Part``/``Content``
    values for generation.  The REST counter accepts only a content object, so
    preserve a supplied ``Content`` (including its role) and wrap all
    part-level forms with the SDK's default ``user`` role. Other values require SDK
    normalization beyond this bounded public projection.
    """
    try:
        if isinstance(instruction, types.Content):
            return instruction
        if isinstance(instruction, types.Part):
            return types.Content(role="user", parts=[instruction])
        if isinstance(instruction, str):
            return types.Content(role="user", parts=[types.Part(text=instruction)])
        if isinstance(instruction, list):
            parts = [
                part
                if isinstance(part, types.Part)
                else types.Part(text=part)
                if isinstance(part, str)
                else types.Part.model_validate(part)
                for part in instruction
            ]
            return types.Content(role="user", parts=parts)
        if isinstance(instruction, Mapping):
            if "parts" in instruction:
                return types.Content.model_validate(instruction)
            return types.Content(
                role="user", parts=[types.Part.model_validate(instruction)]
            )
    except (TypeError, ValueError):
        pass
    raise _UnsupportedCountRequest(
        "Gemini system instruction cannot be publicly serialized for countTokens"
    )


def _match_sdk_generation_wire(serialized_config: dict[str, Any]) -> None:
    """Apply the two SDK 2.23.0 nested wire spellings to a typed config dump.

    The public typed aliases are not universally the aliases emitted by the
    SDK's ``generate_content`` request transformer.  Keep this deliberately
    narrow: broad recursive alias rewriting would hide new unsupported fields
    instead of making counting unavailable.
    """
    tools = serialized_config.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            declarations = tool.get("functionDeclarations")
            if not isinstance(declarations, list):
                continue
            for declaration in declarations:
                if (
                    isinstance(declaration, dict)
                    and "parametersJsonSchema" in declaration
                ):
                    declaration["parameters_json_schema"] = declaration.pop(
                        "parametersJsonSchema"
                    )

    thinking_config = serialized_config.get("thinkingConfig")
    if isinstance(thinking_config, dict):
        for public_alias, generation_key in (
            ("includeThoughts", "include_thoughts"),
            ("thinkingLevel", "thinking_level"),
        ):
            if public_alias in thinking_config:
                thinking_config[generation_key] = thinking_config.pop(public_alias)


def _retrieve_task_exception(task: "asyncio.Future[Any]") -> None:
    """Consume an abandoned close task's exception.

    Without this, a close task we stopped awaiting that later fails makes
    asyncio log "Task exception was never retrieved" at GC time -- noise
    that reads like a new defect. Cancelled tasks have nothing to retrieve.
    """
    if not task.cancelled():
        task.exception()


# ---------------------------------------------------------------------------
# Process-wide concurrency semaphore
# Shared across ALL GeminiProvider instances in this process (including
# parent + delegated child sessions). Prevents simultaneous-delegation
# blast patterns from exhausting Gemini API rate limits.
# Created lazily on the first API call; keyed by event loop so that tests
# using asyncio.run() get fresh semaphores rather than inheriting stale state.
# ---------------------------------------------------------------------------

_process_semaphore: asyncio.Semaphore | None = None
_process_semaphore_loop: Any = None  # asyncio.AbstractEventLoop
_process_semaphore_max: int = 0
_active_requests: int = 0  # currently holding semaphore (executing)
_waiting_requests: int = 0  # waiting to acquire semaphore


async def _get_process_semaphore(max_concurrent: int) -> asyncio.Semaphore | None:
    """Get or create the process-wide concurrency semaphore.

    Returns ``None`` when ``max_concurrent <= 0`` (semaphore disabled).
    Recreates the semaphore when called from a different event loop so that
    unit tests using ``asyncio.run()`` always get a fresh, valid semaphore.
    """
    global _process_semaphore, _process_semaphore_loop, _process_semaphore_max
    if max_concurrent <= 0:
        return None
    current_loop = asyncio.get_running_loop()
    if (
        _process_semaphore is None
        or _process_semaphore_loop is not current_loop
        or _process_semaphore_max != max_concurrent
    ):
        _process_semaphore = asyncio.Semaphore(max_concurrent)
        _process_semaphore_loop = current_loop
        _process_semaphore_max = max_concurrent
    return _process_semaphore


_CLOUDFLARE_403_WARNING = (
    "[PROVIDER] Cloudflare challenge detected (HTTP 403 "
    "with no details). Treating as transient — will retry."
)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Gemini provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including API key

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get API key from config or environment
    # Per Google GenAI SDK: supports both GEMINI_API_KEY and GOOGLE_API_KEY
    # If both are set, GOOGLE_API_KEY takes precedence
    api_key = config.get("api_key")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logger.warning(
            "No API key found for Gemini provider (set GOOGLE_API_KEY or GEMINI_API_KEY)"
        )
        return None

    _totals: dict = {
        "cost_usd": None,
        "has_data": False,
        "unpriced_models": set(),
    }

    def _add_cost(cost, model: str | None = None) -> None:
        if cost is not None:
            _totals["cost_usd"] = (_totals["cost_usd"] or Decimal("0")) + cost
            _totals["has_data"] = True
        elif model is not None:
            # A real API call happened but this model has no rate-table entry.
            # Track it so the session.cost contributor below can surface
            # "cost incomplete: unpriced model(s)" instead of silently
            # reporting nothing at all when this is the only kind of call
            # made in the session (the exact "reports no cost at all for
            # every turn" defect).
            _totals["unpriced_models"].add(model)
            _totals["has_data"] = True

    provider = GeminiProvider(api_key, config, coordinator, add_cost=_add_cost)
    await coordinator.mount("providers", provider, name="gemini")
    logger.info("Mounted GeminiProvider")

    # M2 compaction mitigation: intentionally NOT ported from provider-openai.
    #
    # amplifier-module-provider-openai subscribes to "context:compaction" /
    # "context:pre_compact" / "context:post_compact" and sets a one-shot flag
    # that drops `previous_response_id` on the next request. That mitigation
    # exists because the OpenAI Responses API supports server-side response
    # chaining: the provider persists `previous_response_id` across turns and,
    # when chaining, sends only a *delta* of new messages plus that ID -- the
    # server reconstructs the rest of the context from its own stored state.
    # If compaction shrinks the local message list but the provider keeps
    # chaining from a pre-compaction response ID, the server-side context
    # never shrinks: input tokens grow unboundedly until
    # context_length_exceeded, and cache alignment is irrelevant because the
    # compacted prefix was never actually sent to the server at all.
    #
    # GeminiProvider has no equivalent state to invalidate:
    #   - It holds no per-turn handle analogous to previous_response_id (no
    #     `self._last_response_id` / chain / session reference is stored
    #     across calls -- verified by inspection of GeminiProvider.__init__
    #     and complete()).
    #   - Every request rebuilds `contents` from scratch from
    #     `request.messages` (see the conversion in complete(), around the
    #     `all_messages = context_user_msgs + conversation_msgs` assembly).
    #     Whatever the context manager returns -- pre- or post-compaction --
    #     is exactly what gets sent. There is no delta-plus-serverside-chain
    #     shortcut for compaction to be defeated by.
    #   - This provider does not use Gemini's *explicit* caching API
    #     (`client.caches.create` / `CachedContent` / `cached_content=` on the
    #     request config -- confirmed absent from this module by grep). It
    #     relies solely on Gemini's *implicit* caching, which is automatic,
    #     server-side, and keyed off the request's own content each call --
    #     there is no client-held cache handle to reset either.
    #
    # Net effect: compaction already takes effect on the very next Gemini
    # request, with no unbounded growth and no stale server-side reference to
    # break. The only unavoidable consequence of a changed prefix is the
    # ordinary one-time implicit-cache miss on the first post-compaction
    # request while the new (shorter) prefix populates the cache again --
    # that is a property of context-simple's prefix mutation itself, not
    # something a provider-side compaction hook could fix, and it affects
    # every provider with prefix-keyed caching equally (Anthropic, Gemini,
    # and OpenAI's own implicit prompt-cache keying) rather than being
    # specific to Gemini.
    #
    # If Gemini ever gains client-managed state that persists across turns
    # (e.g. adopting explicit `CachedContent` handles), this reasoning must be
    # revisited -- see tests/test_no_compaction_state.py for the regression
    # guard that documents this decision and will need updating alongside it.

    # Register observability events via contribution channels
    coordinator.register_contributor(
        "observability.events",
        "provider-gemini",
        lambda: [
            "llm:request",
            "llm:response",
            "provider:concurrency",
            "provider:request_budget_unavailable",
            "provider:tool_sequence_repaired",
            "thinking:final",
            "llm:stream_block_start",
            "llm:stream_block_delta",
            "llm:stream_block_end",
            "llm:stream_aborted",
        ],
    )
    def _report_session_cost() -> dict[str, Any] | None:
        """Report accumulated session cost for the "session.cost" channel.

        Returns None only when no Gemini calls have happened at all (the
        pre-existing "no data" case). Once ANY call has happened -- priced
        or not -- this always returns a dict, so a session that only ever
        called an unpriced model reports "cost_usd: None, unpriced_models:
        [...]" rather than silently vanishing and reading as "$0, all free".
        """
        if not _totals["has_data"]:
            return None
        result: dict[str, Any] = {
            "cost_usd": str(_totals["cost_usd"])
            if _totals["cost_usd"] is not None
            else None
        }
        if _totals["unpriced_models"]:
            result["unpriced_models"] = sorted(_totals["unpriced_models"])
        return result

    coordinator.register_contributor(
        "session.cost",
        "provider-gemini",
        _report_session_cost,
    )

    # Return cleanup function that delegates to provider.close().
    # close() handles the lazy-client guard, the shield, CancelledError, and
    # the hard `close_timeout` bound -- this cleanup runs inside the finally
    # that precedes a CLI command's return, so it must never block forever.
    async def cleanup():
        await provider.close()

    return cleanup


def _encode_sig(sig: bytes | str | None) -> str | None:
    """Encode a Gemini thought_signature for the wire format.

    Gemini 2.5+ attaches an opaque ``thought_signature`` to parts that follow
    a thinking burst.  The SDK returns raw bytes; the REST API accepts a
    base64-encoded ASCII string.  Pre-encoded strings (e.g. round-tripped from
    a ThinkingBlock) are returned as-is.

    Args:
        sig: Raw bytes from the SDK, an already-encoded str, or None.

    Returns:
        Base64-encoded ASCII string, or None if sig is falsy.
    """
    if not sig:
        return None
    if isinstance(sig, bytes):
        return base64.b64encode(sig).decode("ascii")
    return sig  # assume already base64-encoded str


# ---------------------------------------------------------------------------
# Thinking level support (google-genai's *current* thinking control)
# ---------------------------------------------------------------------------
# Google's `thinking_budget` (an approximate output-token budget spent on
# internal reasoning) is now the LEGACY thinking control. The current
# control -- and the *only* one some models accept at all -- is
# `thinking_level`, an enum (minimal|low|medium|high). Sending both
# thinking_level and thinking_budget on the same request is rejected by the
# API with a 400.
#
# CRITICAL, verified LIVE against the real Google AI API on 2026-08-29 (not
# documented by Google as of this writing, and NOT the "smaller subset of
# levels per older model" story one might assume): thinking_level support is
# an all-or-nothing split by model generation, not a graduated subset --
#   * gemini-2.5-flash and gemini-2.5-pro REJECT thinking_level outright:
#       400 INVALID_ARGUMENT: "Thinking level is not supported for this model."
#     These models only understand the legacy thinking_budget control, and
#     think by default via a dynamic budget regardless of any config.
#   * gemini-3.x models are the opposite: thinking is MANDATORY and
#     thinking_budget is silently ignored (verified live: thinking_budget=0
#     on gemini-3.7-flash still produced ~26 thinking tokens) -- there is no
#     way to disable thinking on a Gemini 3.x model. thinking_level is their
#     only real, effective control, and even that control cannot express
#     "disabled" -- there is no "none" level in the enum.
#
# _THINKING_LEVEL_TABLE maps a model id to the ThinkingLevel values it is
# known to accept. `None` means "rejects thinking_level entirely -- always
# use the legacy thinking_budget path for this model". Google does not
# publish this table; it is reverse-engineered from live 400 responses.
# Treat it as best-effort and update it as new models ship or vendor
# behavior changes.
_THINKING_LEVEL_TABLE: dict[str, tuple[str, ...] | None] = {
    # Gemini 2.x -- thinking_level rejected outright (verified live,
    # 2026-08-29). These models think by default via a dynamic budget;
    # the legacy thinking_budget path is their only control.
    "gemini-2.5-pro": None,
    "gemini-2.5-flash": None,
    "gemini-2.5-flash-lite": None,
    "gemini-2.0-flash": None,
    "gemini-2.0-flash-lite": None,
    # Gemini 3.7 Flash -- current flagship Flash model. Verified live:
    # low/medium/high accepted; MINIMAL rejected ("Thinking level MINIMAL is
    # not supported for this model. Please retry with other thinking
    # level."). ai.google.dev documents its default (when omitted) as medium.
    "gemini-3.7-flash": ("low", "medium", "high"),
    # Gemini 3.5 family -- verified live: minimal accepted.
    "gemini-3.5-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash-lite": ("minimal", "low", "medium", "high"),
}

# Fallback range for any gemini-3.x model id not listed above -- new preview
# ids ship often (gemini-3.1-*, gemini-3.6-flash, etc.). Assume the full
# range until a live 400 proves a narrower one for that specific id.
_THINKING_LEVEL_DEFAULT_3X: tuple[str, ...] = ("minimal", "low", "medium", "high")

# Ordinal order used for clamping (lowest to highest amount of thinking).
_THINKING_LEVEL_ORDER: tuple[str, ...] = ("minimal", "low", "medium", "high")

# reasoning_effort (Amplifier's portable, cross-provider knob) -> the
# thinking_level it targets. "xhigh"/"max" collapse to "high" -- Gemini has
# no level above high.
_EFFORT_TO_LEVEL: dict[str, str] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _supported_thinking_levels(model: str) -> tuple[str, ...] | None:
    """Return the ThinkingLevel values ``model`` is known to accept.

    ``None`` means the model rejects thinking_level entirely (legacy
    thinking_budget path only). See _THINKING_LEVEL_TABLE for the live
    evidence behind each entry.
    """
    if model in _THINKING_LEVEL_TABLE:
        return _THINKING_LEVEL_TABLE[model]
    if model.startswith("gemini-2."):
        return None
    if model.startswith("gemini-3."):
        return _THINKING_LEVEL_DEFAULT_3X
    # Unknown family (a future gemini-4.x, a tuned model id, etc.) -- assume
    # support; a live 400 surfaces clearly rather than degrading silently.
    return _THINKING_LEVEL_DEFAULT_3X


def _clamp_thinking_level(model: str, requested: str, supported: tuple[str, ...]) -> str:
    """Clamp ``requested`` to the nearest level in ``supported`` for ``model``.

    Prefers the next level UP (more thinking) over down: under-thinking
    silently degrades output quality, while over-thinking only costs a few
    more tokens. Logs one INFO line whenever the clamp actually changes the
    requested value -- clamping is never silent.
    """
    if requested in supported:
        return requested
    req_idx = _THINKING_LEVEL_ORDER.index(requested)
    for idx in range(req_idx + 1, len(_THINKING_LEVEL_ORDER)):
        if _THINKING_LEVEL_ORDER[idx] in supported:
            clamped = _THINKING_LEVEL_ORDER[idx]
            logger.info(
                "[PROVIDER] Gemini: thinking_level '%s' not supported by '%s' "
                "(supports: %s) -- clamped up to '%s'",
                requested,
                model,
                supported,
                clamped,
            )
            return clamped
    for idx in range(req_idx - 1, -1, -1):
        if _THINKING_LEVEL_ORDER[idx] in supported:
            clamped = _THINKING_LEVEL_ORDER[idx]
            logger.info(
                "[PROVIDER] Gemini: thinking_level '%s' not supported by '%s' "
                "(supports: %s) -- clamped down to '%s'",
                requested,
                model,
                supported,
                clamped,
            )
            return clamped
    return supported[0]  # pragma: no cover -- defensive; supported is never empty


# ---------------------------------------------------------------------------
# Config hygiene: bool/numeric coercion, unknown-key sweep, inert-key notes
# ---------------------------------------------------------------------------
# Config commonly arrives as strings: the app-cli wizard writes
# `field_type="boolean"` values as the literal strings "true"/"false" (not
# Python bools), and hand-edited YAML often quotes both booleans and
# numbers. Naive `bool(raw)` or bare `int(raw)`/`float(raw)` are both wrong
# for that -- `bool("false")` is True (any non-empty string is truthy),
# silently inverting the operator's intent, and a bad numeric string
# currently either raises at mount time (max_retries etc., which call
# int()/float() directly) or survives uncoerced all the way to
# asyncio.wait_for(timeout=...), which fails on the FIRST real API call
# with a confusing low-level TypeError instead of a clear config error.

_CONFIG_BOOL_TRUE_STRINGS: frozenset[str] = frozenset({"true", "1", "yes"})
_CONFIG_BOOL_FALSE_STRINGS: frozenset[str] = frozenset({"false", "0", "no"})


def _parse_config_bool(key: str, raw: Any, default: bool) -> bool:
    """Parse a boolean-ish provider-config value, tolerating string bools.

    Accepts:
      - key absent / value None / value "" -> `default`
      - real bool -> itself, unchanged
      - str in {"true", "1", "yes"} (case-insensitive, stripped) -> True
      - str in {"false", "0", "no"} (case-insensitive, stripped) -> False

    Anything else logs a warning and falls back to `default` -- config
    hygiene here is warn-and-default, not raise, so one typo'd flag doesn't
    take down the whole provider mount.
    """
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in _CONFIG_BOOL_TRUE_STRINGS:
            return True
        if normalized in _CONFIG_BOOL_FALSE_STRINGS:
            return False
    logger.warning(
        "[PROVIDER] Gemini: invalid config %r=%r (expected true/false, "
        "also 1/0, yes/no; case-insensitive) -- using default %r",
        key,
        raw,
        default,
    )
    return default


def _parse_config_number(key: str, raw: Any, default: Any, cast) -> Any:
    """Parse a numeric provider-config value, tolerating string numbers.

    Accepts real int/float values and numeric strings (whitespace-stripped),
    coercing via `cast` (int or float). Anything unparseable -- including
    key absent / None / "" -- logs a warning and falls back to `default`.
    Never raises: a bad numeric config value degrades to a safe default
    instead of crashing the provider at mount time or, worse, surviving
    uncoerced as a string into a low-level call (e.g.
    asyncio.wait_for(timeout="600")) that fails confusingly on the first
    real request instead of at mount.
    """
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        pass  # bool is a subclass of int -- never accept it as numeric config
    elif isinstance(raw, (int, float)):
        try:
            return cast(raw)
        except (TypeError, ValueError):
            pass
    elif isinstance(raw, str):
        try:
            return cast(raw.strip())
        except (TypeError, ValueError):
            pass
    logger.warning(
        "[PROVIDER] Gemini: invalid config %r=%r (expected a %s) -- using "
        "default %r",
        key,
        raw,
        cast.__name__,
        default,
    )
    return default


def _read_renamed_config(config: dict[str, Any], new: str, old: str) -> Any:
    """Read `new`, falling back to the deprecated `old` with one warning.

    The new key always wins when present (even when both are set) -- a
    config that has already been migrated to the new name is never
    silently overridden by a stale leftover of the old one. The warning
    fires only when `old` is the value actually used, so a fully migrated
    config stays silent and a config carrying both is told plainly which
    one won.

    Returns None (not a sentinel) when neither key is present, so callers
    can pass the result straight into _parse_config_bool/_parse_config_number,
    which already treat None as "use my own default".
    """
    new_val = config.get(new)
    old_val = config.get(old)
    if new_val not in (None, ""):
        if old_val not in (None, ""):
            logger.warning(
                "[PROVIDER] Gemini: config keys '%s' (deprecated) and '%s' "
                "are BOTH set; '%s' wins. Remove '%s'.",
                old,
                new,
                new,
                old,
            )
        return new_val
    if old_val not in (None, ""):
        logger.warning(
            "[PROVIDER] Gemini: config key '%s' is deprecated -- use '%s' "
            "instead (this is Google's own API parameter name). Falling "
            "back to '%s'=%r for this session.",
            old,
            new,
            old,
            old_val,
        )
        return old_val
    return None


# Config keys this provider actually reads (self.config.get(...) call
# sites). `priority` is included even though this module only stores it
# on self.priority for the orchestrator's provider-selection logic to read
# -- it is a real, consumed key, never a typo to flag. `max_tokens` is kept
# as the deprecated back-compat alias for `max_output_tokens` (Google's own
# API parameter name) -- see _read_renamed_config. `extra_request_params`
# is a settings-only escape hatch (never a ConfigField / interactive
# wizard prompt) -- see _apply_extra_request_params.
_CONSUMED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "default_model",
        "max_output_tokens",
        "max_tokens",  # deprecated alias for max_output_tokens
        "temperature",
        "timeout",
        "close_timeout",
        "priority",
        "raw",
        "use_streaming",
        "max_retries",
        "min_retry_delay",
        "max_retry_delay",
        "retry_jitter",
        "max_concurrent_requests",
        "extra_request_params",
    }
)


def _merge_thinking_config(existing: Any, override: Any):
    """Compose an explicitly configured ThinkingConfig onto an existing one.

    Pydantic assignment validation is disabled on GenerateContentConfig, so a
    direct assignment of a mapping leaves a raw dict in ``thinking_config``.
    Normalize both values and rebuild the SDK type instead. Only fields the
    caller supplied participate in the merge: an empty mapping or
    ThinkingConfig does not replace computed defaults.
    """
    from google import genai

    thinking_config_type = genai.types.ThinkingConfig
    if isinstance(override, thinking_config_type):
        supplied = override
    elif isinstance(override, dict):
        supplied = thinking_config_type(**override)
    else:
        raise ValueError(
            "extra_request_params.thinking_config must be a mapping, "
            "ThinkingConfig, or null"
        )

    supplied_values = supplied.model_dump(exclude_unset=True)
    has_budget = supplied_values.get("thinking_budget") is not None
    has_level = supplied_values.get("thinking_level") is not None
    if has_budget and has_level:
        raise ValueError(
            "extra_request_params.thinking_config cannot set both "
            "thinking_budget and thinking_level"
        )

    if existing is None:
        base = thinking_config_type()
    elif isinstance(existing, thinking_config_type):
        base = existing
    elif isinstance(existing, dict):
        base = thinking_config_type(**existing)
    else:
        base = thinking_config_type.model_validate(existing)

    existing_values = base.model_dump(exclude_unset=True)
    merged_values = existing_values.copy()
    merged_values.update(supplied_values)
    if has_budget and "thinking_level" in merged_values:
        merged_values["thinking_level"] = None
    elif has_level and "thinking_budget" in merged_values:
        merged_values["thinking_budget"] = None

    merged = thinking_config_type(**merged_values)
    overwritten = [
        field
        for field, value in merged_values.items()
        if existing_values.get(field) is not None and existing_values[field] != value
    ]
    if overwritten:
        logger.warning(
            "[PROVIDER] Gemini: extra_request_params overrides thinking_config "
            "field%s %s -- extra_request_params always wins.",
            "s" if len(overwritten) > 1 else "",
            ", ".join(overwritten),
        )

    return merged


def _apply_extra_request_params(
    config, extra_request_params: dict[str, Any], *, warn: bool = True
) -> None:
    """Merge extra_request_params into a GenerateContentConfig, in place.

    `extra_request_params` is a settings-only escape hatch (bundle/settings
    YAML only -- never an interactive ConfigField) for reaching
    GenerateContentConfig fields this provider doesn't otherwise expose:
    safety_settings, top_p, top_k, seed, stop_sequences,
    presence_penalty/frequency_penalty, response_mime_type, labels, and any
    other field google-genai's GenerateContentConfig defines. It is merged
    LAST, after this provider's own computed values (temperature,
    max_output_tokens, thinking_config, tools, ...) -- the caller's extra
    config always wins, and wins LOUDLY: overriding a value this provider
    itself had already set logs a warning naming the field, the old value,
    and the new one, so a confusing production override is never silent.

    ``thinking_config`` is the one field-level exception. Mappings and SDK
    ThinkingConfig instances compose only their explicitly supplied fields
    onto the provider's computed typed config. This keeps independent defaults
    (such as include_thoughts) and ensures the SDK serializes a typed nested
    value rather than a raw mapping.

    An extra_request_params key that isn't a real GenerateContentConfig
    field logs a warning and is skipped -- never raises, since a typo in
    settings.yaml shouldn't crash the whole provider mount.
    """
    if not extra_request_params:
        return
    valid_fields = type(config).model_fields
    # Reject an invalid nested override before applying any of the other
    # extras, regardless of their insertion order.
    merged_thinking_config = None
    if (
        "thinking_config" in valid_fields
        and extra_request_params.get("thinking_config") is not None
    ):
        merged_thinking_config = _merge_thinking_config(
            getattr(config, "thinking_config", None),
            extra_request_params["thinking_config"],
        )
    for key, value in extra_request_params.items():
        if key not in valid_fields:
            if warn:
                logger.warning(
                    "[PROVIDER] Gemini: extra_request_params key %r is not a "
                    "recognized GenerateContentConfig field -- ignored. See "
                    "google.genai.types.GenerateContentConfig for valid fields.",
                    key,
                )
            continue
        if key == "thinking_config" and value is not None:
            config.thinking_config = merged_thinking_config
            continue
        existing = getattr(config, key, None)
        if existing is not None and warn:
            logger.warning(
                "[PROVIDER] Gemini: extra_request_params overrides '%s' "
                "(provider computed %r, extra_request_params sets %r) -- "
                "extra_request_params always wins.",
                key,
                existing,
                value,
            )
        setattr(config, key, value)


# Keys that appeared in past README revisions describing features that were
# never actually implemented in this module (verified by grep: no
# `self.config.get(...)` call site reads any of them). These are not typos
# -- they're documentation ghosts a user may reasonably still have in their
# config from an older guide -- so they get a specific, helpful message
# instead of a generic "did you mean" guess.
_KNOWN_INERT_CONFIG_KEYS: dict[str, str] = {
    "debug": (
        "documented in older README revisions but never implemented -- no "
        "llm:request:debug/llm:response:debug events exist in this "
        "provider. Setting it has no effect."
    ),
    "raw_debug": (
        "documented in older README revisions but never implemented -- no "
        "llm:request:raw/llm:response:raw events exist in this provider. "
        "Setting it has no effect. (This provider's actual raw-I/O capture "
        "is the differently-named 'raw' config key, which IS implemented.)"
    ),
    "debug_truncate_length": (
        "documented in older README revisions but never implemented -- "
        "this provider has no debug-log truncation path. Setting it has "
        "no effect."
    ),
}


def _sweep_unknown_config_keys(config: dict[str, Any]) -> None:
    """Warn (never raise) about config keys this provider doesn't consume.

    Three distinct messages, in priority order:
      1. A known documentation ghost (_KNOWN_INERT_CONFIG_KEYS) -- specific,
         helpful explanation of why it does nothing.
      2. A likely typo of a real key (difflib match) -- "did you mean X?".
      3. Anything else -- generic "unrecognized, ignored" with the full
         list of keys this provider actually reads.
    """
    for key in config:
        if key in _CONSUMED_CONFIG_KEYS:
            continue
        if key in _KNOWN_INERT_CONFIG_KEYS:
            logger.warning(
                "[PROVIDER] Gemini: config key %r is inert -- %s",
                key,
                _KNOWN_INERT_CONFIG_KEYS[key],
            )
            continue
        suggestions = difflib.get_close_matches(
            key, _CONSUMED_CONFIG_KEYS, n=1
        )
        if suggestions:
            logger.warning(
                "[PROVIDER] Gemini: unknown config key %r -- did you mean "
                "%r? (unrecognized keys are ignored)",
                key,
                suggestions[0],
            )
        else:
            logger.warning(
                "[PROVIDER] Gemini: unknown config key %r -- ignored. "
                "Recognized keys: %s",
                key,
                sorted(_CONSUMED_CONFIG_KEYS),
            )


class GeminiChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


class GeminiProvider:
    """Google Gemini API integration."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        add_cost: Callable[[Decimal | None, str | None], None] | None = None,
    ):
        """
        Initialize Gemini provider.

        The SDK client is created lazily on first use, allowing get_info()
        to work without valid credentials.

        Args:
            api_key: Google AI API key (can be None for get_info() calls)
            config: Additional configuration
            coordinator: Module coordinator for event emission
            add_cost: Optional callback to accumulate cost_usd into a session
                total. Called as ``add_cost(cost, model)`` -- ``model`` lets
                the callback distinguish "no cost data for this call" (cost is
                None, model given) from "no calls happened" (never called).
        """
        self._api_key = api_key
        self._client = None  # Lazy init
        # ``None`` means an injected client whose base route cannot be
        # established through public SDK API. Such clients fail closed for the
        # fixed Developer counter; generation keeps using the caller's client.
        self._client_uses_canonical_developer_route: bool | None = None
        self._count_route_client: Any = None
        self._add_cost = (
            add_cost if add_cost is not None else lambda cost, model=None: None
        )
        self.config = config or {}
        self.coordinator = coordinator
        # gemini-3.7-flash is the current flagship Flash model (verified
        # live against this account's key on 2026-08-29: present in
        # list_models(), 40 gemini-* models served). gemini-2.5-flash is two
        # generations back; gemini-3.5-flash is documented as legacy.
        _sweep_unknown_config_keys(self.config)

        self.default_model = self.config.get("default_model", "gemini-3.7-flash")
        self.max_tokens = _parse_config_number(
            "max_output_tokens",
            _read_renamed_config(self.config, "max_output_tokens", "max_tokens"),
            get_limits(self.default_model).max_output_tokens,
            int,
        )
        self.temperature = _parse_config_number(
            "temperature", self.config.get("temperature"), 0.7, float
        )
        self.timeout = _parse_config_number(
            "timeout", self.config.get("timeout"), 600.0, float
        )
        # Hard bound on session-teardown client close -- see close().
        self._close_timeout = _parse_config_number(
            "close_timeout",
            self.config.get("close_timeout"),
            _DEFAULT_CLOSE_TIMEOUT,
            float,
        )
        self.priority = _parse_config_number(
            "priority", self.config.get("priority"), 100, int
        )
        self.raw = _parse_config_bool("raw", self.config.get("raw"), False)
        self.use_streaming = _parse_config_bool(
            "use_streaming", self.config.get("use_streaming"), True
        )
        # Settings-only escape hatch -- deliberately NOT a ConfigField (no
        # interactive wizard prompt). Arbitrary GenerateContentConfig
        # fields (safety_settings, top_p, top_k, seed, stop_sequences,
        # etc.) merged in last, after this provider's own computed values.
        # See _apply_extra_request_params for the merge contract.
        self.extra_request_params = self.config.get("extra_request_params") or {}

        # Retry configuration — delegates to shared retry_with_backoff() from amplifier-core.
        self._retry_config = RetryConfig(
            max_retries=_parse_config_number(
                "max_retries", self.config.get("max_retries"), 5, int
            ),
            initial_delay=_parse_config_number(
                "min_retry_delay", self.config.get("min_retry_delay"), 1.0, float
            ),
            max_delay=_parse_config_number(
                "max_retry_delay", self.config.get("max_retry_delay"), 60.0, float
            ),
            jitter=_parse_config_bool(
                "retry_jitter", self.config.get("retry_jitter"), True
            ),
        )

        # Process-wide concurrency gate.
        # Limits how many API calls this process has in-flight simultaneously,
        # shared across ALL provider instances (parent + delegated child sessions).
        # This prevents blast patterns (e.g. parallel: true recipes spawning many
        # concurrent calls) from exhausting Gemini API rate limits.
        # Set to 0 to disable the semaphore entirely.
        self._max_concurrent_requests = int(
            self.config.get("max_concurrent_requests", 5)
        )

        # Track repaired tool call IDs to prevent infinite detection loops.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

    @staticmethod
    def _is_cloudflare_challenge(error) -> bool:
        """Detect CDN/proxy challenge responses for Gemini.

        When a CDN/proxy returns a 403, the google-genai SDK wraps it as a
        ClientError with details=None (no JSON body to parse).  Real Gemini
        API 403s always include structured details with an 'error' key.

        Note: this method is used for the ClientError (``genai_errors``) path
        only.  The fallback ``PermissionDenied`` path uses a broader falsy
        check (``not getattr(e, 'details', None)``) because google.api_core
        exceptions may carry empty details (``[]``, ``""``, etc.).
        """
        details = getattr(error, "details", None)
        # No details at all → likely CDN/proxy response
        if details is None:
            return True
        # Dict without 'error' key → CDN wrapped as dict
        if isinstance(details, dict) and "error" not in details:
            return True
        return False

    @property
    def client(self):
        """Lazily initialize the Gemini client on first access."""
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key must be provided for API calls")
            from google import genai

            # The SDK selects its route at construction from these environment
            # defaults. Keep the selection with this client so a later
            # environment change cannot make the fixed Developer counter
            # describe an already noncanonical generation client.
            route_is_canonical = _sdk_environment_uses_canonical_developer_route()
            self._client = genai.Client(api_key=self._api_key)
            self._client_uses_canonical_developer_route = route_is_canonical
            self._count_route_client = self._client
        return self._client

    def _native_counting_unavailable_reason(self, model: Any) -> str | None:
        """Return the safe reason code blocking native counting, or ``None``.

        Splits the single eligibility predicate into the two distinguishable
        causes a reader of the diagnostic actually needs: the MODEL is not one
        this counter can count, or the ROUTE the SDK would generate on is not
        the fixed Developer endpoint this counter posts to.
        """
        if (
            not isinstance(model, str)
            or not model.startswith("gemini-")
            or not has_known_limits(model)
        ):
            return _BUDGET_UNAVAILABLE_UNSUPPORTED_MODEL
        if not isinstance(self._api_key, str) or not self._api_key:
            # No usable Developer credential means the fixed Developer counting
            # route cannot be dispatched at all -- a route fact, not a model one.
            return _BUDGET_UNAVAILABLE_UNSUPPORTED_ROUTE
        if self._client is not None:
            # google-genai exposes ``Client.vertexai`` but has no established
            # public base_url property. A false value therefore cannot prove
            # an injected client is canonical, so only our creation snapshot
            # is eligible.
            canonical = (
                self._client is self._count_route_client
                and self._client_uses_canonical_developer_route is True
            )
        else:
            canonical = _sdk_environment_uses_canonical_developer_route()
        return None if canonical else _BUDGET_UNAVAILABLE_UNSUPPORTED_ROUTE

    def _native_counting_available(self, model: Any) -> bool:
        """Return whether a fixed Developer countTokens request can match generation."""
        return self._native_counting_unavailable_reason(model) is None

    async def _report_request_budget_unavailable(
        self, reason: str, model: Any, *, http_status: int | None = None
    ) -> None:
        """Best-effort: publish why this budget call produced no native count.

        Called exactly once per failed ``request_budget()`` invocation, from
        ``request_budget`` alone, so a helper and its caller never double-report
        the same failure. Never raises: a broken diagnostic hook must not turn a
        successful or legitimately-None count into a product failure, so an
        ordinary hook failure is caught -- ``BaseException`` (and therefore
        cancellation) is deliberately NOT caught and keeps propagating.
        """
        payload: dict[str, Any] = {
            "provider": "gemini",
            "method": "developer.countTokens",
            "reason": reason,
        }
        safe_model = _safe_event_model(model)
        if safe_model is not None:
            payload["model"] = safe_model
        if isinstance(http_status, int) and not isinstance(http_status, bool):
            payload["http_status"] = http_status

        if not (self.coordinator and hasattr(self.coordinator, "hooks")):
            # No observability channel exists: the same safe reason still has
            # to be visible, so state it once at warning level.
            logger.warning(
                "[PROVIDER] Gemini native countTokens unavailable (reason=%s, "
                "model=%s, http_status=%s)",
                reason,
                safe_model,
                payload.get("http_status"),
            )
            return
        try:
            await self.coordinator.hooks.emit(
                PROVIDER_REQUEST_BUDGET_UNAVAILABLE, payload
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "[PROVIDER] Gemini request_budget diagnostic hook failed "
                "(reason=%s)",
                reason,
            )

    def get_info(self) -> ProviderInfo:
        """Get provider metadata.

        `context_window` and `max_output_tokens` are REQUIRED here, not
        decorative: `context-simple._calculate_budget` reads this exact pair to
        size the session budget, and falls back to a conservative guess when a
        provider omits either. Omitting them is what kept Gemini sessions
        running on a ~200K budget against a 1,048,576-token window.

        The model reported is `self.default_model` -- what this instance was
        actually configured with -- not a hardcoded id.
        """
        limits = get_limits(self.default_model)
        capabilities = ["streaming", "tools", "thinking", "json_mode", "batch"]
        # The adapter is Developer API-only. Individual request shapes still
        # return no decision when their public serialization cannot be counted.
        if self._native_counting_available(self.default_model):
            capabilities.extend(["request_budget", "request_budget:provider_count"])
        return ProviderInfo(
            id="gemini",
            display_name="Google Gemini",
            credential_env_vars=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            capabilities=capabilities,
            defaults={
                "model": self.default_model,
                "max_tokens": limits.max_output_tokens,
                "context_window": limits.context_window,
                "max_output_tokens": limits.max_output_tokens,
                "temperature": 0.7,
                "timeout": 600.0,
            },
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your Google AI API key",
                    env_var="GOOGLE_API_KEY",
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """List available Gemini models via the live API.

        The query is retried with the same shared retry_with_backoff()/
        _retry_config machinery used by complete() on transient failures
        (5xx, timeouts, Cloudflare/CDN challenges, connection errors).
        The paginated async iterator is consumed to completion *inside*
        the retried attempt so that a failure mid-pagination retries the
        whole listing rather than resuming from a partial result. Raises
        the translated kernel error once retries are exhausted, or
        immediately for non-retryable errors (401/403/404) — no fallback;
        caller handles empty lists.

        This matches the behaviour of provider-anthropic and
        provider-openai, which both hard-fail (after retrying transient
        failures) rather than returning a stale hardcoded fallback.

        The previous implementation kept a hardcoded 5-model fallback list
        that drifted out of sync with actual Google releases and silently
        masked API outages. Removed 2026-04-22.
        """

        async def _do_list_models() -> list[Any]:
            """Single API call attempt with SDK -> kernel error translation.

            Mirrors the error-translation branches used by _do_complete()
            (rate limit, authentication, permission/Cloudflare-403, 5xx via
            genai_errors, google.api_core fallback, catch-all) so
            list_models() shares the same retry policy as complete(). The
            paginated iterator is fully consumed here — inside the
            try/except and therefore inside each individual retry
            attempt — so a failure partway through pagination causes the
            *entire* listing to be retried rather than yielding a partial
            result.
            """
            try:
                page = await self.client.aio.models.list()
                return [model async for model in page]
            except LLMError:
                raise  # Already translated, don't double-wrap
            except Exception as e:
                # --- Primary path: google.genai.errors (always available) ---
                if genai_errors is not None:
                    if isinstance(e, genai_errors.ClientError):
                        code = getattr(e, "code", None)
                        details = getattr(e, "details", None)
                        error_msg = (
                            json.dumps(details) if details is not None else str(e)
                        )
                        if code == 429:
                            # Try to extract Retry-After from httpx response headers.
                            retry_after_val = self._extract_retry_after(e)
                            # Fail-fast: if retry_after exceeds max_delay, mark non-retryable
                            retryable = True
                            if (
                                retry_after_val is not None
                                and retry_after_val > self._retry_config.max_delay
                            ):
                                retryable = False
                            raise RateLimitError(
                                error_msg,
                                provider="gemini",
                                status_code=429,
                                retryable=retryable,
                                retry_after=retry_after_val,
                            ) from e
                        if code == 401:
                            raise AuthenticationError(
                                error_msg, provider="gemini", status_code=401
                            ) from e
                        if code == 403:
                            if self._is_cloudflare_challenge(e):
                                logger.warning(_CLOUDFLARE_403_WARNING)
                                raise ProviderUnavailableError(
                                    "CDN/proxy challenge (transient 403). "
                                    "This typically resolves on retry.",
                                    provider="gemini",
                                    status_code=403,
                                    retryable=True,
                                ) from e
                            raise AccessDeniedError(
                                error_msg, provider="gemini", status_code=403
                            ) from e
                        if code == 404:
                            raise NotFoundError(
                                error_msg, provider="gemini", status_code=404
                            ) from e
                        raise InvalidRequestError(
                            error_msg,
                            provider="gemini",
                            status_code=code or 400,
                        ) from e
                    if isinstance(e, genai_errors.ServerError):
                        code = getattr(e, "code", None)
                        details = getattr(e, "details", None)
                        error_msg = (
                            json.dumps(details) if details is not None else str(e)
                        )
                        retry_after_val = self._extract_retry_after(e)
                        raise ProviderUnavailableError(
                            error_msg,
                            provider="gemini",
                            status_code=code or 500,
                            retryable=True,
                            retry_after=retry_after_val,
                        ) from e

                # --- Fallback: google.api_core.exceptions (if installed) ---
                if google_exceptions is not None:
                    if isinstance(e, google_exceptions.ResourceExhausted):
                        retry_after_val = self._extract_retry_after(e)
                        retryable = True
                        if (
                            retry_after_val is not None
                            and retry_after_val > self._retry_config.max_delay
                        ):
                            retryable = False
                        raise RateLimitError(
                            str(e),
                            provider="gemini",
                            status_code=429,
                            retryable=retryable,
                            retry_after=retry_after_val,
                        ) from e
                    if isinstance(e, google_exceptions.Unauthenticated):
                        raise AuthenticationError(
                            str(e), provider="gemini", status_code=401
                        ) from e
                    if isinstance(e, google_exceptions.PermissionDenied):
                        # Falsy check (not just `is None`) is intentional:
                        # google.api_core exceptions may have empty details
                        # ([], "", etc.) which also indicate a CDN/proxy 403.
                        if not getattr(e, "details", None):
                            logger.warning(_CLOUDFLARE_403_WARNING)
                            raise ProviderUnavailableError(
                                "CDN/proxy challenge (transient 403). "
                                "This typically resolves on retry.",
                                provider="gemini",
                                status_code=403,
                                retryable=True,
                            ) from e
                        raise AccessDeniedError(
                            str(e), provider="gemini", status_code=403
                        ) from e
                    if isinstance(e, google_exceptions.NotFound):
                        raise NotFoundError(
                            str(e), provider="gemini", status_code=404
                        ) from e
                    if isinstance(e, google_exceptions.InvalidArgument):
                        raise InvalidRequestError(
                            str(e), provider="gemini", status_code=400
                        ) from e
                    if isinstance(e, google_exceptions.ServiceUnavailable):
                        raise ProviderUnavailableError(
                            str(e),
                            provider="gemini",
                            status_code=503,
                            retryable=True,
                        ) from e
                    if isinstance(e, google_exceptions.DeadlineExceeded):
                        raise LLMTimeoutError(
                            str(e),
                            provider="gemini",
                            retryable=True,
                        ) from e

                # Unknown errors default to retryable per design doc
                details = getattr(e, "details", None)
                error_msg = (
                    json.dumps(details)
                    if details is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                raise LLMError(
                    error_msg,
                    provider="gemini",
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: LLMError):
            """Callback invoked before each retry sleep."""
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    PROVIDER_RETRY,
                    {
                        "provider": "gemini",
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        raw_models = await retry_with_backoff(
            _do_list_models,
            self._retry_config,
            on_retry=_on_retry,
        )

        models: list[ModelInfo] = []
        for model in raw_models:
            model_name = getattr(model, "name", "")
            # Filter to gemini models only (exclude tuned models, etc.)
            if not model_name or "gemini" not in model_name.lower():
                continue

            # Extract model ID from name (format: models/gemini-2.5-flash)
            model_id = model_name.split("/")[-1] if "/" in model_name else model_name

            # Skip experimental/deprecated models
            if "exp" in model_id or "001" in model_id or "002" in model_id:
                continue

            display_name = getattr(model, "display_name", model_id)
            input_limit = getattr(
                model, "input_token_limit", DEFAULT_LIMITS.context_window
            )
            output_limit = getattr(
                model, "output_token_limit", DEFAULT_LIMITS.max_output_tokens
            )
            supports_thinking = getattr(model, "thinking", False)

            # Determine capabilities based on model
            capabilities = ["streaming", "json_mode"]
            if supports_thinking or "2.5" in model_id or "3" in model_id:
                capabilities.append("thinking")
            # All gemini models except 2.0-flash-lite support tools
            if "2.0-flash-lite" not in model_id:
                capabilities.append("tools")
            if "flash" in model_id.lower():
                capabilities.append("fast")
            # All Gemini 2.x+ models support vision (image input)
            if "2." in model_id or "3" in model_id:
                capabilities.append("vision")

            models.append(
                ModelInfo(
                    id=model_id,
                    display_name=display_name,
                    context_window=input_limit,
                    max_output_tokens=output_limit,
                    capabilities=capabilities,
                    defaults={
                        "temperature": 0.7,
                        "max_tokens": output_limit,
                    },
                )
            )

        return models

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        """Extract Retry-After value from a Gemini SDK exception.

        The GenAI SDK's APIError stores the underlying httpx.Response on
        ``exc.response``.  When a 429 is returned, the Gemini API *may*
        include a ``Retry-After`` header (seconds).

        Returns:
            Parsed delay in seconds, or None if the header is absent or
            unparseable.
        """
        response = getattr(exc, "response", None)
        if response is None:
            return None

        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None

        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.
        Filters out IDs already repaired in previous iterations.

        Returns:
            List of (msg_index, call_id, tool_name, tool_arguments) tuples for
            unpaired calls, where msg_index is the index of the assistant message
            that contains the tool call.
        """
        tool_calls: dict[
            str, tuple[int, str, dict]
        ] = {}  # {call_id: (idx, name, args)}
        tool_results: set[str] = set()  # {call_id}

        for idx, msg in enumerate(messages):
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            # Check tool messages for tool_call_id
            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> Message:
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.
        """
        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    def _repair_missing_tool_results(
        self, request: ChatRequest, *, record_repairs: bool
    ) -> list[tuple[int, str, str, dict]]:
        """Apply the existing tool-result repair to one request and return it."""
        missing = self._find_missing_tool_results(request.messages)
        if not missing:
            return missing

        by_message_index: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for message_index, call_id, tool_name, _ in missing:
            by_message_index[message_index].append((call_id, tool_name))
        for message_index in sorted(by_message_index, reverse=True):
            synthetics = [
                self._create_synthetic_result(call_id, tool_name)
                for call_id, tool_name in by_message_index[message_index]
            ]
            if record_repairs:
                self._repaired_tool_ids.update(
                    call_id for call_id, _ in by_message_index[message_index]
                )
            insert_position = message_index + 1
            for offset, synthetic in enumerate(synthetics):
                request.messages.insert(insert_position + offset, synthetic)
            post_insert_index = insert_position + len(synthetics)
            if post_insert_index >= len(request.messages):
                continue
            next_message = request.messages[post_insert_index]
            is_real_user_message = (
                next_message.role == "user"
                and not getattr(next_message, "tool_call_id", None)
                and not (
                    isinstance(next_message.content, str)
                    and next_message.content.startswith("<system-reminder>")
                )
            )
            if is_real_user_message:
                request.messages.insert(
                    post_insert_index,
                    Message(
                        role="assistant",
                        content="[SYSTEM: Tool results received. Continuing conversation.]",
                    ),
                )
        return missing

    async def complete(
        self,
        request: ChatRequest,
        *,
        request_options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        kwargs = self._merge_request_options(request_options, kwargs)

        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        missing = self._repair_missing_tool_results(request, record_repairs=True)
        if missing:
            logger.warning(
                f"[PROVIDER] Gemini: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    def _resolve_thinking_config(
        self, model: str, request: ChatRequest, kwargs: dict[str, Any]
    ):
        """Resolve the ThinkingConfig to send for this request.

        Precedence (highest first):

        1. Explicit ``thinking_budget`` via kwargs or ``request.metadata`` --
           the legacy override. Honored ALONE: never combined with
           thinking_level (Google 400s if both are set on one request).
        2. ``request.reasoning_effort`` -- mapped to a thinking_level target
           (see _EFFORT_TO_LEVEL), clamped per-model (see
           _supported_thinking_levels / _clamp_thinking_level). Models that
           reject thinking_level entirely fall back to the legacy numeric
           mapping instead (the only control they have):
           none=0, minimal/low=4096, medium/high/xhigh/max=-1 (dynamic).
        3. No directive at all -- omit both thinking_budget and
           thinking_level, keeping only include_thoughts. Gemini models
           think by default (dynamically) even with a config that sets
           neither field; forcing an explicit dynamic budget of -1 is
           functionally equivalent but needlessly forecloses "let the model
           use its own built-in default" for level-only models. Verified
           live: ThinkingConfig(include_thoughts=True) with no budget/level
           still returns thought summaries at the model's own default
           thinking amount, on both gemini-2.5-flash and gemini-3.7-flash.

        A note on disabling thinking: explicitly sending thinking_budget=0
        DOES disable thinking on Gemini 2.x models (verified live:
        thoughts_token_count becomes None) -- but *omitting* the config
        entirely does NOT (verified live: thoughts_token_count is still
        populated with no thinking_config sent at all). These are not
        equivalent, so the explicit-zero case below always sends the
        field rather than omitting it -- omitting it here was the
        pre-existing implementation's bug (thinking_budget=0 never actually
        reached the API), fixed as part of this same change since it's the
        exact code path being redesigned.

        Gemini 3.x cannot disable thinking at all regardless of what is
        sent (verified live: thinking_budget=0 on gemini-3.7-flash still
        produced ~26 thinking tokens; there is no "off" thinking_level).
        That is a vendor limitation, not something this method can work
        around.
        """
        from google import genai

        include_thoughts = True
        if request.metadata and "include_thoughts" in request.metadata:
            include_thoughts = request.metadata["include_thoughts"]
        if "include_thoughts" in kwargs:
            include_thoughts = kwargs["include_thoughts"]

        # --- 1. Explicit thinking_budget (legacy override) -----------------
        explicit_budget = None
        if request.metadata and "thinking_budget" in request.metadata:
            explicit_budget = request.metadata["thinking_budget"]
        if "thinking_budget" in kwargs:
            explicit_budget = kwargs["thinking_budget"]

        if explicit_budget is not None:
            return genai.types.ThinkingConfig(
                thinking_budget=explicit_budget, include_thoughts=include_thoughts
            )

        # --- 2. reasoning_effort -> thinking_level (clamped per-model) ------
        if request.reasoning_effort:
            effort = request.reasoning_effort.lower()
            supported = _supported_thinking_levels(model)

            if supported is None:
                # This model has no thinking_level control at all -- the
                # legacy numeric mapping is the only lever available.
                if effort == "none":
                    budget = 0
                elif effort in ("minimal", "low"):
                    budget = 4096
                else:
                    budget = -1  # medium/high/xhigh/max -> dynamic
                return genai.types.ThinkingConfig(
                    thinking_budget=budget, include_thoughts=include_thoughts
                )

            if effort == "none":
                if "minimal" in supported:
                    return genai.types.ThinkingConfig(
                        thinking_level=genai.types.ThinkingLevel.MINIMAL,
                        include_thoughts=include_thoughts,
                    )
                # This model can't go any lower than its own default, and
                # (for Gemini 3.x) may not be able to disable thinking at
                # all -- fall through to the model's own default amount.
                logger.info(
                    "[PROVIDER] Gemini: reasoning_effort='none' requested but "
                    "'%s' has no thinking_level below its own default -- "
                    "using the model's default thinking amount instead",
                    model,
                )
                return genai.types.ThinkingConfig(include_thoughts=include_thoughts)

            target = _EFFORT_TO_LEVEL.get(effort)
            if target is None:
                logger.warning(
                    "[PROVIDER] Gemini: unknown reasoning_effort '%s' -- "
                    "ignoring, using model default thinking",
                    request.reasoning_effort,
                )
                return genai.types.ThinkingConfig(include_thoughts=include_thoughts)

            level = _clamp_thinking_level(model, target, supported)
            return genai.types.ThinkingConfig(
                thinking_level=genai.types.ThinkingLevel(level.upper()),
                include_thoughts=include_thoughts,
            )

        # --- 3. No directive at all -----------------------------------------
        return genai.types.ThinkingConfig(include_thoughts=include_thoughts)

    @staticmethod
    def _merge_request_options(
        request_options: Mapping[str, Any] | None, kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Merge released Loop options before direct keyword overrides."""
        if request_options is None:
            return dict(kwargs)
        if not isinstance(request_options, Mapping):
            raise ValueError("request_options must be a mapping or None")
        merged = dict(request_options)
        merged.update(kwargs)
        return merged

    def _build_request_plan(
        self, request: ChatRequest, *, warn_on_extra: bool = False, **kwargs: Any
    ) -> _GeminiRequestPlan:
        """Purely assemble the one GenerateContent plan used by both paths."""
        from google import genai

        system_msgs = [
            message for message in request.messages if message.role == "system"
        ]
        developer_msgs = [
            message for message in request.messages if message.role == "developer"
        ]
        conversation = [
            message
            for message in request.messages
            if message.role in ("user", "assistant", "tool")
        ]
        system_instruction = (
            "\n\n".join(
                message.content if isinstance(message.content, str) else ""
                for message in system_msgs
            )
            if system_msgs
            else None
        )
        context_user_messages = []
        for message in developer_msgs:
            developer_content = (
                message.content if isinstance(message.content, str) else ""
            )
            context_user_messages.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"<context_file>\n{developer_content}\n</context_file>"
                            )
                        }
                    ],
                }
            )
        _, conversation_messages = self._convert_messages(
            [message.model_dump() for message in conversation]
        )
        model = kwargs.get("model", self.default_model)
        temperature = request.temperature or kwargs.get("temperature", self.temperature)
        max_tokens = request.max_output_tokens or kwargs.get(
            "max_tokens", self.max_tokens
        )
        config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        config.thinking_config = self._resolve_thinking_config(model, request, kwargs)
        if system_instruction:
            config.system_instruction = system_instruction
        if request.tools:
            config.tools = [
                genai.types.Tool(
                    function_declarations=self._convert_tools_from_request(
                        request.tools
                    )
                )
            ]
        # Disable the SDK's automatic function calling on EVERY request, not
        # only tool-carrying ones. AFC is an SDK-local execution loop: this
        # provider returns tool calls to the Loop and never lets the SDK call a
        # Python callable on its own. Setting it unconditionally keeps one
        # SDK-facing shape for both request kinds. It stays an SDK-local field
        # excluded from the count projection (_COUNT_LOCAL_CONFIG_FIELDS), so
        # it does not change the countTokens body. extra_request_params is
        # still applied afterwards, so an explicit supported override wins.
        config.automatic_function_calling = (
            genai.types.AutomaticFunctionCallingConfig(disable=True)
        )
        _apply_extra_request_params(
            config,
            self.extra_request_params,
            warn=warn_on_extra,
        )
        count_unsupported_reason = None
        if "tools" in self.extra_request_params:
            count_unsupported_reason = (
                "extra tools require SDK tool-list transformation"
            )
        elif "system_instruction" in self.extra_request_params:
            count_unsupported_reason = (
                "extra system instruction requires SDK normalization"
            )
        elif "cached_content" in self.extra_request_params and not (
            isinstance(self.extra_request_params["cached_content"], str)
            and self.extra_request_params["cached_content"].startswith("cachedContents/")
        ):
            count_unsupported_reason = (
                "cached content must use a canonical resource name"
            )
        return _GeminiRequestPlan(
            model=model,
            contents=context_user_messages + conversation_messages,
            config=config,
            count_unsupported_reason=count_unsupported_reason,
        )

    @staticmethod
    def _count_request_payload(plan: _GeminiRequestPlan) -> dict[str, Any]:
        """Project the public typed plan into REST ``generateContentRequest``.

        The projection intentionally has no catch-all omission path: every
        serialized SDK field needs a named top-level, generation, local, or
        unsupported disposition before it can be counted.
        """
        from google import genai

        try:
            contents = [
                json.loads(
                    genai.types.Content.model_validate(content).model_dump_json(
                        by_alias=True,
                        exclude_none=True,
                    )
                )
                for content in plan.contents
            ]
            # extra_request_params is intentionally applied after construction
            # and can contain mappings. Re-validate through the SDK's public
            # typed model before serialization so nested aliases match the
            # SDK's generation transformer as well.
            config = type(plan.config).model_validate(plan.config.model_dump())
            serialized_config = json.loads(
                config.model_dump_json(by_alias=True, exclude_none=True)
            )
        except (TypeError, ValueError) as exc:
            raise _UnsupportedCountRequest(
                "Gemini request plan cannot be publicly serialized for countTokens"
            ) from exc

        _match_sdk_generation_wire(serialized_config)
        # REST CountTokensRequest's `generateContentRequest` variant carries a
        # nested GenerateContentRequest whose `model` field is REQUIRED
        # (google/ai/generativelanguage/v1beta/generative_service.proto). It is a
        # resource name, not a bare id, so it is prefixed here. Count eligibility
        # only ever admits bare `gemini-` ids (see
        # `_native_counting_unavailable_reason`), so this cannot double-prefix.
        # The endpoint path is built from the same `plan.model`, so the URL and
        # this field always name the same effective model.
        request: dict[str, Any] = {
            "model": f"models/{plan.model}",
            "contents": contents,
        }
        generation_config: dict[str, Any] = {}
        for key, value in serialized_config.items():
            if key in _COUNT_LOCAL_CONFIG_FIELDS:
                continue
            if key == "systemInstruction" and not isinstance(
                getattr(config, "system_instruction", None), genai.types.Content
            ):
                instruction = config.system_instruction
                try:
                    value = json.loads(
                        _count_system_instruction_content(
                            instruction, genai.types
                        ).model_dump_json(by_alias=True, exclude_none=True)
                    )
                except _UnsupportedCountRequest:
                    raise
                except (TypeError, ValueError) as exc:
                    raise _UnsupportedCountRequest(
                        "Gemini system instruction cannot be publicly serialized "
                        "for countTokens"
                    ) from exc
            if key in _COUNT_TOP_LEVEL_CONFIG_FIELDS:
                request[key] = value
            elif key in _COUNT_GENERATION_CONFIG_FIELDS:
                generation_config[key] = value
            elif key in _COUNT_UNSUPPORTED_CONFIG_FIELDS:
                raise _UnsupportedCountRequest(
                    f"Gemini Developer countTokens cannot project {key}"
                )
            else:
                raise _UnsupportedCountRequest(
                    f"Gemini count projection lacks a disposition for {key}"
                )
        if generation_config:
            request["generationConfig"] = generation_config
        return {"generateContentRequest": request}

    async def _count_request_tokens(
        self, plan: _GeminiRequestPlan
    ) -> tuple[int | None, str | None, int | None]:
        """Call the documented Developer REST counter without SDK internals.

        Returns ``(count, reason, http_status)``; exactly one of ``count`` and
        ``reason`` is ever set. This helper NEVER emits the diagnostic event
        itself -- it hands the safe reason code back to ``request_budget()``,
        which is the single reporting site. One failed budget call therefore
        produces exactly one report, not one per layer.
        """
        route_or_model_reason = self._native_counting_unavailable_reason(plan.model)
        if route_or_model_reason is not None:
            return None, route_or_model_reason, None
        if plan.count_unsupported_reason is not None:
            # plan.count_unsupported_reason is an internal English string kept
            # for local debugging; only the fixed code leaves this module.
            logger.debug(
                "[PROVIDER] Gemini native countTokens is unavailable: %s",
                plan.count_unsupported_reason,
            )
            return None, _BUDGET_UNAVAILABLE_REQUEST_PROJECTION, None
        try:
            payload = self._count_request_payload(plan)
        except _UnsupportedCountRequest:
            return None, _BUDGET_UNAVAILABLE_REQUEST_PROJECTION, None

        import httpx

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(plan.model, safe='')}:countTokens"
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    endpoint,
                    headers={"x-goog-api-key": self._api_key},
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as exc:
            # The status code is a safe bounded integer. The response body,
            # request body, URL and exception text stay unlogged on purpose.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(status, bool) or not isinstance(status, int):
                status = None
            logger.debug(
                "[PROVIDER] Gemini native countTokens returned an error status "
                "(http_status=%s)",
                status,
            )
            return None, _BUDGET_UNAVAILABLE_HTTP_ERROR, status
        except httpx.HTTPError:
            logger.debug("[PROVIDER] Gemini native countTokens transport failed")
            return None, _BUDGET_UNAVAILABLE_HTTP_ERROR, None
        except (TypeError, ValueError):
            # Unserializable request payload, or an undecodable response body.
            logger.debug(
                "[PROVIDER] Gemini native countTokens produced no decodable result"
            )
            return None, _BUDGET_UNAVAILABLE_INVALID_RESPONSE, None

        total_tokens = result.get("totalTokens") if isinstance(result, dict) else None
        if (
            isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens < 0
        ):
            logger.debug("[PROVIDER] Gemini countTokens returned no valid totalTokens")
            return None, _BUDGET_UNAVAILABLE_INVALID_RESPONSE, None
        return total_tokens, None, None

    async def request_budget(
        self,
        request: ChatRequest,
        *,
        context_estimate: int,
        request_options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Return one exact Developer API input count when this request supports it."""
        if isinstance(context_estimate, bool) or not isinstance(context_estimate, int):
            raise ValueError("context_estimate must be a nonnegative integer")
        if context_estimate < 0:
            raise ValueError("context_estimate must be a nonnegative integer")

        options = self._merge_request_options(request_options, kwargs)
        selected_model = options.get("model", self.default_model)
        reason = self._native_counting_unavailable_reason(selected_model)
        if reason is not None:
            await self._report_request_budget_unavailable(reason, selected_model)
            return None
        # `complete()` repairs incomplete tool pairs before it builds the
        # wire request. Plan the same repair on a deep copy: probing must not
        # mutate conversation history or mark an uncommitted repair as done.
        try:
            planned_request = request.model_copy(deep=True)
            self._repair_missing_tool_results(planned_request, record_repairs=False)
            plan = self._build_request_plan(planned_request, **options)
        except (AttributeError, TypeError, ValueError):
            # No exc_info: a traceback here can carry request content into the
            # log. The fixed code is the whole diagnostic.
            logger.debug(
                "[PROVIDER] Gemini native countTokens request plan was unavailable"
            )
            await self._report_request_budget_unavailable(
                _BUDGET_UNAVAILABLE_REQUEST_PLAN, selected_model
            )
            return None
        reason = self._native_counting_unavailable_reason(plan.model)
        if reason is not None:
            await self._report_request_budget_unavailable(reason, plan.model)
            return None
        max_output_tokens = getattr(plan.config, "max_output_tokens", None)
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            await self._report_request_budget_unavailable(
                _BUDGET_UNAVAILABLE_INVALID_OUTPUT_LIMIT, plan.model
            )
            return None
        count, reason, http_status = await self._count_request_tokens(plan)
        if count is None:
            await self._report_request_budget_unavailable(
                reason or _BUDGET_UNAVAILABLE_INVALID_RESPONSE,
                plan.model,
                http_status=http_status,
            )
            return None

        input_limit = get_limits(plan.model).context_window
        if count <= input_limit:
            target = context_estimate
        elif context_estimate <= 0:
            target = 0
        else:
            target = min(
                context_estimate - 1,
                max(1, math.floor(context_estimate * input_limit / count) - 1),
            )
        return {
            "estimated_input_tokens": count,
            "input_limit_tokens": input_limit,
            "context_token_budget": target,
            "max_output_tokens": max_output_tokens,
            "measurement": {
                "kind": "provider_count",
                "source": "gemini.developer.countTokens",
                "input_tokens": count,
            },
        }

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> ChatResponse:
        """
        Handle ChatRequest format with developer message conversion.

        Includes error translation (native SDK errors -> kernel types) and
        retry with exponential backoff for transient failures.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.debug(f"Received ChatRequest with {len(request.messages)} messages")
        plan = self._build_request_plan(request, warn_on_extra=True, **kwargs)
        model = plan.model
        all_messages = plan.contents
        config = plan.config
        system_instruction = config.system_instruction
        temperature = config.temperature
        max_tokens = config.max_output_tokens

        logger.info(f"Gemini API call - model: {model}, messages: {len(all_messages)}")

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_payload: dict[str, Any] = {
                "provider": "gemini",
                "model": model,
                "message_count": len(all_messages),
                "has_system": bool(system_instruction),
            }
            if self.raw:
                request_payload["raw"] = redact_secrets(
                    {
                        "model": model,
                        "messages": all_messages,
                        "system_instruction": system_instruction,
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
            await self.coordinator.hooks.emit("llm:request", request_payload)

        start_time = time.time()

        # Call Gemini API with shared retry_with_backoff from amplifier-core.
        # Error translation happens inside _do_complete() so that retry_with_backoff
        # sees LLMError (and checks retryable) rather than raw SDK exceptions.

        async def _do_complete():
            """Single API call attempt with SDK → kernel error translation."""
            try:
                # Call Gemini API (use .aio for async)
                return await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=model, contents=all_messages, config=config
                    ),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError as e:
                raise LLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider="gemini",
                    retryable=True,
                ) from e
            except RateLimitError as e:
                # Fail-fast: if retry_after exceeds max_delay, mark non-retryable
                if (
                    e.retry_after is not None
                    and e.retry_after > self._retry_config.max_delay
                ):
                    e.retryable = False
                raise
            except LLMError:
                raise  # Already translated, don't double-wrap
            except Exception as e:
                # --- Primary path: google.genai.errors (always available) ---
                if genai_errors is not None:
                    if isinstance(e, genai_errors.ClientError):
                        code = getattr(e, "code", None)
                        details = getattr(e, "details", None)
                        error_msg = (
                            json.dumps(details) if details is not None else str(e)
                        )
                        if code == 429:
                            # Try to extract Retry-After from httpx response headers.
                            retry_after_val = self._extract_retry_after(e)
                            # Fail-fast: if retry_after exceeds max_delay, mark non-retryable
                            retryable = True
                            if (
                                retry_after_val is not None
                                and retry_after_val > self._retry_config.max_delay
                            ):
                                retryable = False
                            raise RateLimitError(
                                error_msg,
                                provider="gemini",
                                status_code=429,
                                retryable=retryable,
                                retry_after=retry_after_val,
                            ) from e
                        if code == 401:
                            raise AuthenticationError(
                                error_msg, provider="gemini", status_code=401
                            ) from e
                        if code == 403:
                            if self._is_cloudflare_challenge(e):
                                logger.warning(_CLOUDFLARE_403_WARNING)
                                raise ProviderUnavailableError(
                                    "CDN/proxy challenge (transient 403). "
                                    "This typically resolves on retry.",
                                    provider="gemini",
                                    status_code=403,
                                    retryable=True,
                                ) from e
                            raise AccessDeniedError(
                                error_msg, provider="gemini", status_code=403
                            ) from e
                        # Sub-classify 4xx by message body
                        raw_msg = str(e).lower()
                        if (
                            "context length" in raw_msg
                            or "too many tokens" in raw_msg
                            or "token limit" in raw_msg
                            or (
                                "exceeds" in raw_msg
                                and (
                                    "token" in raw_msg
                                    or "context" in raw_msg
                                    or "length" in raw_msg
                                )
                            )
                        ):
                            raise ContextLengthError(
                                error_msg,
                                provider="gemini",
                                status_code=getattr(e, "status_code", 400),
                            ) from e
                        if (
                            "content filter" in raw_msg
                            or "safety" in raw_msg
                            or "blocked" in raw_msg
                            or "harm" in raw_msg
                        ):
                            raise ContentFilterError(
                                error_msg,
                                provider="gemini",
                                status_code=getattr(e, "status_code", 400),
                            ) from e
                        raise InvalidRequestError(
                            error_msg,
                            provider="gemini",
                            status_code=code or 400,
                        ) from e
                    if isinstance(e, genai_errors.ServerError):
                        code = getattr(e, "code", None)
                        details = getattr(e, "details", None)
                        error_msg = (
                            json.dumps(details) if details is not None else str(e)
                        )
                        retry_after_val = self._extract_retry_after(e)
                        raise ProviderUnavailableError(
                            error_msg,
                            provider="gemini",
                            status_code=code or 500,
                            retryable=True,
                            retry_after=retry_after_val,
                        ) from e

                # --- Fallback: google.api_core.exceptions (if installed) ---
                if google_exceptions is not None:
                    if isinstance(e, google_exceptions.ResourceExhausted):
                        retry_after_val = self._extract_retry_after(e)
                        retryable = True
                        if (
                            retry_after_val is not None
                            and retry_after_val > self._retry_config.max_delay
                        ):
                            retryable = False
                        raise RateLimitError(
                            str(e),
                            provider="gemini",
                            status_code=429,
                            retryable=retryable,
                            retry_after=retry_after_val,
                        ) from e
                    if isinstance(e, google_exceptions.Unauthenticated):
                        raise AuthenticationError(
                            str(e), provider="gemini", status_code=401
                        ) from e
                    if isinstance(e, google_exceptions.PermissionDenied):
                        # Falsy check (not just `is None`) is intentional:
                        # google.api_core exceptions may have empty details
                        # ([], "", etc.) which also indicate a CDN/proxy 403.
                        if not getattr(e, "details", None):
                            logger.warning(_CLOUDFLARE_403_WARNING)
                            raise ProviderUnavailableError(
                                "CDN/proxy challenge (transient 403). "
                                "This typically resolves on retry.",
                                provider="gemini",
                                status_code=403,
                                retryable=True,
                            ) from e
                        raise AccessDeniedError(
                            str(e), provider="gemini", status_code=403
                        ) from e
                    if isinstance(e, google_exceptions.InvalidArgument):
                        raise InvalidRequestError(
                            str(e), provider="gemini", status_code=400
                        ) from e
                    if isinstance(e, google_exceptions.ServiceUnavailable):
                        raise ProviderUnavailableError(
                            str(e),
                            provider="gemini",
                            status_code=503,
                            retryable=True,
                        ) from e
                    if isinstance(e, google_exceptions.DeadlineExceeded):
                        raise LLMTimeoutError(
                            str(e),
                            provider="gemini",
                            retryable=True,
                        ) from e

                # Unknown errors default to retryable per design doc
                details = getattr(e, "details", None)
                error_msg = (
                    json.dumps(details)
                    if details is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                raise LLMError(
                    error_msg,
                    provider="gemini",
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: LLMError):
            """Callback invoked before each retry sleep."""
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    PROVIDER_RETRY,
                    {
                        "provider": self.name,
                        "model": model,
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        async def _do_complete_guarded():
            """Semaphore-gated wrapper around _do_complete with concurrency logging.

            Acquires the process-wide concurrency semaphore before each API call
            attempt so that at most max_concurrent_requests calls are in-flight
            simultaneously across all provider instances in this process.

            This is the function passed to retry_with_backoff so that:
            - the semaphore is *released* between retry attempts (during backoff sleep)
            - each fresh attempt must re-acquire before hitting the network
            """
            global _active_requests, _waiting_requests
            sem = await _get_process_semaphore(self._max_concurrent_requests)
            if sem is not None:
                _waiting_requests += 1
                async with sem:
                    _waiting_requests -= 1
                    _active_requests += 1
                    try:
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:concurrency",
                                {
                                    "provider": "gemini",
                                    "model": model,
                                    "active_requests": _active_requests,
                                    "waiting_requests": _waiting_requests,
                                    "max_concurrent": self._max_concurrent_requests,
                                    "process_id": os.getpid(),
                                },
                            )
                        return await _do_complete()
                    finally:
                        _active_requests -= 1
            else:
                # Semaphore disabled (max_concurrent_requests=0) — still log
                _active_requests += 1
                try:
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "provider:concurrency",
                            {
                                "provider": "gemini",
                                "model": model,
                                "active_requests": _active_requests,
                                "waiting_requests": _waiting_requests,
                                "max_concurrent": 0,
                                "process_id": os.getpid(),
                            },
                        )
                    return await _do_complete()
                finally:
                    _active_requests -= 1

        # ----------------------------------------------------------------
        # Per-request streaming override (contract §Per-request stream override)
        # ----------------------------------------------------------------
        _meta = getattr(request, "metadata", None)
        _use_streaming = self.use_streaming
        if isinstance(_meta, dict) and _meta.get("stream") is False:
            _use_streaming = False

        async def _do_complete_streaming():
            """Streaming API call with contract-compliant event emission.

            Iterates generate_content_stream chunks, synthesises block
            start/end boundaries (Gemini has no explicit ones), and emits
            the five contract events per provider-streaming-contract.md:
              llm:stream_block_start, llm:stream_block_delta (text + thinking),
              llm:stream_block_end, llm:stream_aborted.

            Timeout is enforced per-anext() because asyncio.wait_for
            cannot wrap an async generator directly.
            """
            from types import SimpleNamespace as _NS

            request_id = str(uuid.uuid4())
            block_index = -1
            current_block_type: str | None = None
            seq: dict[int, int] = {}          # block_index -> next sequence number
            partial_emitted = False
            hooks_available = bool(
                self.coordinator and hasattr(self.coordinator, "hooks")
            )

            # Per-block text accumulator and running thought_signature
            current_text_buf: list[str] = []
            current_sig = None
            # Collected virtual parts for _convert_to_chat_response
            collected_parts: list = []
            final_usage_metadata = None

            def _flush_block() -> None:
                """Merge text fragments into one part and append to collected_parts."""
                nonlocal current_text_buf, current_sig
                if current_block_type in ("text", "thinking"):
                    combined = "".join(current_text_buf)
                    if combined:
                        collected_parts.append(
                            _NS(
                                text=combined,
                                thought=(current_block_type == "thinking"),
                                thought_signature=current_sig,
                            )
                        )
                current_text_buf.clear()
                current_sig = None

            async def _open_block(btype: str, name: str | None = None) -> None:
                nonlocal block_index, current_block_type
                block_index += 1
                current_block_type = btype
                seq[block_index] = 0
                if hooks_available:
                    payload: dict[str, Any] = {
                        "request_id": request_id,
                        "block_index": block_index,
                        "block_type": btype,
                    }
                    if name is not None:
                        payload["name"] = name
                    await self.coordinator.hooks.emit(
                        "llm:stream_block_start", payload
                    )

            async def _close_block() -> None:
                nonlocal current_block_type
                if current_block_type is None:
                    return
                if hooks_available:
                    await self.coordinator.hooks.emit(
                        "llm:stream_block_end",
                        {
                            "request_id": request_id,
                            "block_index": block_index,
                            "block_type": current_block_type,
                        },
                    )
                _flush_block()
                current_block_type = None

            try:
                # Establish the stream; timeout covers connection setup
                try:
                    stream = await asyncio.wait_for(
                        self.client.aio.models.generate_content_stream(
                            model=model,
                            contents=all_messages,
                            config=config,
                        ),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError as _te:
                    raise LLMTimeoutError(
                        f"Stream connection timed out after {self.timeout}s",
                        provider="gemini",
                        retryable=True,
                    ) from _te

                # Iterate chunks; timeout enforced per-anext()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream.__anext__(),
                            timeout=self.timeout,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as _te:
                        raise LLMTimeoutError(
                            f"Stream timed out waiting for next chunk after {self.timeout}s",
                            provider="gemini",
                            retryable=True,
                        ) from _te

                    # Capture usage (present only on the final chunk)
                    _um = getattr(chunk, "usage_metadata", None)
                    if _um:
                        final_usage_metadata = _um

                    # Guard against heartbeat chunks
                    try:
                        parts = chunk.candidates[0].content.parts
                    except (AttributeError, IndexError, TypeError):
                        continue
                    if not parts:
                        continue

                    for part in parts:
                        # Always capture any thought_signature (seal fragments)
                        _sig = getattr(part, "thought_signature", None)
                        if _sig is not None:
                            current_sig = _sig

                        _has_text = hasattr(part, "text") and bool(
                            getattr(part, "text", None)
                        )
                        _has_fc = getattr(part, "function_call", None) is not None
                        _is_thought = getattr(part, "thought", False) is True

                        # Seal-only part (signature but no text/fc) — already captured
                        if not _has_text and not _has_fc:
                            continue

                        # Determine part type
                        if _is_thought:
                            part_type = "thinking"
                        elif _has_fc:
                            part_type = "tool_use"
                        else:
                            part_type = "text"

                        # ---- tool_use: arrives whole; immediate open+close ----
                        if part_type == "tool_use":
                            fc = part.function_call
                            await _close_block()
                            await _open_block("tool_use", name=fc.name)
                            # Accumulate for final response assembly
                            _tc_sig = getattr(part, "thought_signature", None)
                            collected_parts.append(
                                _NS(function_call=fc, thought_signature=_tc_sig)
                            )
                            await _close_block()
                            continue

                        # ---- text / thinking: type-transition state machine ----
                        text = part.text  # already confirmed truthy via _has_text

                        if part_type != current_block_type:
                            await _close_block()
                            await _open_block(part_type)

                        # Emit delta — ONE event for all block content (contract: guard with if text:)
                        # block_type sourced from current_block_type (equals part_type after _open_block)
                        if text:
                            if hooks_available:
                                await self.coordinator.hooks.emit(
                                    "llm:stream_block_delta",
                                    {
                                        "request_id": request_id,
                                        "block_index": block_index,
                                        "block_type": current_block_type,
                                        "sequence": seq[block_index],
                                        "text": text,
                                    },
                                )
                            seq[block_index] += 1
                            partial_emitted = True
                            current_text_buf.append(text)

                # Close the final open block (synthesised boundary at stream end)
                await _close_block()

                # Assemble ChatResponse by reusing _convert_to_chat_response
                # with a synthetic response built from collected virtual parts
                _synth = _NS(
                    candidates=[_NS(content=_NS(parts=collected_parts))],
                    usage_metadata=final_usage_metadata,
                )
                return self._convert_to_chat_response(_synth, model=model)

            except LLMTimeoutError:
                raise
            except LLMError:
                raise
            except Exception as _exc:
                if partial_emitted and hooks_available:
                    await self.coordinator.hooks.emit(
                        "llm:stream_aborted",
                        {
                            "request_id": request_id,
                            "error": {
                                "type": type(_exc).__name__,
                                "msg": str(_exc),
                            },
                        },
                    )
                raise

        async def _do_complete_streaming_guarded():
            """Semaphore-gated streaming wrapper.

            Holds the semaphore for the ENTIRE stream so that concurrency
            limits apply across the full response, not just the first chunk.
            """
            global _active_requests, _waiting_requests
            sem = await _get_process_semaphore(self._max_concurrent_requests)
            if sem is not None:
                _waiting_requests += 1
                async with sem:
                    _waiting_requests -= 1
                    _active_requests += 1
                    try:
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:concurrency",
                                {
                                    "provider": "gemini",
                                    "model": model,
                                    "active_requests": _active_requests,
                                    "waiting_requests": _waiting_requests,
                                    "max_concurrent": self._max_concurrent_requests,
                                    "process_id": os.getpid(),
                                },
                            )
                        return await _do_complete_streaming()
                    finally:
                        _active_requests -= 1
            else:
                _active_requests += 1
                try:
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "provider:concurrency",
                            {
                                "provider": "gemini",
                                "model": model,
                                "active_requests": _active_requests,
                                "waiting_requests": _waiting_requests,
                                "max_concurrent": 0,
                                "process_id": os.getpid(),
                            },
                        )
                    return await _do_complete_streaming()
                finally:
                    _active_requests -= 1
        try:
            if _use_streaming:
                chat_response = await _do_complete_streaming_guarded()
            else:
                response = await retry_with_backoff(
                    _do_complete_guarded,
                    self._retry_config,
                    on_retry=_on_retry,
                )

                # Validate response structure
                if not response.candidates or len(response.candidates) == 0:
                    raise ValueError("Gemini API returned no candidates in response")

                if (
                    not hasattr(response.candidates[0], "content")
                    or not response.candidates[0].content
                ):
                    logger.error(f"Response structure: {response}")
                    logger.error(
                        f"Candidate: {response.candidates[0] if response.candidates else 'None'}"
                    )
                    raise ValueError("Gemini API response candidate has no content")

                if (
                    not hasattr(response.candidates[0].content, "parts")
                    or not response.candidates[0].content.parts
                ):
                    logger.error(f"Content: {response.candidates[0].content}")
                    raise ValueError("Gemini API response content has no parts")

                # Convert to ChatResponse first (ordering fix — emit uses converted usage)
                chat_response = self._convert_to_chat_response(response, model=model)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Emit llm:response (common to both streaming and blocking paths)
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                usage_data: dict[str, Any] = {}
                if chat_response.usage is not None:
                    usage_data = {
                        "input_tokens": chat_response.usage.input_tokens,
                        "output_tokens": chat_response.usage.output_tokens,
                    }
                    # reasoning_tokens MUST be emitted whenever the API
                    # reported it. output_tokens above is a BLENDED number
                    # (visible candidates + thinking) because that is what
                    # Google bills -- without this key, no consumer can
                    # decompose it, and a 1,360-token answer is
                    # indistinguishable from a 4-token answer with 1,356
                    # tokens of deliberation behind it. Those are very
                    # different things to a user watching cost.
                    #
                    # This payload is a hand-built dict, NOT Usage.model_dump()
                    # -- a field only reaches the event stream if it is
                    # explicitly added here. `reasoning_tokens` is the
                    # canonical Usage field name (also the name in the
                    # protobuf Usage message and in provider-openai's Usage
                    # construction), so no parallel vocabulary is introduced.
                    # Presence-gated on `is not None` to match the
                    # cache_read_tokens convention directly below: 0 is a real
                    # measurement ("thinking ran, produced nothing") and is
                    # emitted; None means the API did not report the field at
                    # all and the key is omitted.
                    if chat_response.usage.reasoning_tokens is not None:
                        usage_data["reasoning_tokens"] = (
                            chat_response.usage.reasoning_tokens
                        )
                    if chat_response.usage.cache_read_tokens is not None:
                        usage_data["cache_read_tokens"] = (
                            chat_response.usage.cache_read_tokens
                        )
                    _cost = getattr(chat_response.usage, "cost_usd", None)
                    usage_data["cost_usd"] = str(_cost) if _cost is not None else None
                    # Surface WHY cost_usd is None when it's because the model
                    # has no rate-table entry (vs. simply no usage_metadata at
                    # all). Only present when applicable -- see
                    # _convert_to_chat_response's cost_unpriced_model comment.
                    _unpriced_model = getattr(
                        chat_response.usage, "cost_unpriced_model", None
                    )
                    if _unpriced_model is not None:
                        usage_data["cost_unpriced_model"] = _unpriced_model
                response_payload: dict[str, Any] = {
                    "provider": "gemini",
                    "model": model,
                    "usage": usage_data,
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                }
                if self.raw and not _use_streaming:
                    # raw logging: only available on the blocking path
                    response_payload["raw"] = redact_secrets(
                        {
                            "content_parts": str(response.candidates[0].content.parts),
                            "raw": str(response)[:1000],
                        }
                    )
                await self.coordinator.hooks.emit("llm:response", response_payload)

            return chat_response

        except LLMError as e:
            # Kernel error types — emit llm:response error event, then propagate
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] Gemini API error: %s", error_msg)

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "gemini",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            raise

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] Gemini API error: %s", error_msg)

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "gemini",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            raise
    def _convert_to_chat_response(
        self, response, *, model: str = ""
    ) -> GeminiChatResponse:
        """
        Convert Gemini response to ChatResponse.

        Args:
            response: Gemini API response

        Returns:
            GeminiChatResponse with content blocks for UI compatibility
        """
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCallBlock

        content_blocks = []
        tool_calls = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        text_accumulator: list[str] = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                # Check if this is thinking content
                # According to Gemini API docs: parts with thought=True are thinking/reasoning
                # Parts with thought_signature (but NOT thought=True) are the final answer
                if hasattr(part, "thought") and part.thought is True:
                    # This is a thinking/reasoning part (internal reasoning process)
                    # ThinkingBlock.signature is str|None; the SDK gives raw bytes —
                    # encode to base64 before storing.
                    content_blocks.append(
                        ThinkingBlock(
                            thinking=part.text,
                            signature=_encode_sig(
                                getattr(part, "thought_signature", None)
                            ),
                            visibility="internal",
                        )
                    )
                    event_blocks.append(ThinkingContent(text=part.text))
                    # NOTE: Do NOT add thinking to text_accumulator - it's internal process, not response content

                    # Emit thinking:final event (fire-and-forget, safe if no loop)
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        # Skip event emission if no event loop running (sync context)
                        with suppress(RuntimeError):
                            asyncio.create_task(
                                self.coordinator.hooks.emit(
                                    "thinking:final", {"text": part.text}
                                )
                            )
                else:
                    # Regular text (including final answer with thought_signature)
                    # Capture any thought_signature as an extra field so the
                    # outbound path can echo it back to the API. Encode to
                    # base64 str at capture time (matching ThinkingBlock's
                    # existing behavior below) rather than storing raw SDK
                    # bytes: this module is stateless full-resend, so the
                    # captured Message list is exactly what later gets
                    # replayed -- and, in practice, also exactly what gets
                    # JSON-serialized by session persistence, event logging,
                    # or any orchestrator that calls model_dump(mode="json").
                    # Raw bytes containing non-UTF-8 sequences (the normal
                    # case for an opaque cryptographic signature) make that
                    # serialization crash outright -- verified directly
                    # against amplifier_core's own TextBlock/ToolCallBlock:
                    # model_dump(mode="json") raises UnicodeDecodeError for
                    # a signature like bytes([0xff, 0xfe, ...]). Encoding to
                    # base64 ASCII here makes the value JSON-safe everywhere
                    # it travels, matching ThinkingBlock's contract (its
                    # signature field is typed str | None for this reason).
                    _text_sig = _encode_sig(getattr(part, "thought_signature", None))
                    _text_kwargs: dict = (
                        {"signature": _text_sig} if _text_sig is not None else {}
                    )
                    content_blocks.append(TextBlock(text=part.text, **_text_kwargs))
                    if _text_sig is not None:
                        logger.debug(
                            "[PROVIDER] Gemini: captured thought_signature on text part (%d chars, base64)",
                            len(_text_sig),
                        )
                    text_accumulator.append(part.text)
                    event_blocks.append(TextContent(text=part.text))
            elif hasattr(part, "function_call"):
                # Extract tool call
                fc = part.function_call
                tool_call_id = self._generate_tool_call_id()

                # Capture thought_signature if present (Gemini 2.5+ thinking
                # models). Encoded to base64 str at capture time for the same
                # JSON-safety reason as the text-part signature above --
                # verified directly that a raw-bytes ToolCallBlock/ToolCall
                # signature fails model_dump(mode="json") for non-UTF-8 byte
                # sequences (the normal case for an opaque signature).
                _fc_sig = _encode_sig(getattr(part, "thought_signature", None))
                _fc_kwargs: dict = {"signature": _fc_sig} if _fc_sig is not None else {}
                if _fc_sig is not None:
                    logger.debug(
                        "[PROVIDER] Gemini: captured thought_signature on function_call "
                        "part '%s' (%d chars, base64)",
                        fc.name,
                        len(_fc_sig),
                    )

                # Create ToolCallBlock
                content_blocks.append(
                    ToolCallBlock(
                        id=tool_call_id,
                        name=fc.name,
                        input=dict(fc.args),  # Convert to dict
                        **_fc_kwargs,
                    )
                )

                # Create ToolCall for tool_calls list
                from amplifier_core.message_models import ToolCall as TCModel

                tool_calls.append(
                    TCModel(
                        id=tool_call_id,
                        name=fc.name,
                        arguments=dict(fc.args),
                        **_fc_kwargs,
                    )
                )
                event_blocks.append(
                    ToolCallContent(
                        id=tool_call_id, name=fc.name, arguments=dict(fc.args)
                    )
                )

        # Build metadata with usage including thought tokens
        metadata = {"raw_response": response}
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            # Gemini includes thoughtsTokenCount in usage metadata when thinking is used
            # Use getattr with defaults to handle missing fields
            input_tokens = (
                getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            )
            candidate_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )
            total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

            # Extract new usage fields (Phase 2)
            # thoughts_token_count: reasoning/thinking tokens (maps to reasoning_tokens)
            # cached_content_token_count: cached input tokens (maps to cache_read_tokens)
            # Preserve 0 as a valid measurement — 0 means "measured, none used",
            # while None means "not reported by the API".  Consistent with
            # OpenAI/vLLM providers.
            thoughts_tokens = getattr(
                response.usage_metadata, "thoughts_token_count", None
            )
            cached_tokens = getattr(
                response.usage_metadata, "cached_content_token_count", None
            )

            # cache_write_tokens: deliberately NOT populated. Verified by reading
            # the installed google-genai SDK's actual response type (not just
            # docs) — `google.genai.types.GenerateContentResponseUsageMetadata`
            # (SDK 1.46.0) exposes only: cache_tokens_details,
            # cached_content_token_count, candidates_token_count,
            # candidates_tokens_details, prompt_token_count,
            # prompt_tokens_details, thoughts_token_count,
            # tool_use_prompt_token_count, tool_use_prompt_tokens_details,
            # total_token_count, traffic_type. There is no field for tokens
            # written to cache on this call.
            #
            # This is consistent with how Gemini caching works here: this
            # provider only uses *implicit* caching (no `client.caches.create` /
            # `CachedContent` / `cached_content=` anywhere in this module —
            # confirmed by grep). Implicit caching is automatic and
            # server-managed; Google's docs describe it as billed identically
            # to a cache hit/miss on ordinary input tokens, with no separate
            # "cache write" charge or count exposed per generateContent call
            # (unlike Anthropic's cache_creation_input_tokens, or OpenAI's
            # GPT-5.6 cache_write_tokens under *explicit* prompt_cache_options).
            # `google.genai.types.CachedContentUsageMetadata` does report
            # storage-token counts, but only for the *explicit* CachedContent
            # object API this provider does not use — it is not part of
            # `GenerateContentResponseUsageMetadata` and not reachable here.
            #
            # Per the no-fabrication rule: a metric the API does not report
            # must stay None (Usage's default), never synthesized or estimated
            # from cached_content_token_count or any other field. Leaving
            # cache_write_tokens unset below is intentional, not an omission —
            # see tests/test_cache_write_tokens_not_fabricated.py for the
            # regression guard.
            #
            # output_tokens reported here is the BILLED output: visible
            # (candidate) tokens PLUS thinking tokens. Google bills thinking
            # tokens at the output rate (every thinking-capable model's
            # pricing-page entry labels its output price "(including thinking
            # tokens)") -- reporting only candidate_tokens here would silently
            # under-report output (and, via compute_cost below, cost) on any
            # turn that used reasoning. reasoning_tokens stays populated
            # separately below so the thinking/visible split remains visible
            # for observability -- this is additive, not a replacement for it.
            billed_output_tokens = candidate_tokens + (thoughts_tokens or 0)
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=billed_output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=thoughts_tokens,
                cache_read_tokens=cached_tokens,
                # cache_write_tokens intentionally omitted — see comment above.
            )

            cost = compute_cost(
                model,
                prompt_token_count=getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                or 0,
                candidates_token_count=candidate_tokens,
                cached_content_token_count=getattr(
                    response.usage_metadata, "cached_content_token_count", None
                )
                or 0,
                thoughts_token_count=thoughts_tokens or 0,
            )
            usage = usage.model_copy(update={"cost_usd": cost})
            if cost is None:
                # Distinguish "no rate data for this model" from "genuinely
                # free" (Decimal('0')). cost_usd stays None per Usage's own
                # documented contract ("None = rate data unavailable, not
                # zero"); this extra field (Usage allows extra="allow")
                # records WHICH model was unpriced so downstream consumers
                # (the llm:response event, the session.cost contributor
                # below) can surface "cost unknown" instead of a call that
                # silently contributes nothing and reads as free. See
                # mount()'s _add_cost() for the session-total counterpart.
                usage = usage.model_copy(update={"cost_unpriced_model": model})
            self._add_cost(cost, model)

        combined_text = "\n\n".join(text_accumulator).strip()

        return GeminiChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            metadata=metadata,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
        )

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from ChatResponse.

        Filters out tool calls with empty/missing arguments to handle
        Gemini API quirk where empty function_call blocks are sometimes generated.

        Args:
            response: Chat response

        Returns:
            List of valid tool calls (with non-empty arguments)
        """
        if not response.tool_calls:
            return []

        return response.tool_calls

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert Amplifier messages to Gemini format.

        Args:
            messages: Amplifier message list

        Returns:
            Tuple of (system_instruction, gemini_contents)
            - system_instruction: Combined system messages or None
            - gemini_contents: List of {role, parts} dicts

        Gemini Format:
            - system_instruction: Combined system messages
            - gemini_contents: List of content dicts with structure:
              {
                "role": "user" | "model",
                "parts": [
                  {"text": "content"},
                  {"function_call": {"name": "...", "args": {...}}},
                  {"function_response": {"name": "...", "response": {...}}}
                ]
              }
        """
        system_messages = []
        gemini_contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Extract system messages
            if role == "system":
                system_messages.append(content)
                continue

            # Convert assistant → model role with potential tool calls
            if role == "assistant":
                gemini_role = "model"
                parts = []
                content_tool_call_blocks: list[dict[str, Any]] = []

                # Add text content if present
                if content:
                    if isinstance(content, list):
                        # Content is a list of blocks - extract text blocks only
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_part: dict[str, Any] = {
                                    "text": block.get("text", "")
                                }
                                if _tsig := block.get("signature"):
                                    text_part["thought_signature"] = _encode_sig(_tsig)
                                    logger.debug(
                                        "[PROVIDER] Gemini: echoing thought_signature on text part"
                                    )
                                parts.append(text_part)
                            elif (
                                isinstance(block, dict)
                                and block.get("type") == "thinking"
                            ):
                                # Echo thinking blocks that carry a thought_signature
                                # (Gemini 2.5+ requires these to maintain reasoning context).
                                # Thinking blocks WITHOUT a signature are from older models
                                # that didn't require round-tripping — skip them as before.
                                if _tsig := block.get("signature"):
                                    thought_part: dict[str, Any] = {"thought": True}
                                    if _thinking_text := block.get("thinking"):
                                        thought_part["text"] = _thinking_text
                                    thought_part["thought_signature"] = _encode_sig(
                                        _tsig
                                    )
                                    logger.debug(
                                        "[PROVIDER] Gemini: echoing thought_signature on thinking part"
                                    )
                                    parts.append(thought_part)
                            elif (
                                isinstance(block, dict)
                                and block.get("type") == "tool_call"
                            ):
                                # Older histories retained signatures here while
                                # tool_calls kept only id, tool, and arguments.
                                content_tool_call_blocks.append(block)
                    else:
                        # Content is a simple string
                        parts.append({"text": content})

                # Handle tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tc in msg["tool_calls"]:
                        # Extract name - handle both old format (tool) and new format (name)
                        tool_name = tc.get("name") or tc.get("tool", "")

                        fc_part: dict[str, Any] = {
                            "function_call": {
                                "name": tool_name,
                                "args": tc.get("arguments", {}),
                            }
                        }
                        _tc_sig = tc.get("signature")
                        if _tc_sig is None:
                            tool_call_id = tc.get("id")
                            if tool_call_id:
                                matching_content_calls = [
                                    block
                                    for block in content_tool_call_blocks
                                    if (
                                        block.get("id") == tool_call_id
                                        and block.get("name") == tool_name
                                        and block.get("input")
                                        == tc.get("arguments", {})
                                    )
                                ]
                                if len(matching_content_calls) == 1:
                                    _tc_sig = matching_content_calls[0].get("signature")
                        # Echo thought_signature if present (Gemini 2.5+ thinking models).
                        # Omitting it causes HTTP 400 "Function call is missing a
                        # thought_signature in functionCall parts".
                        if _tc_sig:
                            fc_part["thought_signature"] = _encode_sig(_tc_sig)
                            logger.debug(
                                "[PROVIDER] Gemini: echoing thought_signature on "
                                "function_call part '%s'",
                                tool_name,
                            )
                        parts.append(fc_part)

                gemini_contents.append({"role": gemini_role, "parts": parts})

            # Handle developer messages → user with XML wrapper
            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                gemini_contents.append({"role": "user", "parts": [{"text": wrapped}]})

            # Handle tool results → function_response
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                tool_name = msg.get("name")

                if not tool_call_id:
                    logger.warning(f"Tool result missing tool_call_id: {msg}")
                    tool_call_id = "unknown"

                if not tool_name:
                    logger.debug(
                        "Tool result missing name field, recovering from function_call history"
                    )
                    # Try to find the tool name from earlier function_call in conversation
                    # Scan backwards to find matching function_call
                    for prev_msg in reversed(gemini_contents):
                        if prev_msg.get("role") == "model" and prev_msg.get("parts"):
                            for part in prev_msg["parts"]:
                                if "function_call" in part:
                                    # Found the function call - use its name
                                    tool_name = part["function_call"]["name"]
                                    logger.info(
                                        f"Recovered tool name '{tool_name}' from function_call history"
                                    )
                                    break
                        if tool_name:
                            break

                    if not tool_name:
                        logger.error(
                            f"Could not determine tool name for tool_call_id: {tool_call_id}"
                        )
                        tool_name = "unknown"

                gemini_contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": content},
                                }
                            }
                        ],
                    }
                )

            # Regular user message
            else:
                # Handle structured content (list of blocks including text and images)
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                parts.append({"text": block.get("text", "")})
                            elif block_type == "image":
                                # Convert ImageBlock to Gemini inline_data format
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": source.get(
                                                    "media_type", "image/jpeg"
                                                ),
                                                "data": source.get("data"),
                                            }
                                        }
                                    )
                                else:
                                    logger.warning(
                                        f"Unsupported image source type: {source.get('type')}"
                                    )

                    if parts:
                        gemini_contents.append({"role": "user", "parts": parts})
                else:
                    # Simple string content
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": content}]}
                    )

        # Combine system messages
        system_instruction = "\n\n".join(system_messages) if system_messages else None

        return system_instruction, gemini_contents

    def _generate_tool_call_id(self) -> str:
        """
        Generate synthetic tool call ID for Gemini.

        Gemini API doesn't provide tool call IDs, so we generate them
        to maintain compatibility with Amplifier's tool protocol.

        Returns:
            Synthetic ID in format: gemini_call_{uuid}
        """
        return f"gemini_call_{uuid.uuid4().hex[:12]}"

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """
        Convert Amplifier tools to Gemini OpenAPI schema format.

        Args:
            tools: List of tool objects with name, description, input_schema

        Returns:
            List of Gemini-formatted tool definitions
        """
        gemini_tools = []

        for tool in tools:
            # Get schema from tool if available, otherwise use empty schema
            input_schema = getattr(
                tool,
                "input_schema",
                {"type": "object", "properties": {}, "required": []},
            )

            gemini_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": input_schema,  # Gemini uses OpenAPI format directly
                }
            )

        return gemini_tools

    def _convert_tools_from_request(self, tools: list) -> list:
        """
        Convert ToolSpec objects from ChatRequest to Gemini FunctionDeclaration objects.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Gemini FunctionDeclaration objects
        """
        from google.genai import types

        gemini_tools = []

        for tool in tools:
            # Create FunctionDeclaration using parametersJsonSchema (camelCase)
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parametersJsonSchema=tool.parameters,  # Already in OpenAPI/JSON Schema format
            )
            gemini_tools.append(func_decl)

        return gemini_tools

    async def _close_client(self, client: Any) -> None:
        """Close both of a genai client's transports, off the event loop.

        `google.genai.Client` builds TWO httpx transports eagerly in its
        constructor and exposes a separate close for each:

          - ``await client.aio.aclose()`` -- the ASYNC transport. This is the
            one that actually carries this provider's traffic: every API call
            here goes through ``self.client.aio.*`` (generate_content,
            generate_content_stream, models.list). Closing only the sync
            surface would leave the real sockets to the garbage collector.
          - ``client.close()`` -- the SYNC transport (plus any authorized
            session). Synchronous, so it is offloaded with
            ``asyncio.to_thread``: calling it inline would block the event
            loop for the duration of the transport shutdown, trading a leak
            for a stall. It is still worth closing -- the sync httpx client
            is constructed eagerly whether or not this provider ever uses it.

        The two run CONCURRENTLY so a wedged async transport cannot starve
        the sync close (and vice versa) inside the single shared bound the
        caller applies. Exceptions are collected rather than raised: a close
        that fails is logged, never propagated into session teardown.

        Both closes are feature-detected. Every supported google-genai
        release (>=1.56.0, this module's floor) has both, but an SDK that
        dropped or renamed one must degrade to "close what exists" rather
        than raise AttributeError out of cleanup.
        """
        coros = []
        labels = []

        aio = getattr(client, "aio", None)
        aclose = getattr(aio, "aclose", None)
        if callable(aclose):
            coros.append(aclose())
            labels.append("async transport (client.aio.aclose)")

        sync_close = getattr(client, "close", None)
        if callable(sync_close):
            coros.append(asyncio.to_thread(sync_close))
            labels.append("sync transport (client.close)")

        if not coros:
            logger.warning(
                "[PROVIDER] Gemini: client object %s exposes neither "
                "aio.aclose() nor close() -- nothing to close, its transport "
                "is left to garbage collection.",
                type(client).__name__,
            )
            return

        results = await asyncio.gather(*coros, return_exceptions=True)
        for label, result in zip(labels, results):
            if isinstance(result, asyncio.CancelledError):
                # Abandonment path (timeout) or caller cancellation -- the
                # caller already logs it; don't double-report.
                continue
            if isinstance(result, BaseException):
                logger.warning(
                    "[PROVIDER] Gemini: closing the %s raised %s: %s -- "
                    "continuing teardown.",
                    label,
                    type(result).__name__,
                    result,
                )

    async def close(self) -> None:
        """Close the underlying genai client to prevent resource leaks.

        Before this, close() only did ``self._client = None``: the client's
        httpx transports were never closed, merely dereferenced, and their
        sockets waited on garbage collection.

        Resets ``self._client`` to ``None`` so the ``client`` property's
        lazy-init contract still holds after close(): that property only
        constructs a client when ``self._client is None``, so leaving a
        closed client in place would make every subsequent call reuse a
        closed transport and fail permanently. Clearing it lets the next use
        lazily rebuild a fresh client, and makes close() idempotent. The slot
        is cleared BEFORE anything is awaited, so a wedged client is never
        handed back out even on the timeout path where we never learn whether
        the close finished.

        The close is HARD BOUNDED at ``close_timeout`` seconds (config key;
        default 5.0). Neither `httpx.AsyncClient.aclose()` nor
        `httpx.Client.close()` has a deadline of its own: on a half-closed
        (CLOSE-WAIT) connection either can block indefinitely. Session
        cleanup runs inside the ``finally`` that PRECEDES a CLI command's
        return, so an unbounded close here does not merely leak a socket --
        it swallows the result of a completed run (recipes-8sr on the sibling
        anthropic provider: 28 minutes, process asleep in the await). On
        timeout the transports are abandoned with a WARNING naming the
        instance; the sockets are reclaimed at process exit.

        Never raises, and never blocks the event loop: the SDK's synchronous
        `Client.close()` runs in a worker thread (`asyncio.to_thread`).
        """
        client = self._client
        self._count_route_client = None
        self._client_uses_canonical_developer_route = None
        if client is None:
            # Client was never built (lazy init) -- nothing to close.
            return
        # Hand off and clear the slot FIRST: the lazy-init contract and
        # idempotency must hold even on the timeout path.
        self._client = None
        close_task = asyncio.ensure_future(self._close_client(client))
        close_task.add_done_callback(_retrieve_task_exception)
        try:
            # shield: cancelling our CALLER does not cancel a close already
            # in flight. wait_for: bounds it. Shield's outer future is a
            # plain Future, so cancelling it completes immediately -- this
            # returns within the timeout even though a wedged transport
            # ignores deadlines and cancellation alike.
            await asyncio.wait_for(
                asyncio.shield(close_task), timeout=self._close_timeout
            )
        except asyncio.CancelledError:
            # Caller cancelled: leave the shielded close running.
            pass
        except TimeoutError:
            # Request cancellation and walk away. Deliberately NOT awaited:
            # awaiting a close that ignores cancellation (a to_thread worker
            # cannot be interrupted at all) would reintroduce the unbounded
            # wait this bound exists to prevent.
            close_task.cancel()
            logger.warning(
                "[PROVIDER] Gemini client close exceeded %.1fs "
                "(half-closed/CLOSE-WAIT connection?) -- abandoning the genai "
                "client for provider instance %s id=0x%x. The sockets are "
                "reclaimed at process exit. Raise the `close_timeout` config "
                "key if this is a false alarm.",
                self._close_timeout,
                self.name,
                id(self),
            )
