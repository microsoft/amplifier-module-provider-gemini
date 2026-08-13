"""Regression tests for two measured cost-reporting defects (Gemini provider).

Defect B -- thinking tokens excluded from output count AND from cost.
    ``_convert_to_chat_response`` previously set ``output_tokens`` from
    ``candidates_token_count`` only, and ``compute_cost`` billed only that
    field. Google bills thinking tokens (``thoughts_token_count``) at the
    OUTPUT rate. The fix bills both together while keeping
    ``reasoning_tokens`` populated separately for observability.

Defect D -- unpriced models silently report no cost at all.
    ``compute_cost`` returning ``None`` for an unrecognised model is correct
    ("rate unavailable" != "free"), but nothing downstream made that
    distinguishable from "no calls happened" -- the ``session.cost``
    contributor simply vanished. The fix stamps
    ``Usage.cost_unpriced_model`` per-call and tracks unpriced models at the
    session level so the gap is visible instead of silent.

Defect B-2 -- reasoning_tokens never reached the llm:response event.
    Found by live verification: a real reasoning-heavy call emitted
    ``{"cost_usd": ..., "input_tokens": ..., "output_tokens": ...}`` -- three
    keys, no ``reasoning_tokens``. The field WAS set on the in-memory
    ``Usage`` (and ``Usage.model_dump()`` includes it), but the event payload
    is a hand-built dict literal that never contained the key, so nothing
    downstream could decompose the now-blended ``output_tokens``.

    This was NOT a serialization drop -- see
    test_usage_model_dump_proves_this_was_never_a_serialization_bug below,
    which pins the distinction so a future reader doesn't re-diagnose it.

    The lesson for this file: asserting on the in-memory object is NOT
    evidence about the event stream. Every event-level guarantee here is
    asserted against a json.dumps()-round-tripped payload.

See _cost.py and __init__.py for the implementation; see test_cost.py for
compute_cost()-level unit tests of the same scenarios. This file covers the
integration path: _convert_to_chat_response, the llm:response event, and the
session.cost contributor registered by mount().
"""

from __future__ import annotations

import asyncio
import json
import sys
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import google
import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_gemini import GeminiProvider, mount

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self) -> None:
        self.hooks = FakeHooks()


def _make_response(
    prompt_token_count: int,
    candidates_token_count: int,
    thoughts_token_count: int | None,
    cached_content_token_count: int | None = None,
):
    """Build a minimal mock Gemini API response for _convert_to_chat_response."""
    part = SimpleNamespace(text="42", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)

    usage_kwargs: dict[str, Any] = {
        "prompt_token_count": prompt_token_count,
        "candidates_token_count": candidates_token_count,
        "total_token_count": prompt_token_count + candidates_token_count,
        "thoughts_token_count": thoughts_token_count,
    }
    if cached_content_token_count is not None:
        usage_kwargs["cached_content_token_count"] = cached_content_token_count
    usage = SimpleNamespace(**usage_kwargs)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _make_provider(model: str = "gemini-2.5-flash") -> GeminiProvider:
    provider = GeminiProvider(
        api_key="test-key",
        config={"max_retries": 0, "default_model": model, "use_streaming": False},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _run_complete(provider: GeminiProvider, mock_response) -> None:
    """Run provider.complete() with google.genai patched out."""
    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    provider._client = mock_client
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    with (
        patch.dict(sys.modules, {"google.genai": mock_genai}),
        patch.object(google, "genai", mock_genai, create=True),
    ):
        asyncio.run(provider.complete(request))


# ---------------------------------------------------------------------------
# Defect B -- measured scenario, via _convert_to_chat_response directly
# ---------------------------------------------------------------------------


def test_measured_scenario_output_tokens_include_thinking():
    """Reproduces the exact bug-report numbers end to end.

    Raw event: input_tokens=10101, output_tokens=4 (candidates),
    reasoning_tokens=228 (thoughts), model=gemini-3-flash-preview.
    Previously reported cost_usd="0.0050625"; corrected value is
    Decimal('0.0057465').
    """
    provider = _make_provider()
    response = _make_response(
        prompt_token_count=10_101,
        candidates_token_count=4,
        thoughts_token_count=228,
    )

    result = provider._convert_to_chat_response(
        response, model="gemini-3-flash-preview"
    )

    assert result.usage is not None
    # output_tokens now reflects BILLED output (candidates + thoughts), not
    # just the 4 visible tokens -- this is the fix for defect B's first half.
    assert result.usage.output_tokens == 232, (
        f"Expected billed output_tokens == 232 (4 + 228), got {result.usage.output_tokens}"
    )
    # reasoning_tokens must remain populated separately -- losing this would
    # be a regression in observability per the bug report's requirement.
    assert result.usage.reasoning_tokens == 228
    assert result.usage.input_tokens == 10_101
    # Corrected cost -- fixes the ~13.5% under-report described in the bug.
    assert result.usage.cost_usd == Decimal("0.0057465"), (
        f"Expected Decimal('0.0057465'), got {result.usage.cost_usd!r}"
    )


def test_no_thinking_activity_output_tokens_unchanged():
    """When thoughts_token_count is None (no thinking), behaviour is unchanged."""
    provider = _make_provider()
    response = _make_response(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=None,
    )

    result = provider._convert_to_chat_response(response, model="gemini-2.5-flash")

    assert result.usage is not None
    assert result.usage.output_tokens == 50
    assert result.usage.reasoning_tokens is None


def test_zero_thinking_tokens_measured_but_none_used():
    """thoughts_token_count=0 means 'measured, none used' -- adds nothing."""
    provider = _make_provider()
    response = _make_response(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=0,
    )

    result = provider._convert_to_chat_response(response, model="gemini-2.5-flash")

    assert result.usage is not None
    assert result.usage.output_tokens == 50
    assert result.usage.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# Defect B-2 -- reasoning_tokens must survive into the SERIALIZED event
#
# The three tests above assert on the in-memory Usage object. That is NOT
# evidence about the event stream -- the live verification proved exactly
# that gap. Everything below asserts against a json.dumps()-round-tripped
# event payload, which is what a downstream consumer actually receives.
# ---------------------------------------------------------------------------


def _emitted_usage_after_json(coordinator) -> dict:
    """Return the llm:response usage dict AFTER a real JSON round-trip.

    Deliberately does not read the payload dict in memory: the whole point of
    this defect is that in-memory state and on-the-wire state diverged.
    """
    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"
    restored = json.loads(json.dumps(response_events[0]))
    return restored.get("usage", {})


def test_reasoning_tokens_reaches_serialized_llm_response_event():
    """The measured live scenario, asserted on the SERIALIZED event payload.

    Live-verified numbers from the container run: 48,733 input / 1,360 billed
    output against gemini-3-flash-preview, cost 0.0284465. The observed
    payload had exactly three keys and no reasoning_tokens; this pins the
    fourth.
    """
    provider = _make_provider(model="gemini-3-flash-preview")
    coordinator = cast(FakeCoordinator, provider.coordinator)
    response = _make_response(
        prompt_token_count=48_733,
        candidates_token_count=4,
        thoughts_token_count=1_356,
    )

    _run_complete(provider, response)

    usage = _emitted_usage_after_json(coordinator)

    assert "reasoning_tokens" in usage, (
        f"reasoning_tokens missing from serialized llm:response usage; got keys {sorted(usage)}"
    )
    assert usage["reasoning_tokens"] == 1_356
    # The blended output_tokens must now be decomposable by a consumer.
    assert usage["output_tokens"] == 1_360
    assert usage["output_tokens"] - usage["reasoning_tokens"] == 4, (
        "consumer must be able to recover visible output tokens"
    )
    assert usage["input_tokens"] == 48_733
    assert usage["cost_usd"] == "0.0284465"


def test_reasoning_tokens_reaches_serialized_event_on_streaming_path():
    """Same guarantee on the STREAMING path.

    use_streaming defaults to True, so this is the path a real session
    actually takes -- the blocking-path test alone would not have caught a
    streaming-only regression.
    """
    provider = _make_provider(model="gemini-3-flash-preview")
    provider.use_streaming = True
    coordinator = cast(FakeCoordinator, provider.coordinator)

    usage_metadata = SimpleNamespace(
        prompt_token_count=48_733,
        candidates_token_count=4,
        total_token_count=50_093,
        thoughts_token_count=1_356,
        cached_content_token_count=None,
    )
    chunks = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="thinking...", thought=True)]
                    )
                )
            ]
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="42", thought=False)]
                    )
                )
            ],
            usage_metadata=usage_metadata,
        ),
    ]

    async def _gen():
        for chunk in chunks:
            yield chunk

    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())
    provider._client = mock_client

    mock_genai = MagicMock()
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    with (
        patch.dict(sys.modules, {"google.genai": mock_genai}),
        patch.object(google, "genai", mock_genai, create=True),
    ):
        asyncio.run(provider.complete(request))

    usage = _emitted_usage_after_json(coordinator)

    assert usage.get("reasoning_tokens") == 1_356, (
        f"streaming path lost reasoning_tokens; got {usage!r}"
    )
    assert usage.get("output_tokens") == 1_360
    assert usage.get("cost_usd") == "0.0284465"


def test_reasoning_tokens_zero_is_emitted_not_omitted():
    """0 is a real measurement ('thinking ran, produced nothing') -- emit it.

    Matches the cache_read_tokens convention: presence-gated on `is not None`,
    not on truthiness. Omitting a measured 0 would be indistinguishable from
    'the API never reported it'.
    """
    provider = _make_provider(model="gemini-2.5-flash")
    coordinator = cast(FakeCoordinator, provider.coordinator)
    response = _make_response(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=0,
    )

    _run_complete(provider, response)

    usage = _emitted_usage_after_json(coordinator)
    assert "reasoning_tokens" in usage
    assert usage["reasoning_tokens"] == 0


def test_reasoning_tokens_omitted_when_api_did_not_report_it():
    """None (field absent from the API response) -> key omitted, never faked."""
    provider = _make_provider(model="gemini-2.5-flash")
    coordinator = cast(FakeCoordinator, provider.coordinator)
    response = _make_response(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=None,
    )

    _run_complete(provider, response)

    usage = _emitted_usage_after_json(coordinator)
    assert "reasoning_tokens" not in usage, (
        "an unreported metric must not be fabricated as 0"
    )


def test_usage_model_dump_proves_this_was_never_a_serialization_bug():
    """Pins the diagnosis: the field was NEVER SET on the event path.

    ``Usage.model_dump()`` has always included reasoning_tokens, and the
    Decimal cost_usd serializer already made the model JSON-safe. So the
    field was not being dropped by serialization -- the llm:response payload
    is a hand-built dict literal that simply never contained the key.

    This distinction matters: a serialization drop would have implied a
    different (and broader) fix. If someone later 'fixes serialization' to
    chase this symptom, this test says the cause was elsewhere.
    """
    from amplifier_core.message_models import Usage

    usage = Usage(
        input_tokens=48_733,
        output_tokens=1_360,
        total_tokens=50_093,
        reasoning_tokens=1_356,
        cost_usd=Decimal("0.0284465"),
    )

    dumped = usage.model_dump()
    assert dumped["reasoning_tokens"] == 1_356
    # And it is JSON-safe as-is, so serialization was never the blocker.
    assert json.loads(json.dumps(dumped))["reasoning_tokens"] == 1_356


# ---------------------------------------------------------------------------
# Defect D -- unpriced model is distinguishable from free, at the Usage level
# ---------------------------------------------------------------------------


def test_usage_marks_unpriced_model_when_rate_missing():
    """Unknown model -> cost_usd stays None, but cost_unpriced_model names it.

    cost_usd's documented contract ("None = rate unavailable, not zero") is
    unchanged. The new extra field says WHY, so a caller inspecting only
    cost_usd doesn't mistake "no rate data" for "free".
    """
    provider = _make_provider()
    response = _make_response(
        prompt_token_count=1_000,
        candidates_token_count=500,
        thoughts_token_count=None,
    )

    result = provider._convert_to_chat_response(
        response, model="gemini-3.1-flash-image-preview"
    )

    assert result.usage is not None
    assert result.usage.cost_usd is None
    assert getattr(result.usage, "cost_unpriced_model", None) == (
        "gemini-3.1-flash-image-preview"
    )


def test_usage_has_no_unpriced_marker_for_known_model():
    """A known/priced model must NOT carry the cost_unpriced_model field."""
    provider = _make_provider()
    response = _make_response(
        prompt_token_count=1_000,
        candidates_token_count=500,
        thoughts_token_count=None,
    )

    result = provider._convert_to_chat_response(response, model="gemini-2.5-flash")

    assert result.usage is not None
    assert result.usage.cost_usd is not None
    assert getattr(result.usage, "cost_unpriced_model", None) is None


# ---------------------------------------------------------------------------
# Defect D -- unpriced model is distinguishable, at the llm:response event
# ---------------------------------------------------------------------------


def test_llm_response_event_carries_unpriced_model_marker():
    """llm:response usage payload includes cost_unpriced_model for unknown models."""
    provider = _make_provider(model="gemini-3.1-flash-image-preview")
    coordinator = cast(FakeCoordinator, provider.coordinator)
    response = _make_response(
        prompt_token_count=1_000,
        candidates_token_count=500,
        thoughts_token_count=None,
    )

    _run_complete(provider, response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1
    usage = response_events[0].get("usage", {})
    assert usage.get("cost_usd") is None
    assert usage.get("cost_unpriced_model") == "gemini-3.1-flash-image-preview"


def test_llm_response_event_omits_unpriced_marker_for_known_model():
    """A priced model's llm:response event must not carry cost_unpriced_model."""
    provider = _make_provider(model="gemini-2.5-flash")
    coordinator = cast(FakeCoordinator, provider.coordinator)
    response = _make_response(
        prompt_token_count=1_000,
        candidates_token_count=500,
        thoughts_token_count=None,
    )

    _run_complete(provider, response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1
    usage = response_events[0].get("usage", {})
    assert usage.get("cost_usd") is not None
    assert "cost_unpriced_model" not in usage


# ---------------------------------------------------------------------------
# Defect D -- session.cost contributor no longer vanishes silently
# ---------------------------------------------------------------------------


class _MountFakeHooks:
    """Minimal hooks registry sufficient for mount() + complete()."""

    def on(self, event: str, handler) -> None:  # pragma: no cover - unused here
        pass

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        return None


class _MountFakeCoordinator:
    """Minimal coordinator sufficient for mount(), capturing contributors."""

    def __init__(self) -> None:
        self.hooks = _MountFakeHooks()
        self.mounted: dict[str, Any] = {}
        self._contributors: dict[str, list[tuple[str, Any]]] = {}

    async def mount(self, mount_point: str, module: Any, name: str) -> None:
        self.mounted[name] = module

    def register_contributor(self, channel: str, name: str, callback) -> None:
        self._contributors.setdefault(channel, []).append((name, callback))

    def get_contributor(self, channel: str, name: str):
        for contrib_name, callback in self._contributors.get(channel, []):
            if contrib_name == name:
                return callback
        raise KeyError(f"No contributor '{name}' registered on channel '{channel}'")


@pytest.mark.asyncio
async def test_session_cost_reports_unpriced_models_instead_of_vanishing(monkeypatch):
    """A session that only calls an unpriced model must not report "no data".

    Before the fix: mount()'s _add_cost() only flipped has_data=True when
    cost was not None, so a session whose ONLY calls were to an unpriced
    model left has_data False forever -- the session.cost contributor
    returned None, exactly matching "reports no cost at all for every turn"
    from the bug report. After the fix, has_data flips True on ANY call
    (priced or not), and unpriced models are named explicitly.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    coordinator = _MountFakeCoordinator()

    cleanup = await mount(
        coordinator,  # type: ignore[arg-type]
        config={
            "default_model": "gemini-3.1-flash-image-preview",
            "use_streaming": False,
        },
    )
    assert cleanup is not None

    provider = coordinator.mounted["gemini"]
    assert isinstance(provider, GeminiProvider)

    session_cost = coordinator.get_contributor("session.cost", "provider-gemini")

    # Before any calls: no data at all -- legitimately reports None.
    assert session_cost() is None

    # Drive a single call through the real provider with an unrecognised model.
    response = _make_response(
        prompt_token_count=1_000,
        candidates_token_count=500,
        thoughts_token_count=None,
    )
    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=response)
    provider._client = mock_client
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    with (
        patch.dict(sys.modules, {"google.genai": mock_genai}),
        patch.object(google, "genai", mock_genai, create=True),
    ):
        await provider.complete(request)

    result = session_cost()
    assert result is not None, (
        "session.cost must not silently vanish after an unpriced-model call"
    )
    assert result["cost_usd"] is None
    assert result["unpriced_models"] == ["gemini-3.1-flash-image-preview"]


@pytest.mark.asyncio
async def test_session_cost_still_reports_plain_total_for_priced_calls(monkeypatch):
    """Sanity check: an all-priced session reports cost_usd with no unpriced_models key."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    coordinator = _MountFakeCoordinator()

    cleanup = await mount(
        coordinator,  # type: ignore[arg-type]
        config={"default_model": "gemini-2.5-flash", "use_streaming": False},
    )
    assert cleanup is not None

    provider = coordinator.mounted["gemini"]
    session_cost = coordinator.get_contributor("session.cost", "provider-gemini")

    response = _make_response(
        prompt_token_count=1_000,
        candidates_token_count=500,
        thoughts_token_count=None,
    )
    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=response)
    provider._client = mock_client
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    with (
        patch.dict(sys.modules, {"google.genai": mock_genai}),
        patch.object(google, "genai", mock_genai, create=True),
    ):
        await provider.complete(request)

    result = session_cost()
    assert result is not None
    assert result["cost_usd"] is not None
    assert Decimal(result["cost_usd"]) > 0
    assert "unpriced_models" not in result
