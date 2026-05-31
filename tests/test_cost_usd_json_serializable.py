"""Regression tests: cost_usd is JSON-serializable in llm:response events.

The Rust hook registry calls json.dumps() on the ``llm:response`` event payload.
``Decimal`` is not JSON-serializable; Brian's fix keeps ``Decimal`` internally on
``Usage`` and converts to ``str`` only at the two emission boundaries:

  1. The ``llm:response`` event payload — ``usage_data["cost_usd"]``
  2. The ``session.cost`` contributor lambda

These tests guard against regressions at the ``llm:response`` boundary and verify
that the internal storage stays as ``Decimal``.

Covers:
  1. test_llm_response_event_is_json_serializable_known_model
  2. test_llm_response_event_cost_usd_is_str_for_known_model
  3. test_llm_response_event_cost_usd_is_none_for_unknown_model
  4. test_llm_response_event_cost_usd_round_trips_through_json
  5. test_usage_model_stores_decimal_internally
"""

import asyncio
import json
import sys
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import google
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_gemini import GeminiProvider


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_llm_response_ordering.py style)
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_gemini_response(
    prompt_token_count: int = 1_000,
    candidates_token_count: int = 200,
    cached_content_token_count: int | None = None,
):
    """Create a minimal mock Gemini API response."""
    part = SimpleNamespace(text="Hello", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage_kwargs: dict = {
        "prompt_token_count": prompt_token_count,
        "candidates_token_count": candidates_token_count,
        "total_token_count": prompt_token_count + candidates_token_count,
        "thoughts_token_count": None,
    }
    if cached_content_token_count is not None:
        usage_kwargs["cached_content_token_count"] = cached_content_token_count
    usage = SimpleNamespace(**usage_kwargs)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _make_provider(
    model: str = "gemini-2.5-flash",
) -> tuple[GeminiProvider, FakeCoordinator]:
    """Create a GeminiProvider wired to a FakeCoordinator, with a mocked client."""
    cfg = {"max_retries": 0, "default_model": model, "use_streaming": False}
    provider = GeminiProvider(api_key="test-key", config=cfg)
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    provider._client = mock_client
    return provider, coordinator


def _run_complete(provider: GeminiProvider, mock_response=None):
    """Run provider.complete() with google.genai patched out; returns the ChatResponse."""
    mock_genai = MagicMock()
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    if mock_response is not None:
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )
    with (
        patch.dict(sys.modules, {"google.genai": mock_genai}),
        patch.object(google, "genai", mock_genai, create=True),
    ):
        return asyncio.run(provider.complete(request))


# ---------------------------------------------------------------------------
# 1. JSON-serializable event for a known model
# ---------------------------------------------------------------------------


def test_llm_response_event_is_json_serializable_known_model():
    """llm:response event payload must be json.dumps-safe for a known model.

    Regression: cost_usd was stored as Decimal on Usage, which crashes the
    Rust hook registry when it calls json.dumps() on the event payload.
    """
    provider, coordinator = _make_provider(model="gemini-2.5-flash")
    mock_response = _make_gemini_response(
        prompt_token_count=1_000, candidates_token_count=200
    )

    _run_complete(provider, mock_response=mock_response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"

    # Must not raise TypeError (Decimal is not JSON serializable)
    json.dumps(response_events[0])


# ---------------------------------------------------------------------------
# 2. cost_usd in the event is str (not Decimal) for a known model
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_is_str_for_known_model():
    """cost_usd in the llm:response event must be str for a known model.

    The Rust hook registry requires JSON-safe values.  Brian's fix converts
    at the emit boundary only — internal Usage storage stays Decimal.
    """
    provider, coordinator = _make_provider(model="gemini-2.5-flash")
    mock_response = _make_gemini_response(
        prompt_token_count=1_000, candidates_token_count=200
    )

    _run_complete(provider, mock_response=mock_response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"

    cost = response_events[0].get("usage", {}).get("cost_usd")
    assert cost is not None, (
        "cost_usd should be present for known model gemini-2.5-flash"
    )
    assert isinstance(cost, str), (
        f"cost_usd in event must be str (not Decimal), got {type(cost).__name__}"
    )
    assert Decimal(cost) > 0, f"cost_usd must be positive, got {cost!r}"


# ---------------------------------------------------------------------------
# 3. cost_usd is None for an unknown model
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_is_none_for_unknown_model():
    """cost_usd in the llm:response event must be None for an unknown model.

    When the model is not in the pricing table, compute_cost returns None;
    the event must reflect that rather than raising or defaulting to 0.
    """
    provider, coordinator = _make_provider(model="gemini-unknown-model-9999")
    mock_response = _make_gemini_response(
        prompt_token_count=1_000, candidates_token_count=200
    )

    _run_complete(provider, mock_response=mock_response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"

    cost = response_events[0].get("usage", {}).get("cost_usd")
    assert cost is None, f"cost_usd should be None for unknown model, got {cost!r}"


# ---------------------------------------------------------------------------
# 4. cost_usd round-trips through json.dumps / json.loads
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_round_trips_through_json():
    """cost_usd survives json.dumps() + json.loads() without data loss.

    The value must serialize cleanly and the restored string must still parse
    to the same Decimal value.
    """
    provider, coordinator = _make_provider(model="gemini-2.5-flash")
    mock_response = _make_gemini_response(
        prompt_token_count=1_000, candidates_token_count=200
    )

    _run_complete(provider, mock_response=mock_response)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1

    payload = response_events[0]
    serialized = json.dumps(payload)
    restored = json.loads(serialized)

    cost_original = payload.get("usage", {}).get("cost_usd")
    cost_restored = restored.get("usage", {}).get("cost_usd")
    assert cost_original == cost_restored, (
        f"cost_usd must survive round-trip: {cost_original!r} → {cost_restored!r}"
    )
    # Verify it parses as a valid, positive Decimal
    assert Decimal(cost_restored) > 0


# ---------------------------------------------------------------------------
# 5. Usage model stores Decimal internally (fix is at boundary, not storage)
# ---------------------------------------------------------------------------


def test_usage_model_stores_decimal_internally():
    """Usage.cost_usd must remain Decimal after _convert_to_chat_response.

    Brian's fix converts to str only at the emission boundaries (llm:response
    event, session.cost contributor).  The Usage model itself must keep
    cost_usd as Decimal so downstream callers retain arithmetic precision.
    """
    provider = GeminiProvider(api_key="test-key", config={"use_streaming": False})
    mock_response = _make_gemini_response(
        prompt_token_count=1_000, candidates_token_count=200
    )

    result = provider._convert_to_chat_response(mock_response, model="gemini-2.5-flash")

    assert result.usage is not None
    assert result.usage.cost_usd is not None, (
        "cost_usd should be stamped for known model gemini-2.5-flash"
    )
    assert isinstance(result.usage.cost_usd, Decimal), (
        f"Usage.cost_usd must be Decimal internally, "
        f"got {type(result.usage.cost_usd).__name__}"
    )
    assert result.usage.cost_usd > 0, (
        f"cost_usd must be positive, got {result.usage.cost_usd!r}"
    )
