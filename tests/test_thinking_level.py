"""Tests for the thinking_level retarget (replaces the old reasoning_effort ->
thinking_budget-only mapping).

Covers:
- _supported_thinking_levels() per-model-family lookups
- _clamp_thinking_level() clamping behavior + INFO logging (never silent)
- End-to-end reasoning_effort -> thinking_level mapping through complete()
  for level-supporting models (gemini-3.x)
- End-to-end reasoning_effort -> legacy thinking_budget mapping for models
  that reject thinking_level entirely (gemini-2.x)
- Explicit thinking_budget always wins and is sent ALONE (never combined
  with thinking_level)

All live-behavior claims embedded in these tests/comments were verified
against the real Google AI API on 2026-08-29 (see _THINKING_LEVEL_TABLE's
module-level comment in amplifier_module_provider_gemini/__init__.py).
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini import _clamp_thinking_level
from amplifier_module_provider_gemini import _supported_thinking_levels


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_gemini_response():
    part = SimpleNamespace(text="Hello", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=10, candidates_token_count=5, total_token_count=15
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _make_provider() -> GeminiProvider:
    provider = GeminiProvider(
        api_key="test-key", config={"max_retries": 0, "use_streaming": False}
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _capture_config(provider: GeminiProvider):
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    provider._client = mock_client
    return mock_client


def _run_complete(provider, request, **kwargs):
    mock_client = _capture_config(provider)
    asyncio.run(provider.complete(request, **kwargs))
    call_kwargs = mock_client.aio.models.generate_content.await_args
    return call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")


def _make_request(**overrides) -> ChatRequest:
    overrides.pop("model", None)  # 'model' isn't a ChatRequest field consumed
    # by the provider today -- it is passed as a complete() kwarg instead
    # (see _complete_chat_request: model = kwargs.get("model", self.default_model)).
    return ChatRequest(messages=[Message(role="user", content="Hello")], **overrides)


# ============================================================
# _supported_thinking_levels() -- per-model-family table
# ============================================================


def test_25_family_rejects_thinking_level_entirely():
    """Verified live: gemini-2.5-{flash,pro,flash-lite} 400 on thinking_level."""
    assert _supported_thinking_levels("gemini-2.5-flash") is None
    assert _supported_thinking_levels("gemini-2.5-pro") is None
    assert _supported_thinking_levels("gemini-2.5-flash-lite") is None


def test_20_family_rejects_thinking_level_by_prefix_fallback():
    assert _supported_thinking_levels("gemini-2.0-flash") is None
    assert _supported_thinking_levels("gemini-2.0-flash-lite") is None
    # Unlisted 2.x id -- prefix fallback still says "no support".
    assert _supported_thinking_levels("gemini-2.9-hypothetical") is None


def test_37_flash_supports_low_medium_high_but_not_minimal():
    """Verified live: MINIMAL is explicitly rejected for gemini-3.7-flash."""
    assert _supported_thinking_levels("gemini-3.7-flash") == ("low", "medium", "high")


def test_35_family_supports_full_range_including_minimal():
    """Verified live: minimal accepted on gemini-3.5-flash(-lite)."""
    assert _supported_thinking_levels("gemini-3.5-flash") == (
        "minimal",
        "low",
        "medium",
        "high",
    )
    assert _supported_thinking_levels("gemini-3.5-flash-lite") == (
        "minimal",
        "low",
        "medium",
        "high",
    )


def test_unknown_3x_model_falls_back_to_full_range():
    assert _supported_thinking_levels("gemini-3.1-flash-lite-preview") == (
        "minimal",
        "low",
        "medium",
        "high",
    )


def test_unknown_family_defaults_to_full_range():
    """A future gemini-4.x (or anything outside 2.x/3.x) is assumed to support
    thinking_level -- a live 400 will surface clearly rather than degrading
    silently."""
    assert _supported_thinking_levels("gemini-4.0-flash") == (
        "minimal",
        "low",
        "medium",
        "high",
    )


# ============================================================
# _clamp_thinking_level() -- clamping + logging
# ============================================================


def test_clamp_noop_when_already_supported():
    assert _clamp_thinking_level("gemini-3.7-flash", "low", ("low", "medium", "high")) == "low"


def test_clamp_up_when_requested_too_low(caplog):
    """minimal isn't supported on 3.7-flash -- clamps UP to low, logs INFO."""
    with caplog.at_level("INFO"):
        result = _clamp_thinking_level(
            "gemini-3.7-flash", "minimal", ("low", "medium", "high")
        )
    assert result == "low"
    assert any(
        "clamped up to 'low'" in rec.message
        and "gemini-3.7-flash" in rec.message
        for rec in caplog.records
    ), f"expected an INFO clamp log, got: {[r.message for r in caplog.records]}"


def test_clamp_down_when_no_higher_option(caplog):
    """A hypothetical model that ONLY supports minimal/low -- requesting
    high must clamp DOWN since there's nothing higher available."""
    with caplog.at_level("INFO"):
        result = _clamp_thinking_level("hypothetical-model", "high", ("minimal", "low"))
    assert result == "low"
    assert any("clamped down to 'low'" in rec.message for rec in caplog.records)


# ============================================================
# End-to-end: reasoning_effort -> thinking_level (level-supporting model)
# ============================================================


@pytest.mark.parametrize(
    "effort,expected_level",
    [
        ("low", "LOW"),
        ("medium", "MEDIUM"),
        ("high", "HIGH"),
        ("xhigh", "HIGH"),
        ("max", "HIGH"),
    ],
)
def test_reasoning_effort_maps_to_thinking_level_on_37_flash(effort, expected_level):
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=effort,
    )
    config = _run_complete(provider, request, model="gemini-3.7-flash")
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level is not None
    assert config.thinking_config.thinking_level.value == expected_level


def test_reasoning_effort_minimal_clamped_to_low_on_37_flash(caplog):
    """37-flash doesn't support minimal -- must clamp up to low, and log it."""
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="minimal",
    )
    with caplog.at_level("INFO"):
        config = _run_complete(provider, request, model="gemini-3.7-flash")
    assert config.thinking_config.thinking_level.value == "LOW"
    assert any("clamped up to 'low'" in rec.message for rec in caplog.records)


def test_reasoning_effort_none_on_37_flash_omits_thinking_directive(caplog):
    """37-flash has no level below its default and can't disable thinking at
    all -- 'none' falls through to the model's own default (no explicit
    budget or level sent), with an INFO note explaining why."""
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="none",
    )
    with caplog.at_level("INFO"):
        config = _run_complete(provider, request, model="gemini-3.7-flash")
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level is None
    assert any("using the model's default thinking amount" in rec.message for rec in caplog.records)


def test_reasoning_effort_none_on_35_flash_uses_minimal():
    """35-flash DOES support minimal, so 'none' maps to it directly."""
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="none",
    )
    config = _run_complete(provider, request, model="gemini-3.5-flash")
    assert config.thinking_config.thinking_level.value == "MINIMAL"


# ============================================================
# End-to-end: reasoning_effort -> legacy thinking_budget (2.x models)
# ============================================================


@pytest.mark.parametrize(
    "effort,expected_budget",
    [
        ("none", 0),
        ("minimal", 4096),
        ("low", 4096),
        ("medium", -1),
        ("high", -1),
        ("xhigh", -1),
        ("max", -1),
    ],
)
def test_reasoning_effort_legacy_mapping_on_25_flash(effort, expected_budget):
    """gemini-2.5-flash rejects thinking_level entirely -- must use the
    legacy numeric budget mapping instead, and NEVER send thinking_level."""
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=effort,
    )
    config = _run_complete(provider, request, model="gemini-2.5-flash")
    assert config.thinking_config.thinking_budget == expected_budget
    assert config.thinking_config.thinking_level is None


# ============================================================
# Never send both thinking_budget and thinking_level
# ============================================================


def test_explicit_thinking_budget_wins_alone_even_on_level_supporting_model():
    """Explicit thinking_budget (legacy override) takes absolute precedence
    over reasoning_effort and is sent ALONE -- never combined with
    thinking_level (Google 400s if both are present)."""
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="high",  # would otherwise map to thinking_level=HIGH
        model="gemini-3.7-flash",
    )
    config = _run_complete(provider, request, thinking_budget=2048)
    assert config.thinking_config.thinking_budget == 2048
    assert config.thinking_config.thinking_level is None


def test_metadata_thinking_budget_wins_alone_on_level_supporting_model():
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="high",
        model="gemini-3.7-flash",
        metadata={"thinking_budget": 1024},
    )
    config = _run_complete(provider, request)
    assert config.thinking_config.thinking_budget == 1024
    assert config.thinking_config.thinking_level is None
