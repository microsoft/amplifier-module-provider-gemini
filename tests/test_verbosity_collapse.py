"""Tests for CP-V Provider Verbosity Collapse (Task 13a).

Verifies:
- `raw` flag replaces `debug` + `raw_debug` config flags
- `llm:request` and `llm:response` events get optional `raw` field instead of tiered emissions
- No `llm:request:debug`, `llm:request:raw`, `llm:response:debug`, `llm:response:raw` events emitted
"""

import asyncio
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import google
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_gemini import GeminiProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_gemini_response():
    """Create a minimal mock Gemini API response."""
    part = SimpleNamespace(text="Hello", thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _make_provider(
    config: dict | None = None,
) -> tuple[GeminiProvider, FakeCoordinator]:
    """Create a GeminiProvider with a fake coordinator and mocked client."""
    cfg = {"max_retries": 0, **(config or {})}
    provider = GeminiProvider(api_key="test-key", config=cfg)
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    provider._client = mock_client

    return provider, coordinator


def _run_complete(provider: GeminiProvider) -> None:
    """Run provider.complete() with google.genai mocked out."""
    mock_genai = MagicMock()
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    with (
        patch.dict(sys.modules, {"google.genai": mock_genai}),
        patch.object(google, "genai", mock_genai, create=True),
    ):
        asyncio.run(provider.complete(request))


# ── Config flag tests ─────────────────────────────────────────────────────────


def test_raw_flag_attribute_exists_on_provider():
    """GeminiProvider should have a `raw` attribute from config."""
    provider, _ = _make_provider(config={"raw": True})
    assert hasattr(provider, "raw"), "Provider should have 'raw' attribute"
    assert provider.raw is True  # type: ignore[attr-defined]


def test_raw_flag_defaults_to_false():
    """GeminiProvider.raw should default to False."""
    provider, _ = _make_provider()
    assert hasattr(provider, "raw"), "Provider should have 'raw' attribute"
    assert provider.raw is False  # type: ignore[attr-defined]


def test_debug_flag_removed():
    """Provider should NOT have a `debug` attribute (replaced by `raw`)."""
    provider, _ = _make_provider()
    assert not hasattr(provider, "debug"), (
        "Provider should not have 'debug' attribute — replaced by 'raw'"
    )


def test_raw_debug_flag_removed():
    """Provider should NOT have a `raw_debug` attribute (replaced by `raw`)."""
    provider, _ = _make_provider()
    assert not hasattr(provider, "raw_debug"), (
        "Provider should not have 'raw_debug' attribute — replaced by 'raw'"
    )


def test_debug_truncate_length_removed():
    """Provider should NOT have a `debug_truncate_length` attribute."""
    provider, _ = _make_provider()
    assert not hasattr(provider, "debug_truncate_length"), (
        "Provider should not have 'debug_truncate_length' attribute"
    )


# ── llm:request event tests ───────────────────────────────────────────────────


def test_llm_request_event_emitted_without_raw_field_by_default():
    """llm:request event should be emitted without a `raw` field by default."""
    provider, coordinator = _make_provider()
    _run_complete(provider)

    request_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:request"
    ]
    assert len(request_events) >= 1, "llm:request event should be emitted"
    assert "raw" not in request_events[0], (
        "llm:request event should NOT have 'raw' field when raw=False"
    )


def test_llm_request_event_has_raw_field_when_raw_true():
    """llm:request event should include a `raw` field when raw=True."""
    provider, coordinator = _make_provider(config={"raw": True})
    _run_complete(provider)

    request_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:request"
    ]
    assert len(request_events) >= 1, "llm:request event should be emitted"
    assert "raw" in request_events[0], (
        "llm:request event should have 'raw' field when raw=True"
    )


def test_llm_request_debug_event_never_emitted():
    """llm:request:debug event should NEVER be emitted (tiered events removed)."""
    provider, coordinator = _make_provider(config={"raw": True})
    _run_complete(provider)

    debug_events = [
        name for name, _ in coordinator.hooks.events if name == "llm:request:debug"
    ]
    assert len(debug_events) == 0, (
        "llm:request:debug event should NEVER be emitted (collapsed pattern)"
    )


def test_llm_request_raw_event_never_emitted():
    """llm:request:raw event should NEVER be emitted (tiered events removed)."""
    provider, coordinator = _make_provider(config={"raw": True})
    _run_complete(provider)

    raw_events = [
        name for name, _ in coordinator.hooks.events if name == "llm:request:raw"
    ]
    assert len(raw_events) == 0, (
        "llm:request:raw event should NEVER be emitted (collapsed pattern)"
    )


def test_llm_request_base_payload_fields_present():
    """llm:request event should still contain the expected base fields."""
    provider, coordinator = _make_provider()
    _run_complete(provider)

    request_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:request"
    ]
    assert len(request_events) >= 1
    payload = request_events[0]
    assert payload["provider"] == "gemini"
    assert "model" in payload
    assert "message_count" in payload


# ── llm:response event tests ──────────────────────────────────────────────────


def test_llm_response_event_emitted_without_raw_field_by_default():
    """llm:response event should be emitted without a `raw` field by default."""
    provider, coordinator = _make_provider()
    _run_complete(provider)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"
    assert "raw" not in response_events[0], (
        "llm:response event should NOT have 'raw' field when raw=False"
    )


def test_llm_response_event_has_raw_field_when_raw_true():
    """llm:response event should include a `raw` field when raw=True."""
    provider, coordinator = _make_provider(config={"raw": True})
    _run_complete(provider)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1, "llm:response event should be emitted"
    assert "raw" in response_events[0], (
        "llm:response event should have 'raw' field when raw=True"
    )


def test_llm_response_debug_event_never_emitted():
    """llm:response:debug event should NEVER be emitted (tiered events removed)."""
    provider, coordinator = _make_provider(config={"raw": True})
    _run_complete(provider)

    debug_events = [
        name for name, _ in coordinator.hooks.events if name == "llm:response:debug"
    ]
    assert len(debug_events) == 0, (
        "llm:response:debug event should NEVER be emitted (collapsed pattern)"
    )


def test_llm_response_raw_event_never_emitted():
    """llm:response:raw event should NEVER be emitted (tiered events removed)."""
    provider, coordinator = _make_provider(config={"raw": True})
    _run_complete(provider)

    raw_events = [
        name for name, _ in coordinator.hooks.events if name == "llm:response:raw"
    ]
    assert len(raw_events) == 0, (
        "llm:response:raw event should NEVER be emitted (collapsed pattern)"
    )


def test_llm_response_base_payload_fields_present():
    """llm:response event should still contain the expected base fields."""
    provider, coordinator = _make_provider()
    _run_complete(provider)

    response_events = [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]
    assert len(response_events) >= 1
    payload = response_events[0]
    assert payload["provider"] == "gemini"
    assert "model" in payload
    assert payload["status"] == "ok"
    assert "duration_ms" in payload


def test_only_two_llm_events_emitted_when_raw_false():
    """With raw=False, exactly one llm:request and one llm:response should be emitted."""
    provider, coordinator = _make_provider(config={"raw": False})
    _run_complete(provider)

    llm_events = [
        name for name, _ in coordinator.hooks.events if name.startswith("llm:")
    ]
    assert llm_events.count("llm:request") == 1
    assert llm_events.count("llm:response") == 1
    # No tiered variants
    assert "llm:request:debug" not in llm_events
    assert "llm:request:raw" not in llm_events
    assert "llm:response:debug" not in llm_events
    assert "llm:response:raw" not in llm_events


def test_only_two_llm_events_emitted_when_raw_true():
    """With raw=True, still exactly one llm:request and one llm:response (with raw field)."""
    provider, coordinator = _make_provider(config={"raw": True})
    _run_complete(provider)

    llm_events = [
        name for name, _ in coordinator.hooks.events if name.startswith("llm:")
    ]
    assert llm_events.count("llm:request") == 1
    assert llm_events.count("llm:response") == 1
    # No tiered variants even with raw=True
    assert "llm:request:debug" not in llm_events
    assert "llm:request:raw" not in llm_events
    assert "llm:response:debug" not in llm_events
    assert "llm:response:raw" not in llm_events
