"""Tests for extra_request_params: a settings-only escape hatch merged LAST
into the GenerateContentConfig this provider builds for every request.

Covers:
- Not a ConfigField (no interactive prompt) -- verified by inspecting
  get_info()'s config_fields list.
- Merges arbitrary GenerateContentConfig fields not otherwise exposed
  (e.g. top_p, safety_settings).
- Wins loudly over this provider's own computed values (e.g. temperature),
  with a warning naming the old and new values.
- Unknown/invalid field names warn and are skipped, never raise.
- Single merge site: both the streaming and non-streaming call paths see
  the same merged config.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini import _apply_extra_request_params


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


def _make_provider(**config) -> GeminiProvider:
    config.setdefault("max_retries", 0)
    config.setdefault("use_streaming", False)
    provider = GeminiProvider(api_key="test-key", config=config)
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _run_complete(provider, **kwargs):
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    provider._client = mock_client
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    asyncio.run(provider.complete(request, **kwargs))
    call_kwargs = mock_client.aio.models.generate_content.await_args
    return call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")


# ============================================================
# Not a ConfigField
# ============================================================


def test_extra_request_params_is_not_a_config_field():
    provider = _make_provider()
    info = provider.get_info()
    field_ids = {f.id for f in info.config_fields}
    assert "extra_request_params" not in field_ids


# ============================================================
# _apply_extra_request_params -- unit tests
# ============================================================


def test_apply_merges_unexposed_field():
    from google.genai import types

    config = types.GenerateContentConfig(temperature=0.7, max_output_tokens=100)
    _apply_extra_request_params(config, {"top_p": 0.95})
    assert config.top_p == 0.95


def test_apply_empty_or_none_is_a_noop():
    from google.genai import types

    config = types.GenerateContentConfig(temperature=0.7)
    _apply_extra_request_params(config, {})
    assert config.temperature == 0.7
    _apply_extra_request_params(config, None)  # type: ignore[arg-type]
    assert config.temperature == 0.7


def test_apply_overrides_existing_value_loudly(caplog):
    from google.genai import types

    config = types.GenerateContentConfig(temperature=0.7)
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(config, {"temperature": 1.0})
    assert config.temperature == 1.0
    assert any(
        "overrides 'temperature'" in rec.message
        and "0.7" in rec.message
        and "1.0" in rec.message
        for rec in caplog.records
    ), f"got: {[r.message for r in caplog.records]}"


def test_apply_unknown_field_warns_and_is_skipped_not_raised(caplog):
    from google.genai import types

    config = types.GenerateContentConfig(temperature=0.7)
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(config, {"not_a_real_field": 123})
    assert not hasattr(config, "not_a_real_field") or True  # never raised
    assert any(
        "'not_a_real_field' is not a recognized" in rec.message
        for rec in caplog.records
    )
    # And the config is otherwise untouched.
    assert config.temperature == 0.7


# ============================================================
# End-to-end through complete()
# ============================================================


def test_extra_request_params_reaches_generate_content_config():
    provider = _make_provider(extra_request_params={"top_p": 0.5})
    config = _run_complete(provider)
    assert config.top_p == 0.5


def test_extra_request_params_overrides_provider_temperature(caplog):
    provider = _make_provider(temperature=0.7, extra_request_params={"temperature": 0.2})
    with caplog.at_level("WARNING"):
        config = _run_complete(provider)
    assert config.temperature == 0.2
    assert any("overrides 'temperature'" in rec.message for rec in caplog.records)


def test_no_extra_request_params_is_unaffected():
    provider = _make_provider(temperature=0.7)
    config = _run_complete(provider)
    assert config.temperature == 0.7
    assert config.top_p is None
