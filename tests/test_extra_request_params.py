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
import json
import warnings
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
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


def _run_complete(provider, *, streaming=False, **kwargs):
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_gemini_response()
    )
    if streaming:

        async def _stream():
            yield _make_gemini_response()

        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=_stream()
        )
    provider._client = mock_client
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    asyncio.run(provider.complete(request, **kwargs))
    method = (
        mock_client.aio.models.generate_content_stream
        if streaming
        else mock_client.aio.models.generate_content
    )
    call_kwargs = method.await_args
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
    provider = _make_provider(
        temperature=0.7, extra_request_params={"temperature": 0.2}
    )
    with caplog.at_level("WARNING"):
        config = _run_complete(provider)
    assert config.temperature == 0.2
    assert any("overrides 'temperature'" in rec.message for rec in caplog.records)


def test_no_extra_request_params_is_unaffected():
    provider = _make_provider(temperature=0.7)
    config = _run_complete(provider)
    assert config.temperature == 0.7
    assert config.top_p is None


# ============================================================
# thinking_config -- typed nested merge
# ============================================================


def test_thinking_config_mapping_composes_typed_config_without_serializer_warning(
    caplog,
):
    """The original direct setattr left a raw dict and warned on serialization."""
    from google.genai import types

    extra = {"thinking_level": "low"}
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(config, {"thinking_config": extra})
        payload = json.loads(config.model_dump_json(by_alias=True, exclude_unset=True))

    assert isinstance(config.thinking_config, types.ThinkingConfig)
    assert config.thinking_config.include_thoughts is True
    assert config.thinking_config.thinking_level is types.ThinkingLevel.LOW
    assert extra == {"thinking_level": "low"}
    assert payload["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "LOW",
    }
    assert not caplog.records


def test_thinking_config_preserves_computed_and_explicit_false(caplog):
    from google.genai import types

    computed = {"include_thoughts": False, "thinking_budget": 1024}
    config = types.GenerateContentConfig(thinking_config=computed)
    extra = {"thinkingLevel": "low"}
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(config, {"thinking_config": extra})

    assert computed == {"include_thoughts": False, "thinking_budget": 1024}
    assert extra == {"thinkingLevel": "low"}
    assert config.thinking_config.include_thoughts is False
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level is types.ThinkingLevel.LOW
    warnings = [
        record
        for record in caplog.records
        if "overrides thinking_config" in record.message
    ]
    assert len(warnings) == 1
    assert "thinking_budget" in warnings[0].message


def test_thinking_config_explicit_false_override_is_preserved(caplog):
    from google.genai import types

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(
            config, {"thinking_config": {"includeThoughts": False}}
        )

    assert config.thinking_config.include_thoughts is False
    warnings = [
        record
        for record in caplog.records
        if "overrides thinking_config" in record.message
    ]
    assert len(warnings) == 1
    assert "include_thoughts" in warnings[0].message


def test_semantically_equal_thinking_config_override_is_quiet(caplog):
    from google.genai import types

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True, thinking_level=types.ThinkingLevel.LOW
        )
    )
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(
            config, {"thinking_config": {"thinkingLevel": "low"}}
        )

    assert config.thinking_config.thinking_level is types.ThinkingLevel.LOW
    assert not caplog.records


@pytest.mark.parametrize(
    "extra",
    [{}, pytest.param(None, id="empty-sdk-config")],
)
def test_empty_thinking_config_does_not_erase_computed_defaults(extra, caplog):
    from google.genai import types

    if extra is None:
        extra = types.ThinkingConfig()
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False, thinking_budget=1024
        )
    )
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(config, {"thinking_config": extra})

    assert config.thinking_config.include_thoughts is False
    assert config.thinking_config.thinking_budget == 1024
    assert config.thinking_config.thinking_level is None
    assert not caplog.records


@pytest.mark.parametrize(
    ("extra", "field", "expected"),
    [
        ({"include_thoughts": False}, "include_thoughts", False),
        ({"includeThoughts": False}, "include_thoughts", False),
        ({"thinking_level": "low"}, "thinking_level", "LOW"),
        ({"thinkingLevel": "low"}, "thinking_level", "LOW"),
        ({"thinking_budget": 2048}, "thinking_budget", 2048),
        ({"thinkingBudget": 2048}, "thinking_budget", 2048),
    ],
)
def test_thinking_config_sdk_aliases_are_normalized(extra, field, expected):
    from google.genai import types

    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig())
    _apply_extra_request_params(config, {"thinking_config": extra})

    actual = getattr(config.thinking_config, field)
    assert actual.value == expected if field == "thinking_level" else actual == expected


def test_typed_thinking_config_composes_only_its_explicit_fields():
    from google.genai import types

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )
    override = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    _apply_extra_request_params(config, {"thinking_config": override})

    assert config.thinking_config.include_thoughts is True
    assert config.thinking_config.thinking_level is types.ThinkingLevel.LOW
    assert override.model_fields_set == {"thinking_level"}


@pytest.mark.parametrize(
    ("computed", "extra", "cleared"),
    [
        ({"thinking_budget": 1024}, {"thinking_level": "low"}, "thinking_budget"),
        ({"thinking_level": "low"}, {"thinking_budget": 1024}, "thinking_level"),
    ],
)
def test_thinking_config_switches_budget_and_level_with_one_warning(
    computed, extra, cleared, caplog
):
    from google.genai import types

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(**computed)
    )
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(config, {"thinking_config": extra})

    assert getattr(config.thinking_config, cleared) is None
    warnings = [
        record
        for record in caplog.records
        if "overrides thinking_config" in record.message
    ]
    assert len(warnings) == 1
    assert cleared in warnings[0].message


def test_thinking_config_explicit_null_clears_only_that_field(caplog):
    from google.genai import types

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True, thinking_budget=1024
        )
    )
    with caplog.at_level("WARNING"):
        _apply_extra_request_params(
            config, {"thinking_config": {"thinkingBudget": None}}
        )

    assert config.thinking_config.include_thoughts is True
    assert config.thinking_config.thinking_budget is None
    assert (
        len(
            [
                record
                for record in caplog.records
                if "overrides thinking_config" in record.message
            ]
        )
        == 1
    )


def test_top_level_null_still_deliberately_overrides_whole_thinking_config():
    from google.genai import types

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )
    _apply_extra_request_params(config, {"thinking_config": None})
    assert config.thinking_config is None


def test_invalid_thinking_config_mapping_fails_before_config_changes(caplog):
    from google.genai import types

    original = types.ThinkingConfig(include_thoughts=True)
    config = types.GenerateContentConfig(temperature=0.7, thinking_config=original)
    with caplog.at_level("WARNING"), pytest.raises(ValueError):
        _apply_extra_request_params(
            config,
            {
                "temperature": 0.2,
                "thinking_config": {"thinking_budget": "not-an-integer"},
            },
        )
    assert config.thinking_config is original
    assert config.temperature == 0.7
    assert not caplog.records


@pytest.mark.parametrize("thinking_first", [False, True])
def test_conflicting_thinking_budget_and_level_is_atomic_and_quiet(
    caplog, thinking_first
):
    from google.genai import types

    original = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
    config = types.GenerateContentConfig(temperature=0.7, thinking_config=original)
    extras = {
        "temperature": 0.2,
        "thinking_config": {"thinkingBudget": 2048, "thinkingLevel": "low"},
    }
    if thinking_first:
        extras = dict(reversed(list(extras.items())))
    with (
        caplog.at_level("WARNING"),
        pytest.raises(
            ValueError, match="cannot set both thinking_budget and thinking_level"
        ),
    ):
        _apply_extra_request_params(config, extras)
    assert config.thinking_config is original
    assert config.temperature == 0.7
    assert not caplog.records


@pytest.mark.parametrize("streaming", [False, True], ids=["nonstreaming", "streaming"])
def test_complete_serializes_typed_thinking_config_for_both_transport_paths(streaming):
    """Both SDK method mocks receive a typed config with JSON-serializable fields."""
    provider = _make_provider(
        use_streaming=streaming,
        extra_request_params={"thinking_config": {"thinking_level": "low"}},
    )
    config = _run_complete(provider, streaming=streaming)

    assert config.thinking_config.include_thoughts is True
    assert config.thinking_config.thinking_level.value == "LOW"
    assert json.loads(config.model_dump_json(by_alias=True, exclude_unset=True))[
        "thinkingConfig"
    ] == {"includeThoughts": True, "thinkingLevel": "LOW"}


@pytest.mark.asyncio
@pytest.mark.parametrize("include_thoughts", [True, False])
async def test_complete_preserves_thinking_fields_in_actual_sdk_http_body(
    caplog, include_thoughts
):
    """Real SDK serialization, intercepted before HTTP; no provider request."""
    import httpx
    from google import genai
    from google.genai import types

    sent = []

    def receive(request):
        assert request.url.host == "unit.test"
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "OK"}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            },
        )

    extra = {"thinking_level": "low"}
    if not include_thoughts:
        extra["include_thoughts"] = False
    provider = _make_provider(extra_request_params={"thinking_config": extra})
    client = genai.Client(
        api_key="unit-test-key",
        http_options=types.HttpOptions(
            base_url="https://unit.test/",
            async_client_args={"transport": httpx.MockTransport(receive)},
        ),
    )
    provider._client = client
    request = ChatRequest(messages=[Message(role="user", content="Hello")])
    try:
        with caplog.at_level("WARNING"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await provider.complete(request)
            await provider.complete(request)
        assert not caught
        assert len(sent) == 2
        for body in sent:
            # SDK versions differ in nested JSON key spelling. Check the same
            # preserved fields after normalizing the SDK's documented aliases.
            aliases = {
                "includeThoughts": "include_thoughts",
                "thinkingLevel": "thinking_level",
            }
            thinking = {
                aliases.get(key, key): value
                for key, value in body["generationConfig"]["thinkingConfig"].items()
            }
            assert thinking == {
                "include_thoughts": include_thoughts,
                "thinking_level": "LOW",
            }
        conflicts = [
            record
            for record in caplog.records
            if "overrides thinking_config" in record.message
        ]
        assert len(conflicts) == (0 if include_thoughts else 2)
    finally:
        await client.aio.aclose()
        client.close()
