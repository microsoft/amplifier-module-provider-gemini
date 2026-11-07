"""Pure selector conformance and Gemini recommendation policy (fixture IDs only)."""

from dataclasses import FrozenInstanceError

import pytest

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini._model_listing import DEFAULT_FILTERS, ModelFilters


def selected(ids, filters):
    return [item["id"] for item in ModelFilters.parse(filters).select([
        {"id": model_id} for model_id in ids
    ])]


# Common conformance vectors: explicit mappings replace, never merge defaults.
@pytest.mark.parametrize("filters,expected", [
    ({}, ["CUSTOM/Chat", "gemini-2.5-pro", "gemini-3.8-flash", "gemini-3.8-flash-2026-10-01"]),
    ({"include": []}, []),
    ({"include": ["GEMINI-*"], "exclude": ["*-PRO"]}, ["gemini-3.8-flash", "gemini-3.8-flash-2026-10-01"]),
    ({"include": ["custom/*"]}, ["CUSTOM/Chat"]),
    ({"include": ["gemini-?.8-*"]}, ["gemini-3.8-flash", "gemini-3.8-flash-2026-10-01"]),
    ({"include": ["gemini-[23].*-pro"]}, ["gemini-2.5-pro"]),
    ({"hide_dated": True}, ["CUSTOM/Chat", "gemini-2.5-pro", "gemini-3.8-flash"]),
    ({"include": ["*-2026-*"], "hide_dated": True}, ["gemini-3.8-flash-2026-10-01"]),
    ({"exclude": ["*"]}, []),
])
def test_common_conformance(filters, expected):
    ids = ["gemini-3.8-flash-2026-10-01", "gemini-2.5-pro", "CUSTOM/Chat", "gemini-3.8-flash"]
    assert selected(ids, filters) == expected


@pytest.mark.parametrize("invalid", [
    None, False, [], "gemini-*", {"include": "*"}, {"include": None},
    {"include": [1]}, {"include": [""]}, {"exclude": ["  "]},
    {"hide_dated": 1}, {"chat_only": "true"}, {"latest_per_tier": None},
    {"unknown_secret_key": "credential-must-not-be-echoed"},
])
def test_invalid_config_is_rejected_without_echoing_values(invalid):
    with pytest.raises((TypeError, ValueError)) as caught:
        GeminiProvider(api_key=None, config={"model_filters": invalid})
    assert "credential-must-not-be-echoed" not in str(caught.value)
    assert "unknown_secret_key" not in str(caught.value)


def test_metadata_and_configuration_cannot_mutate_frozen_defaults():
    filters = {"include": ["CUSTOM/*"], "exclude": ["*-old"]}
    provider = GeminiProvider(api_key=None, config={"model_filters": filters})
    filters["include"].clear()
    info = provider.get_model_list_info()
    assert info["filters_keyword"] == "filters"
    assert info["source"] == "provider-api"
    assert info["version"] == 1
    assert info["override"] == "replace"
    assert info["unfiltered"] == {}
    assert info["default_filters"]["include"] == ["CUSTOM/*"]
    info["default_filters"]["include"].append("*")
    info["filters_schema"]["properties"].clear()
    assert provider.get_model_list_info()["default_filters"]["include"] == ["CUSTOM/*"]
    assert provider.get_model_list_info()["filters_schema"]["properties"]
    with pytest.raises(FrozenInstanceError):
        provider._model_filters.chat_only = True
    default = GeminiProvider(api_key=None).get_model_list_info()["default_filters"]
    assert default == DEFAULT_FILTERS.as_dict()
    assert default["chat_only"] and default["latest_per_tier"] and default["hide_dated"]
    assert GeminiProvider(api_key=None, config={"model_filters": {}})._model_filters == ModelFilters()


@pytest.mark.parametrize("names,expected", [
    (["gemini-3.9-flash", "gemini-3.10-flash", "gemini-3.10-flash-001"], ["gemini-3.10-flash"]),
    (["gemini-3.8-flash", "gemini-3.8-flash-preview", "gemini-3.9-flash-preview"], ["gemini-3.8-flash", "gemini-3.9-flash-preview"]),
    (["gemini-4-pro-preview-01-2027", "gemini-4-pro-preview-12-2026"], ["gemini-4-pro-preview-01-2027"]),
    (["gemini-3.8-flash-2026-10-01", "gemini-3.8-flash-2026-09-30"], ["gemini-3.8-flash-2026-10-01"]),
    (["gemini-3.8-flash-001", "gemini-3.8-flash-002"], ["gemini-3.8-flash-002"]),
    (["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-pro"], ["gemini-2.5-pro", "gemini-3.5-flash-lite", "gemini-3.8-flash"]),
    (["gemini-3.8-flash", "gemini-3-pro-image-preview", "gemini-3.5-flash-lite-preview-tts", "veo-3", "text-embedding-004", "future-family"], ["gemini-3.8-flash"]),
    (["gemini-3.8-flash", "publishers/google/models/gemini-3.7-flash"], ["gemini-3.8-flash", "publishers/google/models/gemini-3.7-flash"]),
    (["gemini-3.8-flash-preview", "gemini-3.8-flash-preview-09-2026"], ["gemini-3.8-flash-preview"]),
])
def test_latest_policy(names, expected):
    assert selected(names, DEFAULT_FILTERS.as_dict()) == expected
    assert selected(names, {}) == sorted(names)


def test_explicit_mapping_has_no_hidden_family_or_snapshot_policy():
    names = ["future-family", "gemini-2.5-pro-001", "gemini-3.8-flash", "gemini-3-pro-image-preview"]
    assert selected(names, {"include": ["*"]}) == sorted(names)
    assert selected(names, {"include": ["*-001"]}) == ["gemini-2.5-pro-001"]


def test_actual_methods_override_name_assumptions():
    records = [
        {"id": "gemini-3.8-flash", "generation_methods": ["embedContent"]},
        {"id": "gemini-3.7-flash", "generation_methods": ["generateContent"]},
        {"id": "gemini-3.5-flash-lite", "generation_methods": []},
        {"id": "custom-chat", "generation_methods": ["generateContent"]},
    ]
    assert [r["id"] for r in DEFAULT_FILTERS.select(records)] == ["gemini-3.7-flash"]
    assert [r["id"] for r in ModelFilters(chat_only=True).select(records)] == ["custom-chat", "gemini-3.7-flash"]
