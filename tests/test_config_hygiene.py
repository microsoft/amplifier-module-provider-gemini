"""Tests for provider config hygiene: bool/numeric coercion, unknown-key
sweep with did-you-mean, and targeted inert-key messages.

Covers the fail-before/pass-after behaviors this hardening fixes:
- A config value of the STRING "false" (as written by the app-cli wizard
  for boolean fields, or hand-edited quoted YAML) used to be silently
  truthy (`bool("false") is True`), inverting the operator's intent.
- A config value of the STRING "600" for timeout/max_tokens/temperature
  used to survive uncoerced all the way to asyncio.wait_for(timeout=...),
  failing confusingly on the first real API call instead of at mount.
- max_retries/min_retry_delay/max_retry_delay used bare int()/float() that
  raised ValueError at mount time for a bad string; now warn-and-default.
- Unknown config keys (typos) are silently ignored with no signal at all.
- The three README "ghost" keys (debug, raw_debug, debug_truncate_length)
  look like unknown keys but deserve a specific explanation, not a generic
  "did you mean" guess.
"""

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini import _parse_config_bool
from amplifier_module_provider_gemini import _parse_config_number


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


# ============================================================
# _parse_config_bool
# ============================================================


def test_bool_real_true_false_unchanged():
    assert _parse_config_bool("raw", True, False) is True
    assert _parse_config_bool("raw", False, True) is False


def test_bool_string_false_is_actually_false():
    """The exact bug: bool("false") is True in naive Python -- must not be here."""
    assert _parse_config_bool("use_streaming", "false", True) is False
    assert _parse_config_bool("use_streaming", "False", True) is False
    assert _parse_config_bool("use_streaming", " FALSE ", True) is False


def test_bool_string_true_variants():
    assert _parse_config_bool("retry_jitter", "true", False) is True
    assert _parse_config_bool("retry_jitter", "1", False) is True
    assert _parse_config_bool("retry_jitter", "yes", False) is True


def test_bool_string_false_variants():
    assert _parse_config_bool("retry_jitter", "0", True) is False
    assert _parse_config_bool("retry_jitter", "no", True) is False


def test_bool_absent_or_empty_uses_default():
    assert _parse_config_bool("raw", None, True) is True
    assert _parse_config_bool("raw", "", False) is False


def test_bool_garbage_warns_and_defaults(caplog):
    with caplog.at_level("WARNING"):
        result = _parse_config_bool("raw", "maybe", False)
    assert result is False
    assert any("invalid config 'raw'" in rec.message for rec in caplog.records)


# ============================================================
# _parse_config_number
# ============================================================


def test_number_real_values_cast():
    assert _parse_config_number("timeout", 600, 300.0, float) == 600.0
    assert _parse_config_number("max_tokens", 8192, 100, int) == 8192


def test_number_string_values_coerced():
    """The exact bug: a string timeout used to survive to asyncio.wait_for."""
    assert _parse_config_number("timeout", "600", 300.0, float) == 600.0
    assert _parse_config_number("max_retries", "5", 3, int) == 5
    assert _parse_config_number("temperature", " 0.9 ", 0.7, float) == 0.9


def test_number_absent_or_empty_uses_default():
    assert _parse_config_number("timeout", None, 300.0, float) == 300.0
    assert _parse_config_number("timeout", "", 300.0, float) == 300.0


def test_number_garbage_warns_and_defaults_never_raises(caplog):
    with caplog.at_level("WARNING"):
        result = _parse_config_number("timeout", "not-a-number", 300.0, float)
    assert result == 300.0
    assert any("invalid config 'timeout'" in rec.message for rec in caplog.records)


def test_number_bool_rejected_as_numeric(caplog):
    """bool is an int subclass in Python -- must not silently become 0/1."""
    with caplog.at_level("WARNING"):
        result = _parse_config_number("max_tokens", True, 8192, int)
    assert result == 8192


# ============================================================
# End-to-end: GeminiProvider.__init__ applies coercion
# ============================================================


def test_provider_init_coerces_string_config_values(caplog):
    provider = GeminiProvider(
        api_key="test-key",
        config={
            "timeout": "45",
            "max_tokens": "2048",
            "temperature": "0.3",
            "use_streaming": "false",
            "raw": "true",
            "max_retries": "2",
            "retry_jitter": "false",
        },
    )
    assert provider.timeout == 45.0
    assert provider.max_tokens == 2048
    assert provider.temperature == 0.3
    assert provider.use_streaming is False
    assert provider.raw is True
    assert provider._retry_config.max_retries == 2
    # RetryConfig itself coerces its bool 'jitter' constructor arg into an
    # internal jitter FACTOR (0.0 disabled / 0.2 enabled) -- that's
    # RetryConfig's own contract, unrelated to this fix. What matters here
    # is that _parse_config_bool resolved the string "false" to the real
    # Python bool False before it ever reached RetryConfig.
    assert not provider._retry_config.jitter


def test_provider_init_never_raises_on_bad_max_retries_string():
    """Pre-fix: int(self.config.get("max_retries", 5)) raised ValueError
    at mount time for a bad string. Now warns and falls back to 5."""
    provider = GeminiProvider(
        api_key="test-key", config={"max_retries": "not-a-number"}
    )
    assert provider._retry_config.max_retries == 5


# ============================================================
# Unknown-key sweep + inert-key messages
# ============================================================


def test_unknown_key_typo_suggests_close_match(caplog):
    with caplog.at_level("WARNING"):
        GeminiProvider(api_key="test-key", config={"max_toekns": 100})
    assert any(
        "unknown config key 'max_toekns'" in rec.message
        and "did you mean 'max_tokens'" in rec.message
        for rec in caplog.records
    ), f"got: {[r.message for r in caplog.records]}"


def test_unknown_key_no_close_match_gets_generic_message(caplog):
    with caplog.at_level("WARNING"):
        GeminiProvider(api_key="test-key", config={"completely_unrelated_xyz": 1})
    assert any(
        "unknown config key 'completely_unrelated_xyz'" in rec.message
        and "ignored" in rec.message
        for rec in caplog.records
    )


def test_known_inert_keys_get_specific_messages(caplog):
    with caplog.at_level("WARNING"):
        GeminiProvider(
            api_key="test-key",
            config={"debug": True, "raw_debug": True, "debug_truncate_length": 180},
        )
    messages = [rec.message for rec in caplog.records]
    assert any("'debug' is inert" in m and "never implemented" in m for m in messages)
    assert any("'raw_debug' is inert" in m for m in messages)
    assert any("'debug_truncate_length' is inert" in m for m in messages)


def test_recognized_keys_produce_no_warnings(caplog):
    with caplog.at_level("WARNING"):
        GeminiProvider(
            api_key="test-key",
            config={
                "default_model": "gemini-3.7-flash",
                "max_output_tokens": 8192,
                "temperature": 0.7,
                "timeout": 600.0,
                "priority": 50,
                "raw": False,
                "use_streaming": True,
                "max_retries": 5,
                "min_retry_delay": 1.0,
                "max_retry_delay": 60.0,
                "retry_jitter": True,
                "max_concurrent_requests": 5,
            },
        )
    assert caplog.records == []


# ============================================================
# max_tokens -> max_output_tokens rename (back-compat alias)
# ============================================================


def test_max_output_tokens_is_the_canonical_key():
    provider = GeminiProvider(api_key="test-key", config={"max_output_tokens": 4096})
    assert provider.max_tokens == 4096


def test_max_tokens_still_works_as_deprecated_alias(caplog):
    with caplog.at_level("WARNING"):
        provider = GeminiProvider(api_key="test-key", config={"max_tokens": 2048})
    assert provider.max_tokens == 2048
    assert any(
        "'max_tokens' is deprecated" in rec.message
        and "'max_output_tokens'" in rec.message
        for rec in caplog.records
    ), f"got: {[r.message for r in caplog.records]}"


def test_max_output_tokens_wins_when_both_set(caplog):
    with caplog.at_level("WARNING"):
        provider = GeminiProvider(
            api_key="test-key",
            config={"max_output_tokens": 4096, "max_tokens": 2048},
        )
    assert provider.max_tokens == 4096
    assert any(
        "are BOTH set" in rec.message and "'max_output_tokens' wins" in rec.message
        for rec in caplog.records
    )


def test_max_tokens_string_value_still_coerced_through_alias():
    provider = GeminiProvider(api_key="test-key", config={"max_tokens": "3000"})
    assert provider.max_tokens == 3000


def test_neither_key_set_uses_default():
    provider = GeminiProvider(api_key="test-key", config={})
    assert provider.max_tokens == 8192
