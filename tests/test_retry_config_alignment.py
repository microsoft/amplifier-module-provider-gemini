"""Tests for Gemini RetryConfig alignment with Rust field names (task-9).

Verifies that:
- RetryConfig uses `initial_delay` (not `min_delay`)
- RetryConfig uses `jitter` as bool (not float)
- No jitter bool/float compat code remains
"""

import inspect

from amplifier_module_provider_gemini import GeminiProvider


def test_retry_config_uses_initial_delay():
    """RetryConfig should be constructed with initial_delay, not min_delay."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"min_retry_delay": 2.5},
    )
    # initial_delay should match the config value
    assert provider._retry_config.initial_delay == 2.5


def test_retry_config_jitter_is_bool_true_by_default():
    """RetryConfig jitter should be a bool (True by default), not a float."""
    provider = GeminiProvider(api_key="test-key", config={})
    # Default retry_jitter should be True -> jitter stored as 0.2 internally
    # but the constructor should pass bool(True) not float(0.2)
    # With Rust RetryConfig, jitter=True produces 0.2 internally
    assert provider._retry_config.jitter == 0.2  # True -> 0.2 internal


def test_retry_config_jitter_false():
    """RetryConfig jitter=False should disable jitter."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"retry_jitter": False},
    )
    assert provider._retry_config.jitter == 0.0  # False -> 0.0


def test_retry_config_jitter_true():
    """RetryConfig jitter=True should enable jitter."""
    provider = GeminiProvider(
        api_key="test-key",
        config={"retry_jitter": True},
    )
    assert provider._retry_config.jitter == 0.2  # True -> 0.2


def test_no_jitter_compat_code_in_init():
    """The __init__ method should NOT contain jitter bool/float compat code.

    Specifically, there should be no:
    - jitter_val variable
    - isinstance(jitter_val, bool) check
    """
    source = inspect.getsource(GeminiProvider.__init__)
    assert "jitter_val" not in source, "jitter_val compat variable should be removed"
    assert (
        "isinstance" not in source
        or "jitter" not in source.split("isinstance")[1].split("\n")[0]
        if "isinstance" in source
        else True
    ), "isinstance check for jitter should be removed"


def test_no_deprecation_warnings_from_retry_config(recwarn):
    """Creating GeminiProvider should not trigger deprecation warnings from RetryConfig."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GeminiProvider(api_key="test-key", config={})
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            f"Got deprecation warnings: {[str(x.message) for x in deprecation_warnings]}"
        )
