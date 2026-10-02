"""A default refresh must not silently change an operator's explicit pin."""

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_core.message_models import ChatRequest, Message
from tests.test_thinking_level import _run_complete


def test_default_is_current_flash():
    assert GeminiProvider(api_key="test").default_model == "gemini-3.8-flash"


def test_explicit_pin_is_preserved():
    provider = GeminiProvider(api_key="test", config={"default_model": "gemini-3.7-flash"})
    assert provider.default_model == "gemini-3.7-flash"


def test_new_default_minimal_clamps_to_low_on_wire():
    provider = GeminiProvider(api_key="test", config={"use_streaming": False, "max_retries": 0})
    from tests.test_thinking_level import FakeCoordinator
    provider.coordinator = FakeCoordinator()
    config = _run_complete(provider, ChatRequest(
        messages=[Message(role="user", content="Hello")], reasoning_effort="minimal"
    ))
    assert config.thinking_config.thinking_level.value == "LOW"


def test_new_default_none_does_not_send_minimal():
    provider = GeminiProvider(api_key="test", config={"use_streaming": False, "max_retries": 0})
    from tests.test_thinking_level import FakeCoordinator
    provider.coordinator = FakeCoordinator()
    config = _run_complete(provider, ChatRequest(
        messages=[Message(role="user", content="Hello")], reasoning_effort="none"
    ))
    assert config.thinking_config is None or config.thinking_config.thinking_level is None