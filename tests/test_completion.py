"""Tests for text completion functionality."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_module_provider_gemini import GeminiProvider


@pytest.fixture
def mock_coordinator():
    """Create mock coordinator with hooks."""
    coordinator = MagicMock()
    coordinator.hooks = AsyncMock()
    coordinator.hooks.emit = AsyncMock()
    return coordinator


@pytest.fixture
def provider(mock_coordinator):
    """Create GeminiProvider instance for testing."""
    return GeminiProvider(api_key="test-key", config={}, coordinator=mock_coordinator)


@pytest.fixture
def mock_gemini_response():
    """Create mock Gemini API response."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Hello! How can I help you today?"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 15
    return mock_response


class TestBasicTextCompletion:
    """Test basic text completion with legacy dict format."""

    @pytest.mark.asyncio
    async def test_basic_text_completion(self, provider, mock_gemini_response):
        """Test basic text completion with simple user message."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)):
            response = await provider.complete(messages)

        assert response.content == "Hello! How can I help you today?"
        assert response.usage == {"input": 10, "output": 15}
        assert len(response.content_blocks) == 1

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, provider, mock_gemini_response):
        """Test multi-turn conversation."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"},
        ]

        with patch.object(provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)):
            response = await provider.complete(messages)

        assert response.content == "Hello! How can I help you today?"
        assert response.usage["input"] == 10

    @pytest.mark.asyncio
    async def test_system_message_handling(self, provider, mock_gemini_response):
        """Test system message is converted to system_instruction."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(
            provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)
        ) as mock_generate:
            await provider.complete(messages)

        # Verify system_instruction was set in config
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["config"].system_instruction == "You are a helpful assistant."


class TestModelSelection:
    """Test model selection and override."""

    @pytest.mark.asyncio
    async def test_model_selection_default(self, provider, mock_gemini_response):
        """Test default model is used."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(
            provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)
        ) as mock_generate:
            await provider.complete(messages)

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_model_selection_override(self, provider, mock_gemini_response):
        """Test model can be overridden via kwargs."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(
            provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)
        ) as mock_generate:
            await provider.complete(messages, model="gemini-2.5-pro")

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-pro"


class TestParameterControl:
    """Test temperature and token control."""

    @pytest.mark.asyncio
    async def test_temperature_control(self, provider, mock_gemini_response):
        """Test temperature can be controlled."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(
            provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)
        ) as mock_generate:
            await provider.complete(messages, temperature=0.5)

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["config"].temperature == 0.5

    @pytest.mark.asyncio
    async def test_max_tokens_control(self, provider, mock_gemini_response):
        """Test max_tokens can be controlled."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(
            provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)
        ) as mock_generate:
            await provider.complete(messages, max_tokens=1000)

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["config"].max_output_tokens == 1000


class TestEventEmission:
    """Test event emission for observability."""

    @pytest.mark.asyncio
    async def test_request_event_emission(self, provider, mock_gemini_response):
        """Test llm:request event is emitted."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)):
            await provider.complete(messages)

        # Check that emit was called with llm:request
        calls = [call[0][0] for call in provider.coordinator.hooks.emit.call_args_list]
        assert "llm:request" in calls

    @pytest.mark.asyncio
    async def test_response_event_emission(self, provider, mock_gemini_response):
        """Test llm:response event is emitted."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)):
            await provider.complete(messages)

        # Check that emit was called with llm:response
        calls = [call[0][0] for call in provider.coordinator.hooks.emit.call_args_list]
        assert "llm:response" in calls

    @pytest.mark.asyncio
    async def test_debug_event_emission(self, mock_coordinator, mock_gemini_response):
        """Test debug events are emitted when debug=True."""
        provider = GeminiProvider(api_key="test-key", config={"debug": True}, coordinator=mock_coordinator)
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)):
            await provider.complete(messages)

        # Check that debug events were emitted
        calls = [call[0][0] for call in provider.coordinator.hooks.emit.call_args_list]
        assert "llm:request:debug" in calls
        assert "llm:response:debug" in calls


class TestTimeoutHandling:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_coordinator):
        """Test timeout is properly handled."""
        import asyncio

        # Create provider with short timeout for testing
        provider = GeminiProvider(api_key="test-key", config={"timeout": 1.0}, coordinator=mock_coordinator)
        messages = [{"role": "user", "content": "Hello"}]

        # Mock a timeout
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)

        with (
            patch.object(provider.client.models, "generate_content", new=slow_response),
            pytest.raises(asyncio.TimeoutError),
        ):
            await provider.complete(messages)


class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.asyncio
    async def test_api_error_handling(self, provider):
        """Test API errors are properly handled and events emitted."""
        messages = [{"role": "user", "content": "Hello"}]

        # Mock an API error
        with (
            patch.object(provider.client.models, "generate_content", new=AsyncMock(side_effect=Exception("API Error"))),
            pytest.raises(Exception) as exc_info,
        ):
            await provider.complete(messages)

        assert "API Error" in str(exc_info.value)

        # Check that error event was emitted
        calls = [call[0][0] for call in provider.coordinator.hooks.emit.call_args_list]
        error_events = [call for call in calls if "llm:response" in call]
        assert len(error_events) > 0


class TestChatRequestFormat:
    """Test ChatRequest format handling."""

    @pytest.mark.asyncio
    async def test_chat_request_basic(self, provider, mock_gemini_response):
        """Test ChatRequest with basic messages."""
        request = ChatRequest(
            messages=[
                Message(role="user", content="Hello"),
            ]
        )

        with patch.object(provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)):
            response = await provider.complete(request)

        assert len(response.content) == 1
        assert response.content[0].text == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_chat_request_with_system(self, provider, mock_gemini_response):
        """Test ChatRequest with system messages."""
        request = ChatRequest(
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
            ]
        )

        with patch.object(
            provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)
        ) as mock_generate:
            await provider.complete(request)

        # Verify system_instruction was set
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["config"].system_instruction == "You are helpful."

    @pytest.mark.asyncio
    async def test_chat_request_with_developer_messages(self, provider, mock_gemini_response):
        """Test ChatRequest with developer messages wrapped in XML."""
        request = ChatRequest(
            messages=[
                Message(role="developer", content="Context information"),
                Message(role="user", content="Hello"),
            ]
        )

        with patch.object(
            provider.client.models, "generate_content", new=AsyncMock(return_value=mock_gemini_response)
        ) as mock_generate:
            await provider.complete(request)

        # Verify developer message was wrapped
        call_kwargs = mock_generate.call_args.kwargs
        messages = call_kwargs["contents"]
        assert len(messages) >= 1
        # First message should have wrapped context
        assert "<context_file>" in str(messages[0])
