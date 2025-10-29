"""Tests for Gemini thinking/reasoning support (Chunk 5)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import ThinkingBlock
from amplifier_module_provider_gemini import GeminiProvider


@pytest.fixture
def mock_coordinator():
    """Mock coordinator with hooks."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return coordinator


@pytest.fixture
def provider(mock_coordinator):
    """Create provider with mock coordinator."""
    return GeminiProvider(
        api_key="test-key", config={"default_model": "gemini-2.5-flash"}, coordinator=mock_coordinator
    )


class TestThinkingConfiguration:
    """Test thinking configuration in _complete_chat_request."""

    @pytest.mark.asyncio
    async def test_thinking_budget_dynamic(self, provider, monkeypatch):
        """Test thinking_budget=-1 (dynamic)."""
        # Mock the API call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Response text", thought=False)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_generate = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(provider.client.models, "generate_content", mock_generate)

        # Create request with thinking_budget in metadata
        request = ChatRequest(
            messages=[Message(role="user", content="Test prompt")],
            metadata={"thinking_budget": -1, "include_thoughts": True},
        )

        # Execute
        await provider._complete_chat_request(request)

        # Verify thinking config was passed
        call_kwargs = mock_generate.call_args[1]
        assert "config" in call_kwargs
        config = call_kwargs["config"]
        assert hasattr(config, "thinking_config")
        assert config.thinking_config.thinking_budget == -1
        assert config.thinking_config.include_thoughts is True

    @pytest.mark.asyncio
    async def test_thinking_budget_disabled(self, provider, monkeypatch):
        """Test thinking_budget=0 (disabled)."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Response text", thought=False)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_generate = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(provider.client.models, "generate_content", mock_generate)

        request = ChatRequest(messages=[Message(role="user", content="Test prompt")], metadata={"thinking_budget": 0})

        await provider._complete_chat_request(request)

        # Verify thinking config with budget=0
        call_kwargs = mock_generate.call_args[1]
        config = call_kwargs["config"]
        assert hasattr(config, "thinking_config")
        assert config.thinking_config.thinking_budget == 0

    @pytest.mark.asyncio
    async def test_thinking_budget_fixed(self, provider, monkeypatch):
        """Test thinking_budget=5000 (fixed token count)."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Response text", thought=False)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_generate = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(provider.client.models, "generate_content", mock_generate)

        request = ChatRequest(
            messages=[Message(role="user", content="Test prompt")], metadata={"thinking_budget": 5000}
        )

        await provider._complete_chat_request(request)

        # Verify fixed thinking budget
        call_kwargs = mock_generate.call_args[1]
        config = call_kwargs["config"]
        assert hasattr(config, "thinking_config")
        assert config.thinking_config.thinking_budget == 5000

    @pytest.mark.asyncio
    async def test_include_thoughts_flag(self, provider, monkeypatch):
        """Test include_thoughts=False."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Response text", thought=False)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_generate = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(provider.client.models, "generate_content", mock_generate)

        request = ChatRequest(
            messages=[Message(role="user", content="Test prompt")],
            metadata={"thinking_budget": -1, "include_thoughts": False},
        )

        await provider._complete_chat_request(request)

        # Verify include_thoughts=False
        call_kwargs = mock_generate.call_args[1]
        config = call_kwargs["config"]
        assert config.thinking_config.include_thoughts is False

    @pytest.mark.asyncio
    async def test_no_thinking_config_when_not_specified(self, provider, monkeypatch):
        """Test that thinking_config is not added when thinking_budget not specified."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Response text", thought=False)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_generate = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(provider.client.models, "generate_content", mock_generate)

        request = ChatRequest(
            messages=[Message(role="user", content="Test prompt")]
            # No metadata with thinking_budget
        )

        await provider._complete_chat_request(request)

        # Verify no thinking_config when not specified
        call_kwargs = mock_generate.call_args[1]
        config = call_kwargs["config"]
        assert not hasattr(config, "thinking_config") or config.thinking_config is None

    @pytest.mark.asyncio
    async def test_kwargs_override(self, provider, monkeypatch):
        """Test that kwargs can provide thinking parameters."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Response text", thought=False)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_generate = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(provider.client.models, "generate_content", mock_generate)

        request = ChatRequest(messages=[Message(role="user", content="Test prompt")])

        # Pass thinking_budget via kwargs
        await provider._complete_chat_request(request, thinking_budget=3000, include_thoughts=True)

        # Verify kwargs were used
        call_kwargs = mock_generate.call_args[1]
        config = call_kwargs["config"]
        assert hasattr(config, "thinking_config")
        assert config.thinking_config.thinking_budget == 3000
        assert config.thinking_config.include_thoughts is True


class TestThinkingContentParsing:
    """Test thinking content parsing in _convert_to_chat_response."""

    def test_thought_part_extraction(self, provider):
        """Test extraction of thinking parts with thought=True."""
        # Mock response with thinking content
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [
            MagicMock(text="Let me analyze this step by step...", thought=True),
            MagicMock(text="Based on my analysis, the answer is X", thought=False),
        ]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        # Convert to ChatResponse
        chat_response = provider._convert_to_chat_response(mock_response)

        # Verify content blocks
        assert len(chat_response.content) == 2
        assert isinstance(chat_response.content[0], ThinkingBlock)
        assert chat_response.content[0].thinking == "Let me analyze this step by step..."
        assert chat_response.content[0].signature is None  # Gemini doesn't have signatures

        assert isinstance(chat_response.content[1], TextBlock)
        assert chat_response.content[1].text == "Based on my analysis, the answer is X"

    def test_thinking_content_block_creation(self, provider):
        """Test ThinkingBlock creation."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Thinking content", thought=True)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        chat_response = provider._convert_to_chat_response(mock_response)

        assert len(chat_response.content) == 1
        block = chat_response.content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.thinking == "Thinking content"
        assert block.signature is None

    @pytest.mark.asyncio
    async def test_thinking_event_emission(self, provider, mock_coordinator):
        """Test that thinking:final events are emitted."""
        # Replace provider's coordinator with our mock
        provider.coordinator = mock_coordinator

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Deep analysis here", thought=True)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        # Convert response (this should emit event)
        provider._convert_to_chat_response(mock_response)

        # Give async task a moment to complete
        import asyncio

        await asyncio.sleep(0.01)

        # Verify event was emitted
        mock_coordinator.hooks.emit.assert_called_once_with("thinking:final", {"text": "Deep analysis here"})

    def test_usage_with_thinking_tokens(self, provider):
        """Test that usage metadata includes thought token counts."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [
            MagicMock(text="Thinking...", thought=True),
            MagicMock(text="Answer", thought=False),
        ]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 150
        mock_response.usage_metadata.total_token_count = 250

        chat_response = provider._convert_to_chat_response(mock_response)

        # Verify usage in metadata
        assert "usage" in chat_response.metadata
        usage = chat_response.metadata["usage"]
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 150
        assert usage["total_tokens"] == 250

    def test_mixed_content_with_thinking(self, provider):
        """Test response with thinking, text, and tool calls mixed."""
        from amplifier_core.message_models import ToolCallBlock

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]

        # Create properly structured mock parts
        thinking_part = MagicMock(text="Let me think about this...", thought=True)
        del thinking_part.function_call  # Ensure no function_call attribute

        text_part = MagicMock(text="I'll need to check the weather", thought=False)
        del text_part.function_call

        tool_call_part = MagicMock()
        del tool_call_part.text  # Tool call parts don't have text
        del tool_call_part.thought
        tool_call_part.function_call = MagicMock()
        tool_call_part.function_call.name = "get_weather"
        tool_call_part.function_call.args = {"location": "San Francisco"}

        mock_response.candidates[0].content.parts = [
            thinking_part,
            text_part,
            tool_call_part,
        ]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        chat_response = provider._convert_to_chat_response(mock_response)

        # Verify all content types
        assert len(chat_response.content) == 3
        assert isinstance(chat_response.content[0], ThinkingBlock)
        assert isinstance(chat_response.content[1], TextBlock)
        assert isinstance(chat_response.content[2], ToolCallBlock)

        # Verify tool_calls list
        assert len(chat_response.tool_calls) == 1
        assert chat_response.tool_calls[0].name == "get_weather"

    def test_no_thinking_in_regular_response(self, provider):
        """Test that regular responses without thinking work correctly."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Regular response", thought=False)]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        chat_response = provider._convert_to_chat_response(mock_response)

        # Verify no ThinkingBlocks
        assert len(chat_response.content) == 1
        assert isinstance(chat_response.content[0], TextBlock)
        assert not any(isinstance(b, ThinkingBlock) for b in chat_response.content)
