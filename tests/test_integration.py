"""
Integration tests for Gemini provider.

Tests end-to-end scenarios combining multiple features.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolSpec
from amplifier_module_provider_gemini import GeminiProvider


@pytest.fixture
def mock_coordinator():
    """Create mock coordinator with hooks."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return coordinator


@pytest.fixture
def provider(mock_coordinator):
    """Create provider with mock coordinator."""
    return GeminiProvider(
        api_key="test-key",
        config={"debug": True},
        coordinator=mock_coordinator,
    )


class TestEndToEndTextCompletion:
    """Test complete text generation flows."""

    @pytest.mark.asyncio
    async def test_simple_text_completion(self, provider):
        """Test basic text completion end-to-end."""
        # Create mock response
        mock_part = MagicMock()
        mock_part.text = "The capital of France is Paris."
        mock_part.thought = False

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 15
        mock_response.usage_metadata.total_token_count = 25

        # Mock API call
        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            # Execute
            request = ChatRequest(messages=[Message(role="user", content="What is the capital of France?")])
            response = await provider.complete(request)

            # Verify
            assert response.content[0].text == "The capital of France is Paris."
            assert response.metadata["usage"]["input_tokens"] == 10
            assert response.metadata["usage"]["output_tokens"] == 15

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, provider):
        """Test multi-turn conversation flow."""
        # Create mock response
        mock_part = MagicMock()
        mock_part.text = "Sure, I can help with that!"
        mock_part.thought = False

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 10
        mock_response.usage_metadata.total_token_count = 30

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            # Multi-turn conversation
            request = ChatRequest(
                messages=[
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi there!"),
                    Message(role="user", content="Can you help me?"),
                ]
            )
            response = await provider.complete(request)

            assert len(response.content) == 1
            assert response.content[0].text == "Sure, I can help with that!"


class TestEndToEndWithTools:
    """Test tool calling flows."""

    @pytest.mark.asyncio
    async def test_tool_call_and_response(self, provider):
        """Test tool call → tool response flow."""
        # Mock tool call response
        mock_fc = MagicMock()
        mock_fc.name = "get_weather"
        mock_fc.args = {"location": "Paris"}

        mock_part = MagicMock()
        mock_part.function_call = mock_fc
        delattr(mock_part, "text")  # No text attribute

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 20

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            # Request with tools
            request = ChatRequest(
                messages=[Message(role="user", content="What's the weather in Paris?")],
                tools=[
                    ToolSpec(
                        name="get_weather",
                        description="Get weather for a location",
                        parameters={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    )
                ],
            )
            response = await provider.complete(request)

            # Verify tool call
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "get_weather"
            assert response.tool_calls[0].arguments == {"location": "Paris"}


class TestEndToEndWithThinking:
    """Test thinking/reasoning flows."""

    @pytest.mark.asyncio
    async def test_thinking_generation(self, provider):
        """Test thinking content generation."""
        # Mock thinking response
        mock_thought_part = MagicMock()
        mock_thought_part.text = "Let me think about this carefully..."
        mock_thought_part.thought = True

        mock_text_part = MagicMock()
        mock_text_part.text = "The answer is 42."
        mock_text_part.thought = False

        mock_content = MagicMock()
        mock_content.parts = [mock_thought_part, mock_text_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 30
        mock_response.usage_metadata.total_token_count = 50

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            # Request with thinking
            request = ChatRequest(
                messages=[Message(role="user", content="Solve this complex problem")],
                metadata={"thinking_budget": -1},
            )
            response = await provider.complete(request)

            # Verify thinking and text content
            assert len(response.content) == 2
            assert response.content[0].thinking == "Let me think about this carefully..."
            assert response.content[1].text == "The answer is 42."


class TestCombinedFeatures:
    """Test combinations of features."""

    @pytest.mark.asyncio
    async def test_text_with_tools_and_thinking(self, provider):
        """Test response with text, tool call, and thinking."""
        # Mock complex response
        mock_thought = MagicMock()
        mock_thought.text = "I need to check the weather..."
        mock_thought.thought = True

        mock_text = MagicMock()
        mock_text.text = "Let me check that for you."
        mock_text.thought = False

        mock_fc = MagicMock()
        mock_fc.name = "get_weather"
        mock_fc.args = {"location": "Paris"}

        mock_tool_part = MagicMock()
        mock_tool_part.function_call = mock_fc
        delattr(mock_tool_part, "text")

        mock_content = MagicMock()
        mock_content.parts = [mock_thought, mock_text, mock_tool_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 35
        mock_response.usage_metadata.total_token_count = 60

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            request = ChatRequest(
                messages=[Message(role="user", content="What's the weather?")],
                tools=[
                    ToolSpec(
                        name="get_weather",
                        description="Get weather",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
                metadata={"thinking_budget": -1},
            )
            response = await provider.complete(request)

            # Verify all content types present
            assert len(response.content) == 3
            assert response.content[0].thinking == "I need to check the weather..."
            assert response.content[1].text == "Let me check that for you."
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "get_weather"


class TestErrorScenarios:
    """Test comprehensive error handling."""

    @pytest.mark.asyncio
    async def test_empty_response_candidates(self, provider):
        """Test handling of empty candidates list."""
        mock_response = MagicMock()
        mock_response.candidates = []

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            request = ChatRequest(messages=[Message(role="user", content="Hello")])

            with pytest.raises(ValueError, match="no candidates"):
                await provider.complete(request)

    @pytest.mark.asyncio
    async def test_missing_content_in_candidate(self, provider):
        """Test handling of missing content."""
        mock_candidate = MagicMock()
        mock_candidate.content = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            request = ChatRequest(messages=[Message(role="user", content="Hello")])

            with pytest.raises(ValueError, match="has no content"):
                await provider.complete(request)

    @pytest.mark.asyncio
    async def test_api_timeout(self):
        """Test API timeout handling."""
        import asyncio

        # Create provider with short timeout
        provider = GeminiProvider(api_key="test-key", config={"timeout": 0.1})

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            # Simulate slow call
            async def slow_call(*args, **kwargs):
                await asyncio.sleep(1)  # Longer than timeout

            mock_gen.side_effect = slow_call

            request = ChatRequest(messages=[Message(role="user", content="Hello")])

            with pytest.raises(asyncio.TimeoutError):
                await provider.complete(request)


class TestEventEmission:
    """Test complete event emission flows."""

    @pytest.mark.asyncio
    async def test_complete_event_flow(self, provider, mock_coordinator):
        """Test all events emitted during completion."""
        # Mock response
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.thought = False

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            await provider.complete(request)

            # Verify events emitted
            calls = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]
            assert "llm:request" in calls
            assert "llm:response" in calls
            assert "llm:request:debug" in calls
            assert "llm:response:debug" in calls


class TestLegacyFormat:
    """Test legacy dict format support."""

    @pytest.mark.asyncio
    async def test_legacy_dict_format(self, provider):
        """Test legacy message dict format still works."""
        # Mock response
        mock_part = MagicMock()
        mock_part.text = "Legacy response"
        mock_part.thought = False

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 10

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            # Legacy format
            messages = [{"role": "user", "content": "Hello"}]
            response = await provider.complete(messages)

            # Should return ProviderResponse
            assert response.content == "Legacy response"
            assert response.usage["input"] == 5
            assert response.usage["output"] == 5


class TestChatRequestFormat:
    """Test ChatRequest format compatibility."""

    @pytest.mark.asyncio
    async def test_chat_request_returns_chat_response(self, provider):
        """Test ChatRequest returns ChatResponse."""
        # Mock response
        mock_part = MagicMock()
        mock_part.text = "ChatResponse"
        mock_part.thought = False

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 10

        with patch.object(provider.client.models, "generate_content", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            response = await provider.complete(request)

            # Should return ChatResponse
            from amplifier_core.message_models import ChatResponse

            assert isinstance(response, ChatResponse)
            assert len(response.content) == 1
            assert response.content[0].text == "ChatResponse"
