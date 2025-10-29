"""Tests for tool calling support in Gemini provider."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolSpec
from amplifier_module_provider_gemini import GeminiProvider


@pytest.fixture
def provider():
    """Create provider instance for testing."""
    return GeminiProvider(api_key="test-key", config={})


@pytest.fixture
def mock_client():
    """Create mock Gemini client."""
    mock = MagicMock()
    mock.aio = MagicMock()
    mock.aio.models = MagicMock()
    mock.aio.models.generate_content = AsyncMock()
    return mock


class TestToolSpecConversion:
    """Test conversion of ToolSpec to Gemini format."""

    def test_tool_spec_conversion_to_gemini(self, provider):
        """Test that ToolSpec is correctly converted to Gemini FunctionDeclaration format."""
        tools = [
            ToolSpec(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"],
                },
            )
        ]

        gemini_tools = provider._convert_tools_from_request(tools)

        # Verify FunctionDeclaration objects created
        assert len(gemini_tools) == 1
        assert gemini_tools[0].name == "get_weather"
        assert gemini_tools[0].description == "Get weather for a location"
        assert gemini_tools[0].parameters_json_schema["type"] == "object"
        assert "location" in gemini_tools[0].parameters_json_schema["properties"]

    def test_multiple_tools_conversion(self, provider):
        """Test conversion of multiple tools."""
        tools = [
            ToolSpec(name="get_weather", description="Get weather", parameters={"type": "object", "properties": {}}),
            ToolSpec(
                name="search_web",
                description="Search the web",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ]

        gemini_tools = provider._convert_tools_from_request(tools)

        # Verify FunctionDeclaration objects
        assert len(gemini_tools) == 2
        assert gemini_tools[0].name == "get_weather"
        assert gemini_tools[1].name == "search_web"


class TestFunctionCallParsing:
    """Test parsing of function calls from Gemini responses."""

    @pytest.mark.asyncio
    async def test_function_call_parsing(self, provider, mock_client):
        """Test that function calls are correctly parsed from response."""
        # Mock response with function call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]

        # Create mock part with function_call using spec
        mock_part = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()
        mock_fc.name = "get_weather"
        mock_fc.args = {"location": "Boston"}
        mock_part.function_call = mock_fc

        mock_response.candidates[0].content.parts = [mock_part]

        # Mock the client
        provider.client = mock_client
        mock_client.aio.models.generate_content.return_value = mock_response

        # Create request with tools
        request = ChatRequest(
            messages=[Message(role="user", content="What's the weather in Boston?")],
            tools=[
                ToolSpec(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"location": {"type": "string"}}},
                )
            ],
        )

        response = await provider._complete_chat_request(request)

        # Verify tool call was parsed
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Boston"}
        assert response.tool_calls[0].id.startswith("gemini_call_")

    @pytest.mark.asyncio
    async def test_function_call_with_text(self, provider, mock_client):
        """Test response with both text and function call."""
        # Mock response with text and function call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]

        # Text part
        text_part = MagicMock(spec=["text"])
        text_part.text = "Let me check the weather for you."

        # Function call part
        fc_part = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()

        mock_fc.name = "get_weather"

        mock_fc.args = {"location": "Boston"}

        fc_part.function_call = mock_fc

        mock_response.candidates[0].content.parts = [text_part, fc_part]

        provider.client = mock_client
        mock_client.aio.models.generate_content.return_value = mock_response

        request = ChatRequest(
            messages=[Message(role="user", content="What's the weather?")],
            tools=[
                ToolSpec(name="get_weather", description="Get weather", parameters={"type": "object", "properties": {}})
            ],
        )

        response = await provider._complete_chat_request(request)

        # Verify both text and tool call
        assert len(response.content) == 2
        assert response.content[0].text == "Let me check the weather for you."
        assert response.content[1].name == "get_weather"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1


class TestFunctionResponseHandling:
    """Test handling of function responses in messages."""

    def test_function_response_conversion(self, provider):
        """Test that function responses are correctly converted to Gemini format."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_123", "tool": "get_weather", "arguments": {"location": "Boston"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_weather",
                "content": '{"temperature": 72, "condition": "sunny"}',
            },
        ]

        _, gemini_messages = provider._convert_messages(messages)

        # Check assistant message with function call
        assert gemini_messages[0]["role"] == "model"
        assert len(gemini_messages[0]["parts"]) == 1
        assert gemini_messages[0]["parts"][0]["function_call"]["name"] == "get_weather"

        # Check function response
        assert gemini_messages[1]["role"] == "user"
        assert len(gemini_messages[1]["parts"]) == 1
        assert gemini_messages[1]["parts"][0]["function_response"]["name"] == "get_weather"


class TestSyntheticIdAssignment:
    """Test synthetic ID generation for tool calls."""

    def test_synthetic_id_assignment(self, provider):
        """Test that synthetic IDs are generated for all tool calls."""
        # Create mock response with function call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]

        fc_part = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()

        mock_fc.name = "get_weather"

        mock_fc.args = {"location": "Boston"}

        fc_part.function_call = mock_fc

        mock_response.candidates[0].content.parts = [fc_part]

        response = provider._convert_to_chat_response(mock_response)

        # Verify synthetic ID was generated
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id.startswith("gemini_call_")
        assert len(response.tool_calls[0].id) == len("gemini_call_") + 12  # UUID hex[:12]

    def test_unique_synthetic_ids(self, provider):
        """Test that each tool call gets a unique synthetic ID."""
        # Create mock response with multiple function calls
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]

        fc_part1 = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()

        mock_fc.name = "get_weather"

        mock_fc.args = {"location": "Boston"}

        fc_part1.function_call = mock_fc

        fc_part2 = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()

        mock_fc.name = "get_weather"

        mock_fc.args = {"location": "NYC"}

        fc_part2.function_call = mock_fc

        mock_response.candidates[0].content.parts = [fc_part1, fc_part2]

        response = provider._convert_to_chat_response(mock_response)

        # Verify unique IDs
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].id != response.tool_calls[1].id


class TestMultipleToolsInTurn:
    """Test handling of multiple tool calls in a single turn."""

    @pytest.mark.asyncio
    async def test_multiple_tools_in_turn(self, provider, mock_client):
        """Test that multiple tool calls in one turn are handled correctly."""
        # Mock response with multiple function calls
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]

        fc_part1 = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()

        mock_fc.name = "get_weather"

        mock_fc.args = {"location": "Boston"}

        fc_part1.function_call = mock_fc

        fc_part2 = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()

        mock_fc.name = "get_time"

        mock_fc.args = {"timezone": "EST"}

        fc_part2.function_call = mock_fc

        mock_response.candidates[0].content.parts = [fc_part1, fc_part2]

        provider.client = mock_client
        mock_client.aio.models.generate_content.return_value = mock_response

        request = ChatRequest(
            messages=[Message(role="user", content="What's the weather and time?")],
            tools=[
                ToolSpec(
                    name="get_weather", description="Get weather", parameters={"type": "object", "properties": {}}
                ),
                ToolSpec(name="get_time", description="Get time", parameters={"type": "object", "properties": {}}),
            ],
        )

        response = await provider._complete_chat_request(request)

        # Verify both tool calls
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[1].name == "get_time"


class TestMultiTurnWithTools:
    """Test multi-turn conversations with tools."""

    @pytest.mark.asyncio
    async def test_multi_turn_with_tools(self, provider, mock_client):
        """Test that tool calls and responses work across multiple turns."""
        # Setup: previous turn with tool call and result
        # Note: Using function role instead of tool for compatibility with Message model
        request = ChatRequest(
            messages=[
                Message(role="user", content="What's the weather in Boston?"),
                Message(role="assistant", content="Let me check that"),
                Message(role="function", tool_call_id="call_123", name="get_weather", content='{"temperature": 72}'),
                Message(role="user", content="What about NYC?"),
            ],
            tools=[
                ToolSpec(name="get_weather", description="Get weather", parameters={"type": "object", "properties": {}})
            ],
        )

        # Mock response for second query
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]

        fc_part = MagicMock(spec=["function_call"])
        mock_fc = MagicMock()

        mock_fc.name = "get_weather"

        mock_fc.args = {"location": "NYC"}

        fc_part.function_call = mock_fc

        mock_response.candidates[0].content.parts = [fc_part]

        provider.client = mock_client
        mock_client.aio.models.generate_content.return_value = mock_response

        response = await provider._complete_chat_request(request)

        # Verify the new tool call
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments["location"] == "NYC"


class TestToolCallsWithEmptyArgs:
    """Test edge case of tool calls with empty arguments (Gemini quirk)."""

    def test_tool_calls_with_empty_args(self, provider):
        """Test that tool calls with empty args are filtered out."""
        from amplifier_core import ProviderResponse
        from amplifier_core import ToolCall

        # Create response with mix of valid and empty tool calls
        response = ProviderResponse(
            content="",
            tool_calls=[
                ToolCall(id="call_1", tool="valid_tool", arguments={"param": "value"}),
                ToolCall(id="call_2", tool="empty_tool", arguments={}),
                ToolCall(id="call_3", tool="another_valid", arguments={"data": "test"}),
            ],
        )

        valid_calls = provider.parse_tool_calls(response)

        # Verify only non-empty tool calls returned
        assert len(valid_calls) == 2
        assert valid_calls[0].tool == "valid_tool"
        assert valid_calls[1].tool == "another_valid"

    def test_all_empty_tool_calls(self, provider):
        """Test handling when all tool calls have empty arguments."""
        from amplifier_core import ProviderResponse
        from amplifier_core import ToolCall

        response = ProviderResponse(
            content="",
            tool_calls=[
                ToolCall(id="call_1", tool="empty1", arguments={}),
                ToolCall(id="call_2", tool="empty2", arguments={}),
            ],
        )

        valid_calls = provider.parse_tool_calls(response)

        # Should return empty list
        assert len(valid_calls) == 0
