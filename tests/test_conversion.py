"""Tests for message and tool conversion logic."""

from unittest.mock import MagicMock

import pytest
from amplifier_module_provider_gemini import GeminiProvider


@pytest.fixture
def mock_coordinator():
    """Create mock coordinator."""
    return MagicMock()


@pytest.fixture
def provider(mock_coordinator):
    """Create GeminiProvider instance for testing."""
    return GeminiProvider(api_key="test-key", config={}, coordinator=mock_coordinator)


class TestSystemExtraction:
    """Test system message extraction."""

    def test_single_system_message(self, provider):
        """Test extraction of single system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        system_instruction, gemini_contents = provider._convert_messages(messages)

        assert system_instruction == "You are a helpful assistant."
        assert len(gemini_contents) == 1
        assert gemini_contents[0]["role"] == "user"

    def test_multiple_system_messages(self, provider):
        """Test combining multiple system messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Hello"},
        ]

        system_instruction, gemini_contents = provider._convert_messages(messages)

        assert system_instruction == "You are a helpful assistant.\n\nAnswer concisely."
        assert len(gemini_contents) == 1

    def test_no_system_messages(self, provider):
        """Test when no system messages present."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        system_instruction, gemini_contents = provider._convert_messages(messages)

        assert system_instruction is None
        assert len(gemini_contents) == 1


class TestRoleConversion:
    """Test role mapping (assistant → model)."""

    def test_assistant_to_model_conversion(self, provider):
        """Test assistant role converts to model."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        _, gemini_contents = provider._convert_messages(messages)

        assert gemini_contents[0]["role"] == "user"
        assert gemini_contents[1]["role"] == "model"

    def test_user_role_unchanged(self, provider):
        """Test user role stays as user."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        _, gemini_contents = provider._convert_messages(messages)

        assert gemini_contents[0]["role"] == "user"


class TestDeveloperWrapping:
    """Test developer message XML wrapping."""

    def test_developer_wrapped_in_xml(self, provider):
        """Test developer messages wrapped in context_file tags."""
        messages = [
            {"role": "developer", "content": "This is context information."},
        ]

        _, gemini_contents = provider._convert_messages(messages)

        assert len(gemini_contents) == 1
        assert gemini_contents[0]["role"] == "user"
        assert gemini_contents[0]["parts"][0]["text"] == (
            "<context_file>\nThis is context information.\n</context_file>"
        )


class TestPartsStructure:
    """Test text → parts array conversion."""

    def test_text_becomes_parts_array(self, provider):
        """Test simple text converted to parts array."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        _, gemini_contents = provider._convert_messages(messages)

        assert "parts" in gemini_contents[0]
        assert isinstance(gemini_contents[0]["parts"], list)
        assert len(gemini_contents[0]["parts"]) == 1
        assert gemini_contents[0]["parts"][0] == {"text": "Hello"}

    def test_assistant_text_in_parts(self, provider):
        """Test assistant text in parts array."""
        messages = [
            {"role": "assistant", "content": "Response text"},
        ]

        _, gemini_contents = provider._convert_messages(messages)

        assert gemini_contents[0]["role"] == "model"
        assert gemini_contents[0]["parts"][0] == {"text": "Response text"}


class TestToolCallConversion:
    """Test tool call conversion."""

    def test_tool_call_to_function_call(self, provider):
        """Test tool call converts to function_call part."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "tool": "get_weather",
                        "arguments": {"location": "San Francisco"},
                    }
                ],
            }
        ]

        _, gemini_contents = provider._convert_messages(messages)

        assert len(gemini_contents) == 1
        assert gemini_contents[0]["role"] == "model"
        parts = gemini_contents[0]["parts"]
        assert len(parts) == 1
        assert "function_call" in parts[0]
        assert parts[0]["function_call"]["name"] == "get_weather"
        assert parts[0]["function_call"]["args"] == {"location": "San Francisco"}

    def test_tool_call_with_text(self, provider):
        """Test tool call with accompanying text."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check the weather.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "tool": "get_weather",
                        "arguments": {"location": "San Francisco"},
                    }
                ],
            }
        ]

        _, gemini_contents = provider._convert_messages(messages)

        parts = gemini_contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Let me check the weather."}
        assert "function_call" in parts[1]

    def test_multiple_tool_calls(self, provider):
        """Test multiple tool calls in same message."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "tool": "tool_a", "arguments": {"x": 1}},
                    {"id": "call_2", "tool": "tool_b", "arguments": {"y": 2}},
                ],
            }
        ]

        _, gemini_contents = provider._convert_messages(messages)

        parts = gemini_contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0]["function_call"]["name"] == "tool_a"
        assert parts[1]["function_call"]["name"] == "tool_b"


class TestToolResultConversion:
    """Test tool result → function_response conversion."""

    def test_tool_result_to_function_response(self, provider):
        """Test tool result converts to function_response part."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_weather",
                "content": '{"temperature": 72, "conditions": "sunny"}',
            }
        ]

        _, gemini_contents = provider._convert_messages(messages)

        assert len(gemini_contents) == 1
        assert gemini_contents[0]["role"] == "user"
        parts = gemini_contents[0]["parts"]
        assert len(parts) == 1
        assert "function_response" in parts[0]
        assert parts[0]["function_response"]["name"] == "get_weather"
        assert "result" in parts[0]["function_response"]["response"]


class TestSyntheticIdGeneration:
    """Test synthetic tool call ID generation."""

    def test_id_format(self, provider):
        """Test generated ID has correct format."""
        tool_id = provider._generate_tool_call_id()

        assert tool_id.startswith("gemini_call_")
        assert len(tool_id) == len("gemini_call_") + 12  # 12 hex chars

    def test_id_uniqueness(self, provider):
        """Test generated IDs are unique."""
        ids = {provider._generate_tool_call_id() for _ in range(100)}

        assert len(ids) == 100  # All unique


class TestMultipleMessagesConversion:
    """Test converting multiple messages."""

    def test_multi_turn_conversation(self, provider):
        """Test complete multi-turn conversation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"id": "call_1", "tool": "get_weather", "arguments": {"location": "SF"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": "Sunny, 72F",
            },
            {"role": "assistant", "content": "It's sunny and 72°F!"},
        ]

        system_instruction, gemini_contents = provider._convert_messages(messages)

        # System extracted
        assert system_instruction == "You are helpful."

        # 6 conversation messages (excluding system)
        assert len(gemini_contents) == 6

        # Check roles
        assert gemini_contents[0]["role"] == "user"
        assert gemini_contents[1]["role"] == "model"
        assert gemini_contents[2]["role"] == "user"
        assert gemini_contents[3]["role"] == "model"
        assert gemini_contents[4]["role"] == "user"  # Tool result
        assert gemini_contents[5]["role"] == "model"


class TestToolConversion:
    """Test tool specification conversion."""

    def test_convert_tools_basic(self, provider):
        """Test basic tool conversion."""
        # Mock tool object
        tool = MagicMock()
        tool.name = "get_weather"
        tool.description = "Get weather for a location"
        tool.input_schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }

        tools = [tool]
        gemini_tools = provider._convert_tools(tools)

        assert len(gemini_tools) == 1
        assert gemini_tools[0]["name"] == "get_weather"
        assert gemini_tools[0]["description"] == "Get weather for a location"
        assert gemini_tools[0]["parameters"]["type"] == "object"

    def test_convert_tools_from_request(self, provider):
        """Test ToolSpec conversion."""
        # Mock ToolSpec object
        tool_spec = MagicMock()
        tool_spec.name = "calculate"
        tool_spec.description = "Perform calculation"
        tool_spec.parameters = {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
        }

        tools = [tool_spec]
        gemini_tools = provider._convert_tools_from_request(tools)

        # Verify FunctionDeclaration objects
        assert len(gemini_tools) == 1
        assert gemini_tools[0].name == "calculate"
        assert gemini_tools[0].description == "Perform calculation"
        assert gemini_tools[0].parameters_json_schema["type"] == "object"

    def test_convert_multiple_tools(self, provider):
        """Test converting multiple tools."""
        tool1 = MagicMock()
        tool1.name = "tool_a"
        tool1.description = "Tool A"
        tool1.input_schema = {"type": "object", "properties": {}}

        tool2 = MagicMock()
        tool2.name = "tool_b"
        tool2.description = "Tool B"
        tool2.input_schema = {"type": "object", "properties": {}}

        tools = [tool1, tool2]
        gemini_tools = provider._convert_tools(tools)

        assert len(gemini_tools) == 2
        assert gemini_tools[0]["name"] == "tool_a"
        assert gemini_tools[1]["name"] == "tool_b"
