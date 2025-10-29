"""
Gemini provider module for Amplifier.
Integrates with Google's Gemini API.
"""

import logging
import os
import uuid
from typing import Any
from typing import Optional

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from google import genai

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Gemini provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including API key

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        logger.warning("No API key found for Gemini provider")
        return None

    provider = GeminiProvider(api_key, config, coordinator)
    await coordinator.mount("providers", provider, name="gemini")
    logger.info("Mounted GeminiProvider")

    # Return cleanup function
    async def cleanup():
        # genai.Client doesn't require explicit cleanup
        pass

    return cleanup


class GeminiProvider:
    """Google Gemini API integration."""

    name = "gemini"

    def __init__(
        self, api_key: str, config: dict[str, Any] | None = None, coordinator: ModuleCoordinator | None = None
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI API key
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        self.client = genai.Client(api_key=api_key)
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model = self.config.get("default_model", "gemini-2.5-flash")
        self.max_tokens = self.config.get("max_tokens", 8192)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = self.config.get("timeout", 300.0)
        self.priority = self.config.get("priority", 100)
        self.debug = self.config.get("debug", False)

    async def complete(self, messages: list[dict] | ChatRequest, **kwargs) -> ProviderResponse | ChatResponse:
        """
        Complete a chat request.

        Args:
            messages: Either list of message dicts or ChatRequest object
            **kwargs: Additional parameters

        Returns:
            ProviderResponse or ChatResponse depending on input type
        """
        raise NotImplementedError("Chunk 3")

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """
        Parse tool calls from response.

        Args:
            response: Provider response

        Returns:
            List of tool calls
        """
        return response.tool_calls or []

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert Amplifier messages to Gemini format.

        Args:
            messages: Amplifier message list

        Returns:
            Tuple of (system_instruction, gemini_contents)
            - system_instruction: Combined system messages or None
            - gemini_contents: List of {role, parts} dicts

        Gemini Format:
            - system_instruction: Combined system messages
            - gemini_contents: List of content dicts with structure:
              {
                "role": "user" | "model",
                "parts": [
                  {"text": "content"},
                  {"function_call": {"name": "...", "args": {...}}},
                  {"function_response": {"name": "...", "response": {...}}}
                ]
              }
        """
        system_messages = []
        gemini_contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Extract system messages
            if role == "system":
                system_messages.append(content)
                continue

            # Convert assistant → model role with potential tool calls
            if role == "assistant":
                gemini_role = "model"
                parts = []

                # Add text content if present
                if content:
                    parts.append({"text": content})

                # Handle tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tc in msg["tool_calls"]:
                        # Generate synthetic ID if not present
                        tool_call_id = tc.get("id")
                        if not tool_call_id:
                            tool_call_id = self._generate_tool_call_id()

                        parts.append(
                            {
                                "function_call": {
                                    "name": tc.get("tool", ""),
                                    "args": tc.get("arguments", {}),
                                }
                            }
                        )

                gemini_contents.append({"role": gemini_role, "parts": parts})

            # Handle developer messages → user with XML wrapper
            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                gemini_contents.append({"role": "user", "parts": [{"text": wrapped}]})

            # Handle tool results → function_response
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    logger.warning(f"Tool result missing tool_call_id: {msg}")
                    tool_call_id = "unknown"

                # Extract tool name from message (may need to track this separately)
                # For now, we'll use "unknown" as Gemini requires tool name
                tool_name = msg.get("name", "unknown")

                gemini_contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": content},
                                }
                            }
                        ],
                    }
                )

            # Regular user message
            else:
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})

        # Combine system messages
        system_instruction = "\n\n".join(system_messages) if system_messages else None

        return system_instruction, gemini_contents

    def _generate_tool_call_id(self) -> str:
        """
        Generate synthetic tool call ID for Gemini.

        Gemini API doesn't provide tool call IDs, so we generate them
        to maintain compatibility with Amplifier's tool protocol.

        Returns:
            Synthetic ID in format: gemini_call_{uuid}
        """
        return f"gemini_call_{uuid.uuid4().hex[:12]}"

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """
        Convert Amplifier tools to Gemini OpenAPI schema format.

        Args:
            tools: List of tool objects with name, description, input_schema

        Returns:
            List of Gemini-formatted tool definitions
        """
        gemini_tools = []

        for tool in tools:
            # Get schema from tool if available, otherwise use empty schema
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            gemini_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": input_schema,  # Gemini uses OpenAPI format directly
                }
            )

        return gemini_tools

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """
        Convert ToolSpec objects from ChatRequest to Gemini format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Gemini-formatted tool definitions
        """
        gemini_tools = []

        for tool in tools:
            gemini_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters,  # Already in OpenAPI format
                }
            )

        return gemini_tools
