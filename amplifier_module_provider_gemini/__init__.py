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
        if hasattr(provider.client, "close"):
            await provider.client.close()

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
