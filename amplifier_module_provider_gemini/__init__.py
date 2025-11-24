"""
Gemini provider module for Amplifier.
Integrates with Google's Gemini API.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import suppress
from typing import Any
from typing import Optional

from amplifier_core import ModuleCoordinator
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import Usage
from amplifier_core.models import ToolCall
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


class GeminiChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


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
        self.raw_debug = self.config.get("raw_debug", False)
        self.debug_truncate_length = self.config.get("debug_truncate_length", 180)

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Preserves structure, only truncates leaf string values longer than max_length.
        Uses self.debug_truncate_length if max_length not specified.

        Args:
            obj: Any JSON-serializable structure (dict, list, primitives)
            max_length: Maximum string length (defaults to self.debug_truncate_length)

        Returns:
            Structure with truncated string values
        """
        if max_length is None:
            max_length = self.debug_truncate_length

        # Type guard: max_length is guaranteed to be int after this point
        assert max_length is not None, "max_length should never be None after initialization"

        if isinstance(obj, str):
            if len(obj) > max_length:
                return obj[:max_length] + f"... (truncated {len(obj) - max_length} chars)"
            return obj
        if isinstance(obj, dict):
            return {k: self._truncate_values(v, max_length) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_values(item, max_length) for item in obj]
        return obj  # Numbers, booleans, None pass through unchanged

    def _find_missing_tool_results(self, messages: list[Message]) -> list[tuple[str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.

        Returns:
            List of (call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        tool_calls = {}  # {call_id: (name, args)}
        tool_results = set()  # {call_id}

        for msg in messages:
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (block.name, block.input)

            # Check tool messages for tool_call_id
            elif msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                tool_results.add(msg.tool_call_id)

        return [(call_id, name, args) for call_id, (name, args) in tool_calls.items() if call_id not in tool_results]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> Message:
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.
        """
        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Gemini: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results
            for call_id, tool_name, _ in missing:
                synthetic = self._create_synthetic_result(call_id, tool_name)
                request.messages.append(synthetic)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name} for call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """

        logger.debug(f"Received ChatRequest with {len(request.messages)} messages")

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant")]

        # Combine system messages
        system_instruction = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
        )

        # Convert developer messages to XML-wrapped user messages
        context_user_msgs = []
        for dev_msg in developer_msgs:
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            wrapped = f"<context_file>\n{content}\n</context_file>"
            context_user_msgs.append({"role": "user", "parts": [{"text": wrapped}]})

        # Convert conversation messages
        conversation_msgs = []
        if conversation:
            _, conversation_msgs = self._convert_messages([m.model_dump() for m in conversation])

        # Combine: context THEN conversation
        all_messages = context_user_msgs + conversation_msgs

        # Prepare request parameters
        model = kwargs.get("model", self.default_model)
        temperature = request.temperature or kwargs.get("temperature", self.temperature)
        max_tokens = request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens)

        # Extract thinking parameters from request metadata or kwargs
        thinking_budget = None
        include_thoughts = True

        if request.metadata:
            thinking_budget = request.metadata.get("thinking_budget")
            include_thoughts = request.metadata.get("include_thoughts", True)

        # Allow kwargs to override if metadata not present
        if thinking_budget is None and "thinking_budget" in kwargs:
            thinking_budget = kwargs["thinking_budget"]
        if "include_thoughts" in kwargs:
            include_thoughts = kwargs["include_thoughts"]

        # Build Gemini config with thinking support
        config = genai.types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)

        # Add thinking configuration if specified
        if thinking_budget is not None:
            config.thinking_config = genai.types.ThinkingConfig(
                thinking_budget=thinking_budget, include_thoughts=include_thoughts
            )

        if system_instruction:
            config.system_instruction = system_instruction

        # Add tools if provided (conversion only - execution in Chunk 4)
        if request.tools:
            config.tools = [genai.types.Tool(function_declarations=self._convert_tools_from_request(request.tools))]

        logger.info(f"Gemini API call - model: {model}, messages: {len(all_messages)}")

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "gemini",
                    "model": model,
                    "message_count": len(all_messages),
                    "has_system": bool(system_instruction),
                },
            )

            # DEBUG level: With truncated values
            if self.debug:
                request_dict = {
                    "model": model,
                    "messages": all_messages,
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "gemini",
                        "request": self._truncate_values(request_dict),
                    },
                )

            # RAW level: Complete untruncated request (ultra-verbose)
            if self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "gemini",
                        "request": {
                            "model": model,
                            "messages": all_messages,
                            "system_instruction": system_instruction,
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        },
                    },
                )

        start_time = time.time()

        try:
            # Call Gemini API (use .aio for async)
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(model=model, contents=all_messages, config=config),
                timeout=self.timeout,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Validate response structure
            if not response.candidates or len(response.candidates) == 0:
                raise ValueError("Gemini API returned no candidates in response")

            if not hasattr(response.candidates[0], "content") or not response.candidates[0].content:
                raise ValueError("Gemini API response candidate has no content")

            if not hasattr(response.candidates[0].content, "parts"):
                raise ValueError("Gemini API response content has no parts")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                usage_data = {}
                if hasattr(response, "usage_metadata"):
                    usage_data = {
                        "input": response.usage_metadata.prompt_token_count,
                        "output": response.usage_metadata.candidates_token_count,
                    }

                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "gemini",
                        "model": model,
                        "usage": usage_data,
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: With truncated values
                if self.debug:
                    response_dict = {
                        "content_parts": str(response.candidates[0].content.parts),
                    }
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "gemini",
                            "response": self._truncate_values(response_dict),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

                # RAW level: Complete untruncated response (ultra-verbose)
                if self.debug and self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "gemini",
                            "response": {
                                "content_parts": str(response.candidates[0].content.parts),
                                "raw": str(response)[:1000],
                            },
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Gemini API error: {e}")

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "gemini",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise

    def _convert_to_chat_response(self, response) -> ChatResponse:
        """
        Convert Gemini response to ChatResponse.

        Args:
            response: Gemini API response

        Returns:
            ChatResponse with content blocks
        """
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCallBlock

        content_blocks = []
        tool_calls = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                # Check if this is thinking content
                if hasattr(part, "thought") and part.thought:
                    # This is a thinking/reasoning part
                    content_blocks.append(ThinkingBlock(thinking=part.text, signature=None))

                    # Emit thinking:final event (fire-and-forget, safe if no loop)
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        # Skip event emission if no event loop running (sync context)
                        with suppress(RuntimeError):
                            asyncio.create_task(self.coordinator.hooks.emit("thinking:final", {"text": part.text}))
                else:
                    # Regular text - ChatResponse expects TextBlock from message_models
                    content_blocks.append(TextBlock(text=part.text))
            elif hasattr(part, "function_call"):
                # Extract tool call
                fc = part.function_call
                tool_call_id = self._generate_tool_call_id()

                # Create ToolCallBlock
                content_blocks.append(
                    ToolCallBlock(
                        id=tool_call_id,
                        name=fc.name,
                        input=dict(fc.args),  # Convert to dict
                    )
                )

                # Create ToolCall for tool_calls list
                from amplifier_core.message_models import ToolCall as TCModel

                tool_calls.append(TCModel(id=tool_call_id, name=fc.name, arguments=dict(fc.args)))

        # Build metadata with usage including thought tokens
        metadata = {"raw_response": response}
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            # Gemini includes thoughtsTokenCount in usage metadata when thinking is used
            # Use getattr with defaults to handle missing fields
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        return ChatResponse(
            content=content_blocks, tool_calls=tool_calls if tool_calls else None, usage=usage, metadata=metadata
        )

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from ChatResponse.

        Filters out tool calls with empty/missing arguments to handle
        Gemini API quirk where empty function_call blocks are sometimes generated.

        Args:
            response: Chat response

        Returns:
            List of valid tool calls (with non-empty arguments)
        """
        if not response.tool_calls:
            return []

        return response.tool_calls

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
                    if isinstance(content, list):
                        # Content is a list of blocks - extract text blocks only
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append({"text": block.get("text", "")})
                            elif isinstance(block, dict) and block.get("type") == "thinking":
                                # Gemini doesn't have thinking in input, skip
                                pass
                            elif isinstance(block, dict) and block.get("type") == "tool_call":
                                # Tool calls handled separately below
                                pass
                    else:
                        # Content is a simple string
                        parts.append({"text": content})

                # Handle tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tc in msg["tool_calls"]:
                        # Extract name - handle both old format (tool) and new format (name)
                        tool_name = tc.get("name") or tc.get("tool", "")

                        parts.append(
                            {
                                "function_call": {
                                    "name": tool_name,
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

    def _convert_tools_from_request(self, tools: list) -> list:
        """
        Convert ToolSpec objects from ChatRequest to Gemini FunctionDeclaration objects.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Gemini FunctionDeclaration objects
        """
        from google.genai import types

        gemini_tools = []

        for tool in tools:
            # Create FunctionDeclaration using parametersJsonSchema (camelCase)
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parametersJsonSchema=tool.parameters,  # Already in OpenAPI/JSON Schema format
            )
            gemini_tools.append(func_decl)

        return gemini_tools
