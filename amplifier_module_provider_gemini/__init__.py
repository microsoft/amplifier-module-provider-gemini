"""
Gemini provider module for Amplifier.
Integrates with Google's Gemini API.
"""

__all__ = ["mount", "GeminiProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import base64
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal
import json
import logging
import os
import time
import uuid
from contextlib import suppress
from typing import Any
from typing import TYPE_CHECKING

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.events import PROVIDER_RETRY
from amplifier_core.llm_errors import AccessDeniedError
from amplifier_core.llm_errors import AuthenticationError
from amplifier_core.llm_errors import ContentFilterError
from amplifier_core.llm_errors import ContextLengthError
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.llm_errors import LLMError
from amplifier_core.llm_errors import LLMTimeoutError
from amplifier_core.llm_errors import ProviderUnavailableError
from amplifier_core.llm_errors import RateLimitError
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from amplifier_core.utils import redact_secrets
from amplifier_core.message_models import ChatRequest
from ._cost import compute_cost
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import ToolCall
from amplifier_core.message_models import Usage

# google.genai.errors provides the native exception hierarchy for the GenAI SDK.
# Guard the import so the module still loads in unusual environments.
try:
    from google.genai import errors as genai_errors
except ImportError:
    genai_errors = None  # type: ignore[assignment]

# google.api_core.exceptions may be available as a transitive dependency.
# Some environments install it alongside google-genai; others don't.
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from google import genai  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-wide concurrency semaphore
# Shared across ALL GeminiProvider instances in this process (including
# parent + delegated child sessions). Prevents simultaneous-delegation
# blast patterns from exhausting Gemini API rate limits.
# Created lazily on the first API call; keyed by event loop so that tests
# using asyncio.run() get fresh semaphores rather than inheriting stale state.
# ---------------------------------------------------------------------------

_process_semaphore: asyncio.Semaphore | None = None
_process_semaphore_loop: Any = None  # asyncio.AbstractEventLoop
_process_semaphore_max: int = 0
_active_requests: int = 0  # currently holding semaphore (executing)
_waiting_requests: int = 0  # waiting to acquire semaphore


async def _get_process_semaphore(max_concurrent: int) -> asyncio.Semaphore | None:
    """Get or create the process-wide concurrency semaphore.

    Returns ``None`` when ``max_concurrent <= 0`` (semaphore disabled).
    Recreates the semaphore when called from a different event loop so that
    unit tests using ``asyncio.run()`` always get a fresh, valid semaphore.
    """
    global _process_semaphore, _process_semaphore_loop, _process_semaphore_max
    if max_concurrent <= 0:
        return None
    current_loop = asyncio.get_running_loop()
    if (
        _process_semaphore is None
        or _process_semaphore_loop is not current_loop
        or _process_semaphore_max != max_concurrent
    ):
        _process_semaphore = asyncio.Semaphore(max_concurrent)
        _process_semaphore_loop = current_loop
        _process_semaphore_max = max_concurrent
    return _process_semaphore


_CLOUDFLARE_403_WARNING = (
    "[PROVIDER] Cloudflare challenge detected (HTTP 403 "
    "with no details). Treating as transient — will retry."
)


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
    # Per Google GenAI SDK: supports both GEMINI_API_KEY and GOOGLE_API_KEY
    # If both are set, GOOGLE_API_KEY takes precedence
    api_key = config.get("api_key")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logger.warning(
            "No API key found for Gemini provider (set GOOGLE_API_KEY or GEMINI_API_KEY)"
        )
        return None

    _totals: dict = {"cost_usd": None, "has_data": False}

    def _add_cost(cost) -> None:
        if cost is not None:
            _totals["cost_usd"] = (_totals["cost_usd"] or Decimal("0")) + cost
            _totals["has_data"] = True

    provider = GeminiProvider(api_key, config, coordinator, add_cost=_add_cost)
    await coordinator.mount("providers", provider, name="gemini")
    logger.info("Mounted GeminiProvider")

    # Register observability events via contribution channels
    coordinator.register_contributor(
        "observability.events",
        "provider-gemini",
        lambda: [
            "llm:request",
            "llm:response",
            "provider:concurrency",
            "provider:tool_sequence_repaired",
            "thinking:final",
            "llm:stream_block_start",
            "llm:stream_block_delta",
            "llm:stream_block_end",
            "llm:stream_aborted",
        ],
    )
    coordinator.register_contributor(
        "session.cost",
        "provider-gemini",
        lambda: {
            "cost_usd": str(_totals["cost_usd"])
            if _totals["cost_usd"] is not None
            else None
        }
        if _totals["has_data"]
        else None,
    )

    # Return cleanup function
    async def cleanup():
        # genai.Client doesn't require explicit cleanup
        pass

    return cleanup


def _encode_sig(sig: bytes | str | None) -> str | None:
    """Encode a Gemini thought_signature for the wire format.

    Gemini 2.5+ attaches an opaque ``thought_signature`` to parts that follow
    a thinking burst.  The SDK returns raw bytes; the REST API accepts a
    base64-encoded ASCII string.  Pre-encoded strings (e.g. round-tripped from
    a ThinkingBlock) are returned as-is.

    Args:
        sig: Raw bytes from the SDK, an already-encoded str, or None.

    Returns:
        Base64-encoded ASCII string, or None if sig is falsy.
    """
    if not sig:
        return None
    if isinstance(sig, bytes):
        return base64.b64encode(sig).decode("ascii")
    return sig  # assume already base64-encoded str


class GeminiChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


class GeminiProvider:
    """Google Gemini API integration."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        add_cost: Callable[[Decimal | None], None] | None = None,
    ):
        """
        Initialize Gemini provider.

        The SDK client is created lazily on first use, allowing get_info()
        to work without valid credentials.

        Args:
            api_key: Google AI API key (can be None for get_info() calls)
            config: Additional configuration
            coordinator: Module coordinator for event emission
            add_cost: Optional callback to accumulate cost_usd into a session total
        """
        self._api_key = api_key
        self._client = None  # Lazy init
        self._add_cost = add_cost if add_cost is not None else lambda cost: None
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model = self.config.get("default_model", "gemini-2.5-flash")
        self.max_tokens = self.config.get("max_tokens", 8192)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = self.config.get("timeout", 600.0)
        self.priority = self.config.get("priority", 100)
        self.raw = self.config.get("raw", False)
        self.use_streaming = self.config.get("use_streaming", True)

        # Retry configuration — delegates to shared retry_with_backoff() from amplifier-core.
        self._retry_config = RetryConfig(
            max_retries=int(self.config.get("max_retries", 5)),
            initial_delay=float(self.config.get("min_retry_delay", 1.0)),
            max_delay=float(self.config.get("max_retry_delay", 60.0)),
            jitter=bool(self.config.get("retry_jitter", True)),
        )

        # Process-wide concurrency gate.
        # Limits how many API calls this process has in-flight simultaneously,
        # shared across ALL provider instances (parent + delegated child sessions).
        # This prevents blast patterns (e.g. parallel: true recipes spawning many
        # concurrent calls) from exhausting Gemini API rate limits.
        # Set to 0 to disable the semaphore entirely.
        self._max_concurrent_requests = int(
            self.config.get("max_concurrent_requests", 5)
        )

        # Track repaired tool call IDs to prevent infinite detection loops.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

    @staticmethod
    def _is_cloudflare_challenge(error) -> bool:
        """Detect CDN/proxy challenge responses for Gemini.

        When a CDN/proxy returns a 403, the google-genai SDK wraps it as a
        ClientError with details=None (no JSON body to parse).  Real Gemini
        API 403s always include structured details with an 'error' key.

        Note: this method is used for the ClientError (``genai_errors``) path
        only.  The fallback ``PermissionDenied`` path uses a broader falsy
        check (``not getattr(e, 'details', None)``) because google.api_core
        exceptions may carry empty details (``[]``, ``""``, etc.).
        """
        details = getattr(error, "details", None)
        # No details at all → likely CDN/proxy response
        if details is None:
            return True
        # Dict without 'error' key → CDN wrapped as dict
        if isinstance(details, dict) and "error" not in details:
            return True
        return False

    @property
    def client(self):
        """Lazily initialize the Gemini client on first access."""
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key must be provided for API calls")
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="gemini",
            display_name="Google Gemini",
            credential_env_vars=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            capabilities=["streaming", "tools", "thinking", "json_mode", "batch"],
            defaults={
                "model": "gemini-2.5-flash",
                "max_tokens": 8192,
                "temperature": 0.7,
                "timeout": 600.0,
            },
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your Google AI API key",
                    env_var="GOOGLE_API_KEY",
                ),
                ConfigField(
                    id="thinking_budget",
                    display_name="Thinking Budget",
                    field_type="choice",
                    prompt="Select thinking budget (-1 = dynamic, 0 = disabled)",
                    choices=["-1 (dynamic)", "0 (disabled)", "8192", "16384", "32768"],
                    default="-1 (dynamic)",
                    required=False,
                    requires_model=True,
                    show_when={"default_model": "gemini-2.5-flash"},
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """List available Gemini models via the live API.

        Raises the underlying exception if the API query fails. Callers are
        expected to handle empty lists and propagated errors — this matches
        the behaviour of provider-anthropic and provider-openai, which both
        hard-fail rather than returning a stale hardcoded fallback.

        The previous implementation kept a hardcoded 5-model fallback list
        that drifted out of sync with actual Google releases and silently
        masked API outages. Removed 2026-04-22.
        """
        models: list[ModelInfo] = []
        async for model in await self.client.aio.models.list():
            model_name = getattr(model, "name", "")
            # Filter to gemini models only (exclude tuned models, etc.)
            if not model_name or "gemini" not in model_name.lower():
                continue

            # Extract model ID from name (format: models/gemini-2.5-flash)
            model_id = model_name.split("/")[-1] if "/" in model_name else model_name

            # Skip experimental/deprecated models
            if "exp" in model_id or "001" in model_id or "002" in model_id:
                continue

            display_name = getattr(model, "display_name", model_id)
            input_limit = getattr(model, "input_token_limit", 1048576)
            output_limit = getattr(model, "output_token_limit", 8192)
            supports_thinking = getattr(model, "thinking", False)

            # Determine capabilities based on model
            capabilities = ["streaming", "json_mode"]
            if supports_thinking or "2.5" in model_id or "3" in model_id:
                capabilities.append("thinking")
            # All gemini models except 2.0-flash-lite support tools
            if "2.0-flash-lite" not in model_id:
                capabilities.append("tools")
            if "flash" in model_id.lower():
                capabilities.append("fast")
            # All Gemini 2.x+ models support vision (image input)
            if "2." in model_id or "3" in model_id:
                capabilities.append("vision")

            models.append(
                ModelInfo(
                    id=model_id,
                    display_name=display_name,
                    context_window=input_limit,
                    max_output_tokens=output_limit,
                    capabilities=capabilities,
                    defaults={
                        "temperature": 0.7,
                        "max_tokens": min(8192, output_limit),
                    },
                )
            )

        return models

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        """Extract Retry-After value from a Gemini SDK exception.

        The GenAI SDK's APIError stores the underlying httpx.Response on
        ``exc.response``.  When a 429 is returned, the Gemini API *may*
        include a ``Retry-After`` header (seconds).

        Returns:
            Parsed delay in seconds, or None if the header is absent or
            unparseable.
        """
        response = getattr(exc, "response", None)
        if response is None:
            return None

        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None

        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.
        Filters out IDs already repaired in previous iterations.

        Returns:
            List of (msg_index, call_id, tool_name, tool_arguments) tuples for
            unpaired calls, where msg_index is the index of the assistant message
            that contains the tool call.
        """
        tool_calls: dict[
            str, tuple[int, str, dict]
        ] = {}  # {call_id: (idx, name, args)}
        tool_results: set[str] = set()  # {call_id}

        for idx, msg in enumerate(messages):
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            # Check tool messages for tool_call_id
            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

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
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            # Group missing results by the index of their source assistant message
            # so that synthetics are inserted immediately after that message,
            # preserving the required tool_call → tool_result → ... ordering.
            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            # Process groups in reverse order so that earlier insertions don't
            # shift the indices of later groups that haven't been processed yet.
            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    # Track this ID so we don't detect it as missing again in future iterations
                    self._repaired_tool_ids.add(call_id)

                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

                # FM3: If a real user message immediately follows the injected synthetics,
                # the assistant turn is incomplete (tool calls with no follow-up assistant
                # text). Insert a minimal assistant bridge to satisfy the API's alternating
                # turn requirement before the user message.
                post_insert_idx = insert_pos + len(synthetics)
                if post_insert_idx < len(request.messages):
                    next_msg = request.messages[post_insert_idx]
                    is_real_user_msg = (
                        next_msg.role == "user"
                        and not getattr(next_msg, "tool_call_id", None)
                        and not (
                            isinstance(next_msg.content, str)
                            and next_msg.content.startswith("<system-reminder>")
                        )
                    )
                    if is_real_user_msg:
                        assistant_bridge = Message(
                            role="assistant",
                            content=(
                                "[SYSTEM: Tool results received. Continuing conversation.]"
                            ),
                        )
                        request.messages.insert(post_insert_idx, assistant_bridge)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> ChatResponse:
        """
        Handle ChatRequest format with developer message conversion.

        Includes error translation (native SDK errors -> kernel types) and
        retry with exponential backoff for transient failures.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        from google import genai

        logger.debug(f"Received ChatRequest with {len(request.messages)} messages")

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        # Combine system messages
        system_instruction = (
            "\n\n".join(
                m.content if isinstance(m.content, str) else "" for m in system_msgs
            )
            if system_msgs
            else None
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
            _, conversation_msgs = self._convert_messages(
                [m.model_dump() for m in conversation]
            )

        # Combine: context THEN conversation
        all_messages = context_user_msgs + conversation_msgs

        # Prepare request parameters
        model = kwargs.get("model", self.default_model)
        temperature = request.temperature or kwargs.get("temperature", self.temperature)
        max_tokens = request.max_output_tokens or kwargs.get(
            "max_tokens", self.max_tokens
        )

        # Extract thinking parameters from request metadata or kwargs
        # Default: Enable dynamic thinking with text summaries for 2.5+ models
        thinking_budget = -1  # -1 = dynamic (model decides), 0 = disabled
        include_thoughts = True  # Get text summaries of thoughts

        if request.metadata:
            if "thinking_budget" in request.metadata:
                thinking_budget = request.metadata.get("thinking_budget")
            include_thoughts = request.metadata.get("include_thoughts", True)

        # reasoning_effort support (portable interface, checked after metadata but before kwargs)
        # Maps reasoning_effort to thinking_budget values per design doc.
        if request.reasoning_effort and "thinking_budget" not in kwargs:
            effort = request.reasoning_effort.lower()
            if effort == "low":
                thinking_budget = 4096
            elif effort in ("medium", "high"):
                thinking_budget = -1  # dynamic

        # Allow kwargs to override (backward compat — takes absolute precedence)
        if "thinking_budget" in kwargs:
            thinking_budget = kwargs["thinking_budget"]
        if "include_thoughts" in kwargs:
            include_thoughts = kwargs["include_thoughts"]

        # Build Gemini config with thinking support
        config = genai.types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens
        )

        # Add thinking configuration (enabled by default for 2.5+ models)
        if thinking_budget != 0:  # 0 explicitly disables thinking
            config.thinking_config = genai.types.ThinkingConfig(
                thinking_budget=thinking_budget, include_thoughts=include_thoughts
            )

        if system_instruction:
            config.system_instruction = system_instruction

        # Add tools if provided
        if request.tools:
            config.tools = [
                genai.types.Tool(
                    function_declarations=self._convert_tools_from_request(
                        request.tools
                    )
                )
            ]
            # CRITICAL: Disable automatic function calling - Amplifier handles tool execution
            config.automatic_function_calling = (
                genai.types.AutomaticFunctionCallingConfig(disable=True)
            )

        logger.info(f"Gemini API call - model: {model}, messages: {len(all_messages)}")

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_payload: dict[str, Any] = {
                "provider": "gemini",
                "model": model,
                "message_count": len(all_messages),
                "has_system": bool(system_instruction),
            }
            if self.raw:
                request_payload["raw"] = redact_secrets(
                    {
                        "model": model,
                        "messages": all_messages,
                        "system_instruction": system_instruction,
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
            await self.coordinator.hooks.emit("llm:request", request_payload)

        start_time = time.time()

        # Call Gemini API with shared retry_with_backoff from amplifier-core.
        # Error translation happens inside _do_complete() so that retry_with_backoff
        # sees LLMError (and checks retryable) rather than raw SDK exceptions.

        async def _do_complete():
            """Single API call attempt with SDK → kernel error translation."""
            try:
                # Call Gemini API (use .aio for async)
                return await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=model, contents=all_messages, config=config
                    ),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError as e:
                raise LLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider="gemini",
                    retryable=True,
                ) from e
            except RateLimitError as e:
                # Fail-fast: if retry_after exceeds max_delay, mark non-retryable
                if (
                    e.retry_after is not None
                    and e.retry_after > self._retry_config.max_delay
                ):
                    e.retryable = False
                raise
            except LLMError:
                raise  # Already translated, don't double-wrap
            except Exception as e:
                # --- Primary path: google.genai.errors (always available) ---
                if genai_errors is not None:
                    if isinstance(e, genai_errors.ClientError):
                        code = getattr(e, "code", None)
                        details = getattr(e, "details", None)
                        error_msg = (
                            json.dumps(details) if details is not None else str(e)
                        )
                        if code == 429:
                            # Try to extract Retry-After from httpx response headers.
                            retry_after_val = self._extract_retry_after(e)
                            # Fail-fast: if retry_after exceeds max_delay, mark non-retryable
                            retryable = True
                            if (
                                retry_after_val is not None
                                and retry_after_val > self._retry_config.max_delay
                            ):
                                retryable = False
                            raise RateLimitError(
                                error_msg,
                                provider="gemini",
                                status_code=429,
                                retryable=retryable,
                                retry_after=retry_after_val,
                            ) from e
                        if code == 401:
                            raise AuthenticationError(
                                error_msg, provider="gemini", status_code=401
                            ) from e
                        if code == 403:
                            if self._is_cloudflare_challenge(e):
                                logger.warning(_CLOUDFLARE_403_WARNING)
                                raise ProviderUnavailableError(
                                    "CDN/proxy challenge (transient 403). "
                                    "This typically resolves on retry.",
                                    provider="gemini",
                                    status_code=403,
                                    retryable=True,
                                ) from e
                            raise AccessDeniedError(
                                error_msg, provider="gemini", status_code=403
                            ) from e
                        # Sub-classify 4xx by message body
                        raw_msg = str(e).lower()
                        if (
                            "context length" in raw_msg
                            or "too many tokens" in raw_msg
                            or "token limit" in raw_msg
                            or (
                                "exceeds" in raw_msg
                                and (
                                    "token" in raw_msg
                                    or "context" in raw_msg
                                    or "length" in raw_msg
                                )
                            )
                        ):
                            raise ContextLengthError(
                                error_msg,
                                provider="gemini",
                                status_code=getattr(e, "status_code", 400),
                            ) from e
                        if (
                            "content filter" in raw_msg
                            or "safety" in raw_msg
                            or "blocked" in raw_msg
                            or "harm" in raw_msg
                        ):
                            raise ContentFilterError(
                                error_msg,
                                provider="gemini",
                                status_code=getattr(e, "status_code", 400),
                            ) from e
                        raise InvalidRequestError(
                            error_msg,
                            provider="gemini",
                            status_code=code or 400,
                        ) from e
                    if isinstance(e, genai_errors.ServerError):
                        code = getattr(e, "code", None)
                        details = getattr(e, "details", None)
                        error_msg = (
                            json.dumps(details) if details is not None else str(e)
                        )
                        retry_after_val = self._extract_retry_after(e)
                        raise ProviderUnavailableError(
                            error_msg,
                            provider="gemini",
                            status_code=code or 500,
                            retryable=True,
                            retry_after=retry_after_val,
                        ) from e

                # --- Fallback: google.api_core.exceptions (if installed) ---
                if google_exceptions is not None:
                    if isinstance(e, google_exceptions.ResourceExhausted):
                        retry_after_val = self._extract_retry_after(e)
                        retryable = True
                        if (
                            retry_after_val is not None
                            and retry_after_val > self._retry_config.max_delay
                        ):
                            retryable = False
                        raise RateLimitError(
                            str(e),
                            provider="gemini",
                            status_code=429,
                            retryable=retryable,
                            retry_after=retry_after_val,
                        ) from e
                    if isinstance(e, google_exceptions.Unauthenticated):
                        raise AuthenticationError(
                            str(e), provider="gemini", status_code=401
                        ) from e
                    if isinstance(e, google_exceptions.PermissionDenied):
                        # Falsy check (not just `is None`) is intentional:
                        # google.api_core exceptions may have empty details
                        # ([], "", etc.) which also indicate a CDN/proxy 403.
                        if not getattr(e, "details", None):
                            logger.warning(_CLOUDFLARE_403_WARNING)
                            raise ProviderUnavailableError(
                                "CDN/proxy challenge (transient 403). "
                                "This typically resolves on retry.",
                                provider="gemini",
                                status_code=403,
                                retryable=True,
                            ) from e
                        raise AccessDeniedError(
                            str(e), provider="gemini", status_code=403
                        ) from e
                    if isinstance(e, google_exceptions.InvalidArgument):
                        raise InvalidRequestError(
                            str(e), provider="gemini", status_code=400
                        ) from e
                    if isinstance(e, google_exceptions.ServiceUnavailable):
                        raise ProviderUnavailableError(
                            str(e),
                            provider="gemini",
                            status_code=503,
                            retryable=True,
                        ) from e
                    if isinstance(e, google_exceptions.DeadlineExceeded):
                        raise LLMTimeoutError(
                            str(e),
                            provider="gemini",
                            retryable=True,
                        ) from e

                # Unknown errors default to retryable per design doc
                details = getattr(e, "details", None)
                error_msg = (
                    json.dumps(details)
                    if details is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                raise LLMError(
                    error_msg,
                    provider="gemini",
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: LLMError):
            """Callback invoked before each retry sleep."""
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    PROVIDER_RETRY,
                    {
                        "provider": self.name,
                        "model": model,
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        async def _do_complete_guarded():
            """Semaphore-gated wrapper around _do_complete with concurrency logging.

            Acquires the process-wide concurrency semaphore before each API call
            attempt so that at most max_concurrent_requests calls are in-flight
            simultaneously across all provider instances in this process.

            This is the function passed to retry_with_backoff so that:
            - the semaphore is *released* between retry attempts (during backoff sleep)
            - each fresh attempt must re-acquire before hitting the network
            """
            global _active_requests, _waiting_requests
            sem = await _get_process_semaphore(self._max_concurrent_requests)
            if sem is not None:
                _waiting_requests += 1
                async with sem:
                    _waiting_requests -= 1
                    _active_requests += 1
                    try:
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:concurrency",
                                {
                                    "provider": "gemini",
                                    "model": model,
                                    "active_requests": _active_requests,
                                    "waiting_requests": _waiting_requests,
                                    "max_concurrent": self._max_concurrent_requests,
                                    "process_id": os.getpid(),
                                },
                            )
                        return await _do_complete()
                    finally:
                        _active_requests -= 1
            else:
                # Semaphore disabled (max_concurrent_requests=0) — still log
                _active_requests += 1
                try:
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "provider:concurrency",
                            {
                                "provider": "gemini",
                                "model": model,
                                "active_requests": _active_requests,
                                "waiting_requests": _waiting_requests,
                                "max_concurrent": 0,
                                "process_id": os.getpid(),
                            },
                        )
                    return await _do_complete()
                finally:
                    _active_requests -= 1

        # ----------------------------------------------------------------
        # Per-request streaming override (contract §Per-request stream override)
        # ----------------------------------------------------------------
        _meta = getattr(request, "metadata", None)
        _use_streaming = self.use_streaming
        if isinstance(_meta, dict) and _meta.get("stream") is False:
            _use_streaming = False

        async def _do_complete_streaming():
            """Streaming API call with contract-compliant event emission.

            Iterates generate_content_stream chunks, synthesises block
            start/end boundaries (Gemini has no explicit ones), and emits
            the five contract events per provider-streaming-contract.md:
              llm:stream_block_start, llm:stream_block_delta (text + thinking),
              llm:stream_block_end, llm:stream_aborted.

            Timeout is enforced per-anext() because asyncio.wait_for
            cannot wrap an async generator directly.
            """
            from types import SimpleNamespace as _NS

            request_id = str(uuid.uuid4())
            block_index = -1
            current_block_type: str | None = None
            seq: dict[int, int] = {}          # block_index -> next sequence number
            partial_emitted = False
            hooks_available = bool(
                self.coordinator and hasattr(self.coordinator, "hooks")
            )

            # Per-block text accumulator and running thought_signature
            current_text_buf: list[str] = []
            current_sig = None
            # Collected virtual parts for _convert_to_chat_response
            collected_parts: list = []
            final_usage_metadata = None

            def _flush_block() -> None:
                """Merge text fragments into one part and append to collected_parts."""
                nonlocal current_text_buf, current_sig
                if current_block_type in ("text", "thinking"):
                    combined = "".join(current_text_buf)
                    if combined:
                        collected_parts.append(
                            _NS(
                                text=combined,
                                thought=(current_block_type == "thinking"),
                                thought_signature=current_sig,
                            )
                        )
                current_text_buf.clear()
                current_sig = None

            async def _open_block(btype: str, name: str | None = None) -> None:
                nonlocal block_index, current_block_type
                block_index += 1
                current_block_type = btype
                seq[block_index] = 0
                if hooks_available:
                    payload: dict[str, Any] = {
                        "request_id": request_id,
                        "block_index": block_index,
                        "block_type": btype,
                    }
                    if name is not None:
                        payload["name"] = name
                    await self.coordinator.hooks.emit(
                        "llm:stream_block_start", payload
                    )

            async def _close_block() -> None:
                nonlocal current_block_type
                if current_block_type is None:
                    return
                if hooks_available:
                    await self.coordinator.hooks.emit(
                        "llm:stream_block_end",
                        {
                            "request_id": request_id,
                            "block_index": block_index,
                            "block_type": current_block_type,
                        },
                    )
                _flush_block()
                current_block_type = None

            try:
                # Establish the stream; timeout covers connection setup
                try:
                    stream = await asyncio.wait_for(
                        self.client.aio.models.generate_content_stream(
                            model=model,
                            contents=all_messages,
                            config=config,
                        ),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError as _te:
                    raise LLMTimeoutError(
                        f"Stream connection timed out after {self.timeout}s",
                        provider="gemini",
                        retryable=True,
                    ) from _te

                # Iterate chunks; timeout enforced per-anext()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream.__anext__(),
                            timeout=self.timeout,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as _te:
                        raise LLMTimeoutError(
                            f"Stream timed out waiting for next chunk after {self.timeout}s",
                            provider="gemini",
                            retryable=True,
                        ) from _te

                    # Capture usage (present only on the final chunk)
                    _um = getattr(chunk, "usage_metadata", None)
                    if _um:
                        final_usage_metadata = _um

                    # Guard against heartbeat chunks
                    try:
                        parts = chunk.candidates[0].content.parts
                    except (AttributeError, IndexError, TypeError):
                        continue
                    if not parts:
                        continue

                    for part in parts:
                        # Always capture any thought_signature (seal fragments)
                        _sig = getattr(part, "thought_signature", None)
                        if _sig is not None:
                            current_sig = _sig

                        _has_text = hasattr(part, "text") and bool(
                            getattr(part, "text", None)
                        )
                        _has_fc = getattr(part, "function_call", None) is not None
                        _is_thought = getattr(part, "thought", False) is True

                        # Seal-only part (signature but no text/fc) — already captured
                        if not _has_text and not _has_fc:
                            continue

                        # Determine part type
                        if _is_thought:
                            part_type = "thinking"
                        elif _has_fc:
                            part_type = "tool_use"
                        else:
                            part_type = "text"

                        # ---- tool_use: arrives whole; immediate open+close ----
                        if part_type == "tool_use":
                            fc = part.function_call
                            await _close_block()
                            await _open_block("tool_use", name=fc.name)
                            # Accumulate for final response assembly
                            _tc_sig = getattr(part, "thought_signature", None)
                            collected_parts.append(
                                _NS(function_call=fc, thought_signature=_tc_sig)
                            )
                            await _close_block()
                            continue

                        # ---- text / thinking: type-transition state machine ----
                        text = part.text  # already confirmed truthy via _has_text

                        if part_type != current_block_type:
                            await _close_block()
                            await _open_block(part_type)

                        # Emit delta — ONE event for all block content (contract: guard with if text:)
                        # block_type sourced from current_block_type (equals part_type after _open_block)
                        if text:
                            if hooks_available:
                                await self.coordinator.hooks.emit(
                                    "llm:stream_block_delta",
                                    {
                                        "request_id": request_id,
                                        "block_index": block_index,
                                        "block_type": current_block_type,
                                        "sequence": seq[block_index],
                                        "text": text,
                                    },
                                )
                            seq[block_index] += 1
                            partial_emitted = True
                            current_text_buf.append(text)

                # Close the final open block (synthesised boundary at stream end)
                await _close_block()

                # Assemble ChatResponse by reusing _convert_to_chat_response
                # with a synthetic response built from collected virtual parts
                _synth = _NS(
                    candidates=[_NS(content=_NS(parts=collected_parts))],
                    usage_metadata=final_usage_metadata,
                )
                return self._convert_to_chat_response(_synth, model=model)

            except LLMTimeoutError:
                raise
            except LLMError:
                raise
            except Exception as _exc:
                if partial_emitted and hooks_available:
                    await self.coordinator.hooks.emit(
                        "llm:stream_aborted",
                        {
                            "request_id": request_id,
                            "error": {
                                "type": type(_exc).__name__,
                                "msg": str(_exc),
                            },
                        },
                    )
                raise

        async def _do_complete_streaming_guarded():
            """Semaphore-gated streaming wrapper.

            Holds the semaphore for the ENTIRE stream so that concurrency
            limits apply across the full response, not just the first chunk.
            """
            global _active_requests, _waiting_requests
            sem = await _get_process_semaphore(self._max_concurrent_requests)
            if sem is not None:
                _waiting_requests += 1
                async with sem:
                    _waiting_requests -= 1
                    _active_requests += 1
                    try:
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:concurrency",
                                {
                                    "provider": "gemini",
                                    "model": model,
                                    "active_requests": _active_requests,
                                    "waiting_requests": _waiting_requests,
                                    "max_concurrent": self._max_concurrent_requests,
                                    "process_id": os.getpid(),
                                },
                            )
                        return await _do_complete_streaming()
                    finally:
                        _active_requests -= 1
            else:
                _active_requests += 1
                try:
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "provider:concurrency",
                            {
                                "provider": "gemini",
                                "model": model,
                                "active_requests": _active_requests,
                                "waiting_requests": _waiting_requests,
                                "max_concurrent": 0,
                                "process_id": os.getpid(),
                            },
                        )
                    return await _do_complete_streaming()
                finally:
                    _active_requests -= 1
        try:
            if _use_streaming:
                chat_response = await _do_complete_streaming_guarded()
            else:
                response = await retry_with_backoff(
                    _do_complete_guarded,
                    self._retry_config,
                    on_retry=_on_retry,
                )

                # Validate response structure
                if not response.candidates or len(response.candidates) == 0:
                    raise ValueError("Gemini API returned no candidates in response")

                if (
                    not hasattr(response.candidates[0], "content")
                    or not response.candidates[0].content
                ):
                    logger.error(f"Response structure: {response}")
                    logger.error(
                        f"Candidate: {response.candidates[0] if response.candidates else 'None'}"
                    )
                    raise ValueError("Gemini API response candidate has no content")

                if (
                    not hasattr(response.candidates[0].content, "parts")
                    or not response.candidates[0].content.parts
                ):
                    logger.error(f"Content: {response.candidates[0].content}")
                    raise ValueError("Gemini API response content has no parts")

                # Convert to ChatResponse first (ordering fix — emit uses converted usage)
                chat_response = self._convert_to_chat_response(response, model=model)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Emit llm:response (common to both streaming and blocking paths)
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                usage_data: dict[str, Any] = {}
                if chat_response.usage is not None:
                    usage_data = {
                        "input_tokens": chat_response.usage.input_tokens,
                        "output_tokens": chat_response.usage.output_tokens,
                    }
                    if chat_response.usage.cache_read_tokens is not None:
                        usage_data["cache_read_tokens"] = (
                            chat_response.usage.cache_read_tokens
                        )
                    _cost = getattr(chat_response.usage, "cost_usd", None)
                    usage_data["cost_usd"] = str(_cost) if _cost is not None else None
                response_payload: dict[str, Any] = {
                    "provider": "gemini",
                    "model": model,
                    "usage": usage_data,
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                }
                if self.raw and not _use_streaming:
                    # raw logging: only available on the blocking path
                    response_payload["raw"] = redact_secrets(
                        {
                            "content_parts": str(response.candidates[0].content.parts),
                            "raw": str(response)[:1000],
                        }
                    )
                await self.coordinator.hooks.emit("llm:response", response_payload)

            return chat_response

        except LLMError as e:
            # Kernel error types — emit llm:response error event, then propagate
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] Gemini API error: %s", error_msg)

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "gemini",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            raise

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] Gemini API error: %s", error_msg)

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "gemini",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            raise
    def _convert_to_chat_response(
        self, response, *, model: str = ""
    ) -> GeminiChatResponse:
        """
        Convert Gemini response to ChatResponse.

        Args:
            response: Gemini API response

        Returns:
            GeminiChatResponse with content blocks for UI compatibility
        """
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCallBlock

        content_blocks = []
        tool_calls = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        text_accumulator: list[str] = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                # Check if this is thinking content
                # According to Gemini API docs: parts with thought=True are thinking/reasoning
                # Parts with thought_signature (but NOT thought=True) are the final answer
                if hasattr(part, "thought") and part.thought is True:
                    # This is a thinking/reasoning part (internal reasoning process)
                    # ThinkingBlock.signature is str|None; the SDK gives raw bytes —
                    # encode to base64 before storing.
                    content_blocks.append(
                        ThinkingBlock(
                            thinking=part.text,
                            signature=_encode_sig(
                                getattr(part, "thought_signature", None)
                            ),
                            visibility="internal",
                        )
                    )
                    event_blocks.append(ThinkingContent(text=part.text))
                    # NOTE: Do NOT add thinking to text_accumulator - it's internal process, not response content

                    # Emit thinking:final event (fire-and-forget, safe if no loop)
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        # Skip event emission if no event loop running (sync context)
                        with suppress(RuntimeError):
                            asyncio.create_task(
                                self.coordinator.hooks.emit(
                                    "thinking:final", {"text": part.text}
                                )
                            )
                else:
                    # Regular text (including final answer with thought_signature)
                    # Capture any thought_signature as an extra field (bytes) so the
                    # outbound path can echo it back to the API.
                    _text_sig = getattr(part, "thought_signature", None)
                    _text_kwargs: dict = (
                        {"signature": _text_sig} if _text_sig is not None else {}
                    )
                    content_blocks.append(TextBlock(text=part.text, **_text_kwargs))
                    if _text_sig is not None:
                        logger.debug(
                            "[PROVIDER] Gemini: captured thought_signature on text part (%d bytes)",
                            len(_text_sig),
                        )
                    text_accumulator.append(part.text)
                    event_blocks.append(TextContent(text=part.text))
            elif hasattr(part, "function_call"):
                # Extract tool call
                fc = part.function_call
                tool_call_id = self._generate_tool_call_id()

                # Capture thought_signature if present (Gemini 2.5+ thinking models).
                # Store as raw bytes in an extra field so the outbound path can echo
                # it back without an additional encode/decode round-trip.
                _fc_sig = getattr(part, "thought_signature", None)
                _fc_kwargs: dict = {"signature": _fc_sig} if _fc_sig is not None else {}
                if _fc_sig is not None:
                    logger.debug(
                        "[PROVIDER] Gemini: captured thought_signature on function_call "
                        "part '%s' (%d bytes)",
                        fc.name,
                        len(_fc_sig),
                    )

                # Create ToolCallBlock
                content_blocks.append(
                    ToolCallBlock(
                        id=tool_call_id,
                        name=fc.name,
                        input=dict(fc.args),  # Convert to dict
                        **_fc_kwargs,
                    )
                )

                # Create ToolCall for tool_calls list
                from amplifier_core.message_models import ToolCall as TCModel

                tool_calls.append(
                    TCModel(
                        id=tool_call_id,
                        name=fc.name,
                        arguments=dict(fc.args),
                        **_fc_kwargs,
                    )
                )
                event_blocks.append(
                    ToolCallContent(
                        id=tool_call_id, name=fc.name, arguments=dict(fc.args)
                    )
                )

        # Build metadata with usage including thought tokens
        metadata = {"raw_response": response}
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            # Gemini includes thoughtsTokenCount in usage metadata when thinking is used
            # Use getattr with defaults to handle missing fields
            input_tokens = (
                getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            )
            output_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )
            total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

            # Extract new usage fields (Phase 2)
            # thoughts_token_count: reasoning/thinking tokens (maps to reasoning_tokens)
            # cached_content_token_count: cached input tokens (maps to cache_read_tokens)
            # Preserve 0 as a valid measurement — 0 means "measured, none used",
            # while None means "not reported by the API".  Consistent with
            # OpenAI/vLLM providers.
            thoughts_tokens = getattr(
                response.usage_metadata, "thoughts_token_count", None
            )
            cached_tokens = getattr(
                response.usage_metadata, "cached_content_token_count", None
            )

            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=thoughts_tokens,
                cache_read_tokens=cached_tokens,
            )

            cost = compute_cost(
                model,
                prompt_token_count=getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                or 0,
                candidates_token_count=getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )
                or 0,
                cached_content_token_count=getattr(
                    response.usage_metadata, "cached_content_token_count", None
                )
                or 0,
            )
            usage = usage.model_copy(update={"cost_usd": cost})
            self._add_cost(cost)

        combined_text = "\n\n".join(text_accumulator).strip()

        return GeminiChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            metadata=metadata,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
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

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
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
                                text_part: dict[str, Any] = {
                                    "text": block.get("text", "")
                                }
                                if _tsig := block.get("signature"):
                                    text_part["thought_signature"] = _encode_sig(_tsig)
                                    logger.debug(
                                        "[PROVIDER] Gemini: echoing thought_signature on text part"
                                    )
                                parts.append(text_part)
                            elif (
                                isinstance(block, dict)
                                and block.get("type") == "thinking"
                            ):
                                # Echo thinking blocks that carry a thought_signature
                                # (Gemini 2.5+ requires these to maintain reasoning context).
                                # Thinking blocks WITHOUT a signature are from older models
                                # that didn't require round-tripping — skip them as before.
                                if _tsig := block.get("signature"):
                                    thought_part: dict[str, Any] = {"thought": True}
                                    if _thinking_text := block.get("thinking"):
                                        thought_part["text"] = _thinking_text
                                    thought_part["thought_signature"] = _encode_sig(
                                        _tsig
                                    )
                                    logger.debug(
                                        "[PROVIDER] Gemini: echoing thought_signature on thinking part"
                                    )
                                    parts.append(thought_part)
                            elif (
                                isinstance(block, dict)
                                and block.get("type") == "tool_call"
                            ):
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

                        fc_part: dict[str, Any] = {
                            "function_call": {
                                "name": tool_name,
                                "args": tc.get("arguments", {}),
                            }
                        }
                        # Echo thought_signature if present (Gemini 2.5+ thinking models).
                        # Omitting it causes HTTP 400 "Function call is missing a
                        # thought_signature in functionCall parts".
                        if _tc_sig := tc.get("signature"):
                            fc_part["thought_signature"] = _encode_sig(_tc_sig)
                            logger.debug(
                                "[PROVIDER] Gemini: echoing thought_signature on "
                                "function_call part '%s'",
                                tool_name,
                            )
                        parts.append(fc_part)

                gemini_contents.append({"role": gemini_role, "parts": parts})

            # Handle developer messages → user with XML wrapper
            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                gemini_contents.append({"role": "user", "parts": [{"text": wrapped}]})

            # Handle tool results → function_response
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                tool_name = msg.get("name")

                if not tool_call_id:
                    logger.warning(f"Tool result missing tool_call_id: {msg}")
                    tool_call_id = "unknown"

                if not tool_name:
                    logger.debug(
                        "Tool result missing name field, recovering from function_call history"
                    )
                    # Try to find the tool name from earlier function_call in conversation
                    # Scan backwards to find matching function_call
                    for prev_msg in reversed(gemini_contents):
                        if prev_msg.get("role") == "model" and prev_msg.get("parts"):
                            for part in prev_msg["parts"]:
                                if "function_call" in part:
                                    # Found the function call - use its name
                                    tool_name = part["function_call"]["name"]
                                    logger.info(
                                        f"Recovered tool name '{tool_name}' from function_call history"
                                    )
                                    break
                        if tool_name:
                            break

                    if not tool_name:
                        logger.error(
                            f"Could not determine tool name for tool_call_id: {tool_call_id}"
                        )
                        tool_name = "unknown"

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
                # Handle structured content (list of blocks including text and images)
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                parts.append({"text": block.get("text", "")})
                            elif block_type == "image":
                                # Convert ImageBlock to Gemini inline_data format
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": source.get(
                                                    "media_type", "image/jpeg"
                                                ),
                                                "data": source.get("data"),
                                            }
                                        }
                                    )
                                else:
                                    logger.warning(
                                        f"Unsupported image source type: {source.get('type')}"
                                    )

                    if parts:
                        gemini_contents.append({"role": "user", "parts": parts})
                else:
                    # Simple string content
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": content}]}
                    )

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
            input_schema = getattr(
                tool,
                "input_schema",
                {"type": "object", "properties": {}, "required": []},
            )

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

    async def close(self) -> None:
        """Release the genai client reference."""
        self._client = None
