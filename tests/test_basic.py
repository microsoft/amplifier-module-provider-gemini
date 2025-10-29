"""Basic tests for Gemini provider foundation."""

import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini import mount


@pytest.fixture
def mock_coordinator():
    """Create mock coordinator."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    return coordinator


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "api_key": "test-api-key",
        "default_model": "gemini-2.5-flash",
        "max_tokens": 4096,
        "temperature": 0.5,
        "timeout": 60.0,
        "priority": 50,
        "debug": True,
    }


class TestProviderInitialization:
    """Test provider initialization."""

    @patch("amplifier_module_provider_gemini.genai.Client")
    def test_init_with_config(self, mock_client_class, test_config, mock_coordinator):
        """Test initialization with full config."""
        provider = GeminiProvider("test-key", test_config, mock_coordinator)

        # Verify client created
        mock_client_class.assert_called_once_with(api_key="test-key")

        # Verify config stored
        assert provider.default_model == "gemini-2.5-flash"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.5
        assert provider.timeout == 60.0
        assert provider.priority == 50
        assert provider.debug is True
        assert provider.coordinator is mock_coordinator

    @patch("amplifier_module_provider_gemini.genai.Client")
    def test_init_with_defaults(self, mock_client_class, mock_coordinator):
        """Test initialization with defaults."""
        provider = GeminiProvider("test-key", None, mock_coordinator)

        # Verify defaults applied
        assert provider.default_model == "gemini-2.5-flash"
        assert provider.max_tokens == 8192
        assert provider.temperature == 0.7
        assert provider.timeout == 300.0
        assert provider.priority == 100
        assert provider.debug is False

    @patch("amplifier_module_provider_gemini.genai.Client")
    def test_provider_name(self, mock_client_class, mock_coordinator):
        """Test provider name attribute."""
        provider = GeminiProvider("test-key", None, mock_coordinator)
        assert provider.name == "gemini"


class TestMountFunction:
    """Test mount function."""

    @pytest.mark.asyncio
    @patch("amplifier_module_provider_gemini.genai.Client")
    async def test_mount_with_config_key(self, mock_client_class, mock_coordinator, test_config):
        """Test mount with API key from config."""
        cleanup = await mount(mock_coordinator, test_config)

        # Verify provider mounted
        mock_coordinator.mount.assert_called_once()
        call_args = mock_coordinator.mount.call_args
        assert call_args[0][0] == "providers"
        assert isinstance(call_args[0][1], GeminiProvider)
        assert call_args[1]["name"] == "gemini"

        # Verify cleanup function returned
        assert cleanup is not None
        assert callable(cleanup)

    @pytest.mark.asyncio
    @patch("amplifier_module_provider_gemini.genai.Client")
    async def test_mount_with_env_key(self, mock_client_class, mock_coordinator):
        """Test mount with API key from environment."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            cleanup = await mount(mock_coordinator, {})

            # Verify provider mounted
            mock_coordinator.mount.assert_called_once()
            assert cleanup is not None

    @pytest.mark.asyncio
    async def test_mount_without_key(self, mock_coordinator):
        """Test mount without API key returns None."""
        with patch.dict(os.environ, {}, clear=True):
            cleanup = await mount(mock_coordinator, {})

            # Verify no mount, None returned
            mock_coordinator.mount.assert_not_called()
            assert cleanup is None

    @pytest.mark.asyncio
    @patch("amplifier_module_provider_gemini.genai.Client")
    async def test_mount_cleanup_function(self, mock_client_class, mock_coordinator, test_config):
        """Test cleanup function works."""
        # Create mock client with close method
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        cleanup = await mount(mock_coordinator, test_config)

        # Verify cleanup calls close
        await cleanup()
        mock_client.close.assert_called_once()


class TestPlaceholderMethods:
    """Test placeholder methods return expected behavior."""

    @patch("amplifier_module_provider_gemini.genai.Client")
    def test_complete_not_implemented(self, mock_client_class, mock_coordinator):
        """Test complete() raises NotImplementedError."""
        provider = GeminiProvider("test-key", None, mock_coordinator)

        with pytest.raises(NotImplementedError, match="Chunk 3"):
            import asyncio

            asyncio.run(provider.complete([]))

    @patch("amplifier_module_provider_gemini.genai.Client")
    def test_parse_tool_calls_returns_empty(self, mock_client_class, mock_coordinator):
        """Test parse_tool_calls() returns empty list for None."""
        provider = GeminiProvider("test-key", None, mock_coordinator)

        # Create mock response with no tool calls
        mock_response = MagicMock()
        mock_response.tool_calls = None

        result = provider.parse_tool_calls(mock_response)
        assert result == []

    @patch("amplifier_module_provider_gemini.genai.Client")
    def test_parse_tool_calls_returns_existing(self, mock_client_class, mock_coordinator):
        """Test parse_tool_calls() returns existing tool calls."""
        provider = GeminiProvider("test-key", None, mock_coordinator)

        # Create mock response with tool calls
        mock_tool_calls = [MagicMock(), MagicMock()]
        mock_response = MagicMock()
        mock_response.tool_calls = mock_tool_calls

        result = provider.parse_tool_calls(mock_response)
        assert result == mock_tool_calls
