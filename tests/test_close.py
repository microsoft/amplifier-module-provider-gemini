"""Tests for GeminiProvider.close() method."""

from unittest.mock import MagicMock

import pytest

from amplifier_module_provider_gemini import GeminiProvider


@pytest.mark.asyncio
async def test_close_nils_client_when_initialized():
    """close() should set _client to None when it was previously initialized."""
    provider = GeminiProvider(api_key="fake-key")
    provider._client = MagicMock()
    assert provider._client is not None

    await provider.close()

    assert provider._client is None


@pytest.mark.asyncio
async def test_close_is_safe_when_client_is_none():
    """close() should not crash when _client is already None."""
    provider = GeminiProvider(api_key="fake-key")
    assert provider._client is None

    await provider.close()  # Should not raise

    assert provider._client is None


@pytest.mark.asyncio
async def test_close_can_be_called_twice():
    """close() should be idempotent — calling it twice should not crash."""
    provider = GeminiProvider(api_key="fake-key")
    provider._client = MagicMock()

    await provider.close()
    await provider.close()

    assert provider._client is None
