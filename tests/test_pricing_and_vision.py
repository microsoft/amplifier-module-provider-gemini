"""Tests for vision capability on Gemini models."""

import pytest

from amplifier_module_provider_gemini import GeminiProvider


class TestFallbackModelsVision:
    """Verify hardcoded fallback models include vision capability."""

    @pytest.fixture
    def provider(self):
        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_fallback_all_models_have_vision(self, provider):
        provider._client = None
        provider._api_key = None

        models = await provider.list_models()
        for model in models:
            assert "vision" in model.capabilities, f"{model.id} missing vision"

    @pytest.mark.asyncio
    async def test_fallback_flash_is_fast(self, provider):
        provider._client = None
        provider._api_key = None

        models = await provider.list_models()
        flash_models = [m for m in models if "flash" in m.id]
        for model in flash_models:
            assert "fast" in model.capabilities, f"{model.id} should be fast"
