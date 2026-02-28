"""Tests for pricing data, cost fields in ModelInfo, and vision capability.

Validates:
1. _gemini_pricing_for_model helper returns correct values
2. Fallback (hardcoded) models have cost fields and vision
3. Dynamic path adds cost fields and vision
"""

import pytest

from amplifier_module_provider_gemini import GeminiProvider


# --- 1. Pricing helper ---


class TestGeminiPricingHelper:
    """Verify _gemini_pricing_for_model returns correct values."""

    def test_flash_pricing(self):
        from amplifier_module_provider_gemini import _gemini_pricing_for_model

        p = _gemini_pricing_for_model("gemini-2.5-flash")
        assert p["input"] == 0.075e-6
        assert p["output"] == 0.30e-6
        assert p["tier"] == "low"

    def test_flash_lite_pricing(self):
        from amplifier_module_provider_gemini import _gemini_pricing_for_model

        p = _gemini_pricing_for_model("gemini-2.5-flash-lite")
        assert p["tier"] == "low"

    def test_pro_pricing(self):
        from amplifier_module_provider_gemini import _gemini_pricing_for_model

        p = _gemini_pricing_for_model("gemini-2.5-pro")
        assert p["input"] == 1.25e-6
        assert p["output"] == 10.0e-6
        assert p["tier"] == "medium"

    def test_default_pricing(self):
        from amplifier_module_provider_gemini import _gemini_pricing_for_model

        p = _gemini_pricing_for_model("gemini-unknown-model")
        assert p["tier"] == "medium"


# --- 2. Fallback models have cost fields and vision ---


class TestFallbackModelsCostAndVision:
    """Verify hardcoded fallback models include cost data and vision."""

    @pytest.fixture
    def provider(self):
        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_fallback_flash_has_cost_and_vision(self, provider):
        """Force fallback by making dynamic path fail."""
        # Don't set a client so it falls back
        provider._client = None
        provider._api_key = None  # Force exception in dynamic path

        models = await provider.list_models()
        flash = [m for m in models if m.id == "gemini-2.5-flash"][0]

        assert flash.cost_per_input_token == 0.075e-6
        assert flash.cost_per_output_token == 0.30e-6
        assert flash.metadata == {"cost_tier": "low"}
        assert "vision" in flash.capabilities

    @pytest.mark.asyncio
    async def test_fallback_pro_has_cost_and_vision(self, provider):
        provider._client = None
        provider._api_key = None

        models = await provider.list_models()
        pro = [m for m in models if m.id == "gemini-2.5-pro"][0]

        assert pro.cost_per_input_token == 1.25e-6
        assert pro.cost_per_output_token == 10.0e-6
        assert pro.metadata == {"cost_tier": "medium"}
        assert "vision" in pro.capabilities

    @pytest.mark.asyncio
    async def test_fallback_all_models_have_cost_tier(self, provider):
        provider._client = None
        provider._api_key = None

        models = await provider.list_models()
        for model in models:
            assert "cost_tier" in model.metadata, f"{model.id} missing cost_tier"

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
