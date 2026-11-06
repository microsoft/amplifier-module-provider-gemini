"""Behavioral tests for gemini provider.

Inherits authoritative tests from amplifier-core.
"""

import pytest
import httpx

from amplifier_core.validation.behavioral import ProviderBehaviorTests
from tests.test_model_discovery import provider_at


class TestGeminiProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for gemini.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Exercise the real SDK HTTP boundary, never a live list request."""
        original = provider_module._client
        async with provider_at(lambda _: httpx.Response(200, json={"models": [{
            "name": "models/gemini-3.8-flash", "displayName": "Flash",
            "inputTokenLimit": 1_000_000, "outputTokenLimit": 65536,
        }]})) as (offline, sent):
            provider_module._client = offline._client
            try:
                await super().test_list_models_returns_list(provider_module)
                assert len(sent) == 1
            finally:
                provider_module._client = original
