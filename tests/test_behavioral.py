"""Behavioral tests for gemini provider.

Inherits authoritative tests from amplifier-core.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.validation.behavioral import ProviderBehaviorTests


class TestGeminiProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for gemini.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Exercise actual async-pager catalog mapping without a network call."""
        async def models():
            yield SimpleNamespace(name="models/gemini-3.8-flash", display_name="Flash",
                                  input_token_limit=1_000_000, output_token_limit=65536)
        provider_module.client.aio.models.list = AsyncMock(return_value=models())
        await super().test_list_models_returns_list(provider_module)
