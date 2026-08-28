"""Tests for vision capability on Gemini models."""

import pytest

from amplifier_core.llm_errors import LLMError
from amplifier_module_provider_gemini import GeminiProvider


class TestNoCredentialsHardFails:
    """``list_models()`` must hard-fail without credentials, not fall back.

    A hardcoded 5-model fallback list used to be returned here when no
    live API call could be made. It was intentionally removed (see the
    docstring on ``GeminiProvider.list_models``) because a stale hardcoded
    list drifts out of sync with real Google releases and silently masks
    API outages -- matching the anthropic/openai providers, which also
    hard-fail rather than returning a fallback.

    This replaces the old ``TestFallbackModelsVision`` tests, which
    asserted on the removed fallback list's contents (vision/fast
    capabilities) and could never pass again now that the list is gone.
    """

    @pytest.fixture
    def provider(self):
        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_list_models_without_credentials_raises(self, provider):
        provider._client = None
        provider._api_key = None

        with pytest.raises(LLMError, match="api_key must be provided"):
            await provider.list_models()
