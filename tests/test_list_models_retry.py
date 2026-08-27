"""Retry behavior tests for list_models().

Verifies that list_models() uses the same shared retry_with_backoff()/
_retry_config machinery as complete(): transient failures (5xx) are
retried with backoff, non-retryable failures (401) raise immediately,
and persistent transient failures raise the translated kernel error
once retries are exhausted.

See test_retry.py for the equivalent tests on the complete() path --
this file mirrors that call shape for list_models(). Also mirrors the
sibling fixes in provider-openai (PR #61) and provider-anthropic
(PR #90), which established this exact 4-case pattern.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    AuthenticationError,
    ProviderUnavailableError,
)
from amplifier_module_provider_gemini import GeminiProvider
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider(
    max_retries: int = 3, max_retry_delay: float = 60.0
) -> GeminiProvider:
    provider = GeminiProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": max_retries,
            "min_retry_delay": 0.01,  # Fast for tests
            "max_retry_delay": max_retry_delay,
            "retry_jitter": False,  # Deterministic for tests
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


class _FakeAsyncPager:
    """Minimal async-iterable stand-in for google.genai's AsyncPager.

    ``list_models()`` does ``page = await self.client.aio.models.list()``
    then ``[model async for model in page]`` -- this fakes the paginated
    async-iteration shape without depending on the real SDK's pager
    implementation.
    """

    def __init__(self, models: list):
        self._models = models

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for m in self._models:
            yield m


def _fake_model(model_id_suffix: str) -> SimpleNamespace:
    """Create a fake google.genai Model object."""
    return SimpleNamespace(
        name=f"models/{model_id_suffix}",
        display_name=model_id_suffix,
        input_token_limit=1048576,
        output_token_limit=8192,
        thinking=False,
    )


def _fake_models_page(model_id_suffixes: list[str]) -> _FakeAsyncPager:
    return _FakeAsyncPager([_fake_model(mid) for mid in model_id_suffixes])


def _make_server_error(
    code: int = 500, message: str = "error"
) -> genai_errors.ServerError:
    return genai_errors.ServerError(
        code, {"error": {"message": message, "status": "ERROR"}}
    )


def _make_client_error(code: int, message: str = "error") -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code, {"error": {"message": message, "status": "ERROR"}}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_models_succeeds_first_try():
    """No transient failure: exactly one API call, result unchanged."""
    provider = _make_provider()
    mock_client = MagicMock()
    mock_client.aio.models.list = AsyncMock(
        return_value=_fake_models_page(["gemini-2.5-flash"])
    )
    provider._client = mock_client

    with pytest.MonkeyPatch.context() as mp:
        mock_sleep = AsyncMock()
        mp.setattr(asyncio, "sleep", mock_sleep)
        models = asyncio.run(provider.list_models())

        assert mock_client.aio.models.list.await_count == 1
        mock_sleep.assert_not_awaited()

    assert len(models) == 1
    assert models[0].id == "gemini-2.5-flash"


def test_list_models_recovers_from_transient_500():
    """A single transient 500 is retried, then the call succeeds."""
    provider = _make_provider()
    fake_coordinator = cast(FakeCoordinator, provider.coordinator)
    exc = _make_server_error(500, "Temporarily down")
    mock_client = MagicMock()
    mock_client.aio.models.list = AsyncMock(
        side_effect=[exc, _fake_models_page(["gemini-2.5-flash"])]
    )
    provider._client = mock_client

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        models = asyncio.run(provider.list_models())

    assert mock_client.aio.models.list.await_count == 2
    assert len(models) == 1
    assert models[0].id == "gemini-2.5-flash"

    # One provider:retry event, correctly attributed
    retry_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 1
    assert retry_events[0][1]["provider"] == "gemini"
    assert retry_events[0][1]["error_type"] == "ProviderUnavailableError"


def test_list_models_raises_after_retries_exhausted():
    """Persistent transient failure raises the kernel error after retries."""
    provider = _make_provider(max_retries=2)
    exc = _make_server_error(503, "Still down")
    mock_client = MagicMock()
    mock_client.aio.models.list = AsyncMock(side_effect=exc)
    provider._client = mock_client

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        with pytest.raises(ProviderUnavailableError):
            asyncio.run(provider.list_models())

    # 1 initial + 2 retries = 3 total attempts
    assert mock_client.aio.models.list.await_count == 3


def test_list_models_non_retryable_error_raised_immediately():
    """A non-retryable error (401) raises immediately without retrying."""
    provider = _make_provider()
    exc = _make_client_error(401, "Bad key")
    mock_client = MagicMock()
    mock_client.aio.models.list = AsyncMock(side_effect=exc)
    provider._client = mock_client

    with pytest.MonkeyPatch.context() as mp:
        mock_sleep = AsyncMock()
        mp.setattr(asyncio, "sleep", mock_sleep)
        with pytest.raises(AuthenticationError):
            asyncio.run(provider.list_models())

        assert mock_client.aio.models.list.await_count == 1
        mock_sleep.assert_not_awaited()


def test_list_models_mid_pagination_failure_retries_whole_listing():
    """A failure partway through pagination retries the ENTIRE listing.

    Regression guard for the gemini-specific detail: the async iterator
    must be fully consumed *inside* the retried attempt, so a mid-page
    failure doesn't leave a partial result -- the whole page is re-fetched
    and re-iterated from scratch on retry.
    """

    class _FlakyPager:
        """First iteration dies partway through; second succeeds fully."""

        def __init__(self):
            self.attempts = 0

        def __aiter__(self):
            self.attempts += 1
            return self._agen(self.attempts)

        async def _agen(self, attempt: int):
            yield _fake_model("gemini-2.5-flash")
            if attempt == 1:
                raise _make_server_error(500, "Mid-pagination failure")
            yield _fake_model("gemini-2.5-pro")

    provider = _make_provider()
    flaky_pager = _FlakyPager()
    mock_client = MagicMock()
    mock_client.aio.models.list = AsyncMock(return_value=flaky_pager)
    provider._client = mock_client

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        models = asyncio.run(provider.list_models())

    # The whole listing was re-iterated from scratch on retry, not resumed --
    # both models from the successful second attempt are present.
    ids = {m.id for m in models}
    assert ids == {"gemini-2.5-flash", "gemini-2.5-pro"}


def test_list_models_filters_experimental_and_versioned_models():
    """Existing model filtering (exp/001/002 exclusions) is preserved."""
    provider = _make_provider()
    mock_client = MagicMock()
    mock_client.aio.models.list = AsyncMock(
        return_value=_fake_models_page(
            [
                "gemini-2.5-flash",
                "gemini-2.0-flash-exp",
                "gemini-1.5-flash-001",
                "gemini-1.5-pro-002",
            ]
        )
    )
    provider._client = mock_client

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        models = asyncio.run(provider.list_models())

    ids = {m.id for m in models}
    assert ids == {"gemini-2.5-flash"}
