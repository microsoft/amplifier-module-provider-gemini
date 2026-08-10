"""Regression guard for the M2 compaction-mitigation investigation.

amplifier-module-provider-openai subscribes to "context:compaction" /
"context:pre_compact" / "context:post_compact" and breaks its server-side
response chain (`previous_response_id`) on the next request, because that
chain lets the OpenAI Responses API keep serving pre-compaction context from
server-side state even after the local (compacted) message list has shrunk.

GeminiProvider was investigated for an equivalent need and deliberately does
NOT carry an equivalent mitigation -- see the comment in `mount()` in
amplifier_module_provider_gemini/__init__.py for the full reasoning. This
test encodes the two facts that reasoning depends on, so a future change that
invalidates either one is caught rather than silently drifting:

1. mount() does not subscribe to any compaction-related hook event (there is
   no chain-like local state for such a subscription to reset).
2. GeminiProvider carries no per-turn state that would let one call's
   messages leak into the next -- each request is built solely from the
   `request.messages` passed to `complete()` for that call.

If GeminiProvider ever grows persistent cross-turn state (e.g. adopting
Gemini's *explicit* `CachedContent` handles), this test -- and the
accompanying comment in mount() -- must be revisited.
"""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from amplifier_module_provider_gemini import GeminiProvider
from amplifier_module_provider_gemini import mount


class _FakeHooks:
    """Minimal hooks registry that records every event name subscribed to."""

    def __init__(self) -> None:
        self.subscribed_events: list[str] = []

    def on(self, event: str, handler) -> None:
        self.subscribed_events.append(event)

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        return None


class _FakeCoordinator:
    """Minimal coordinator sufficient for mount()."""

    def __init__(self) -> None:
        self.hooks = _FakeHooks()
        self.mounted: dict[str, Any] = {}
        self._contributors: dict[str, list] = {}

    async def mount(self, mount_point: str, module: Any, name: str) -> None:
        self.mounted[name] = module

    def register_contributor(self, channel: str, name: str, callback) -> None:
        self._contributors.setdefault(channel, []).append((name, callback))


@pytest.mark.asyncio
async def test_mount_does_not_subscribe_to_compaction_events(monkeypatch):
    """GeminiProvider has no chain-like state, so mount() must not register
    any compaction hook handlers (unlike provider-openai, which does).

    This is an intentional absence, not an oversight -- see the comment block
    in mount() for the full investigation. If this test starts failing
    because someone *added* a compaction subscription, that's a deliberate
    change that should update this test and the comment together.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    coordinator = _FakeCoordinator()

    cleanup = await mount(coordinator, config={})  # type: ignore[arg-type]
    assert cleanup is not None

    compaction_events = {
        "context:compaction",
        "context:pre_compact",
        "context:post_compact",
    }
    subscribed = set(coordinator.hooks.subscribed_events)
    assert not (subscribed & compaction_events), (
        "GeminiProvider.mount() subscribed to a compaction event "
        f"({subscribed & compaction_events}), but the provider has no "
        "chain-like local state for such a subscription to reset. If a "
        "real need was found, update the mount() comment block explaining "
        "why the earlier 'no equivalent state' analysis no longer holds."
    )


def test_provider_holds_no_cross_turn_response_state():
    """GeminiProvider must not carry any attribute that references a prior
    turn's server-side response (the class of state OpenAI's mitigation
    resets). Every request is rebuilt fresh from `request.messages`.
    """
    provider = GeminiProvider(api_key="fake-key")

    forbidden_attr_fragments = ("response_id", "chain", "_last_response")
    provider_attrs = vars(provider)
    offending = [
        attr
        for attr in provider_attrs
        if any(fragment in attr for fragment in forbidden_attr_fragments)
    ]
    assert offending == [], (
        f"GeminiProvider unexpectedly carries cross-turn response state: {offending}. "
        "If this state was added intentionally (e.g. for explicit CachedContent "
        "handles), the M2 no-mitigation-needed analysis in mount() must be "
        "revisited, since the reasoning depends on there being no such state."
    )


@pytest.mark.asyncio
async def test_successive_complete_calls_are_independent(monkeypatch):
    """Two successive complete() calls with different message lists must
    each send exactly their own messages -- no leakage of a prior call's
    content, and no persisted state that would need a compaction-triggered
    reset.
    """
    from amplifier_core.message_models import ChatRequest
    from amplifier_core.message_models import Message

    # use_streaming=False: exercise the non-streaming complete() path
    # (_do_complete), which is the code path documented in the mount()
    # comment and read during the M2 investigation.
    provider = GeminiProvider(api_key="fake-key", config={"use_streaming": False})

    captured_calls: list[dict[str, Any]] = []

    def _make_fake_response() -> Any:
        part = MagicMock()
        part.text = "ok"
        part.thought = False
        part.function_call = None
        content = MagicMock()
        content.parts = [part]
        candidate = MagicMock()
        candidate.content = content
        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = None
        return response

    async def _fake_generate_content(*, model, contents, config):
        captured_calls.append({"contents": contents})
        return _make_fake_response()

    fake_models = MagicMock()
    fake_models.generate_content = AsyncMock(side_effect=_fake_generate_content)
    fake_aio = MagicMock()
    fake_aio.models = fake_models
    fake_client = MagicMock()
    fake_client.aio = fake_aio
    monkeypatch.setattr(GeminiProvider, "client", property(lambda self: fake_client))

    first_request = ChatRequest(
        messages=[Message(role="user", content="first turn (pre-compaction)")]
    )
    second_request = ChatRequest(
        messages=[
            Message(role="user", content="second turn (post-compaction, shorter)")
        ]
    )

    await provider.complete(first_request)
    await provider.complete(second_request)

    assert len(captured_calls) == 2
    first_texts = str(captured_calls[0]["contents"])
    second_texts = str(captured_calls[1]["contents"])

    assert "first turn" in first_texts
    assert "first turn" not in second_texts, (
        "The second call's contents leaked state from the first call -- "
        "GeminiProvider is expected to be stateless per request."
    )
    assert "second turn" in second_texts
