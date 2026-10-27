"""Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances
"""

import asyncio
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def event_loop_policy(event_loop_policy):
    """Own pytest-asyncio's policy without changing the caller's loop slot."""
    previous = event_loop_policy
    if sys.version_info >= (3, 14):
        yield previous
        return
    policy = asyncio.DefaultEventLoopPolicy()
    policy.set_event_loop(None)
    # The plugin reads the current policy before installing its fixture value.
    asyncio.set_event_loop_policy(policy)
    try:
        yield policy
    finally:
        asyncio.set_event_loop_policy(previous)


@pytest.fixture(autouse=True)
def offline_contract_credentials(request, monkeypatch):
    """Mount real providers for offline inherited contracts, without secrets."""
    if request.node.path.name in {"test_behavioral.py", "test_validation.py"}:
        monkeypatch.setenv("GOOGLE_API_KEY", "offline-contract-placeholder")
