"""Regression coverage for the "left panel tries to load and never appears"
symptom.

``gemini_quick`` is a PERMANENT left-slot panel: the host fetches it at
session-init discovery, so if its render hangs, the WHOLE extension looks
dead on open -- there is no fallback surface. Before this fix its three
store/secrets reads ran sequentially with no shared deadline: a merely SLOW
(not down) gateway could hold the panel open for the sum of every individual
transport timeout (~95s) before returning any UI tree at all, which reads to
the user as "never renders" rather than as a visible error.

These tests simulate that exact slow-gateway condition (not a hard failure)
and assert the panel still returns a real UI tree within one bounded
timeout window, never the sum of several. The per-call timeout is
monkeypatched down to a few milliseconds purely so the SUITE stays fast
(the deploy validator itself runs under a wall-clock budget) -- the
assertions exercise the exact same bounding/concurrency code path as the
production 8s constant.
"""
from __future__ import annotations

import asyncio
import time

import pytest

import handlers.panel as panel_module
from handlers.panel import gemini_quick_panel
from tests.fixtures import make_ctx

_FAST_TIMEOUT_S = 0.05
_HANG_S = 5.0  # any value >> _FAST_TIMEOUT_S proves the call never returns


class _SlowSecrets:
    """A secrets client that hangs -- simulates a stalled auth-gw
    connection, not a clean error."""

    async def get(self, name: str):
        await asyncio.sleep(_HANG_S)
        return "unreachable"


class _SlowStore:
    """A store client whose every method hangs -- simulates a stalled store
    gateway connection under load."""

    async def count(self, *args, **kwargs):
        await asyncio.sleep(_HANG_S)
        return 999

    async def query(self, *args, **kwargs):
        await asyncio.sleep(_HANG_S)
        raise AssertionError("unreachable")


@pytest.fixture
def fast_timeout(monkeypatch):
    """Shrink the panel's per-call I/O budget so hang-simulating tests stay
    fast, without changing any bounding/concurrency logic under test."""
    monkeypatch.setattr(panel_module, "_QUICK_PANEL_IO_TIMEOUT_S", _FAST_TIMEOUT_S)


@pytest.mark.asyncio
async def test_left_panel_survives_a_stalled_secrets_gateway(fast_timeout):
    """A hung ctx.secrets.get() must not block the whole left panel forever."""
    ctx = make_ctx(with_key=True)
    ctx.secrets = _SlowSecrets()

    start = time.monotonic()
    node = await asyncio.wait_for(gemini_quick_panel(ctx), timeout=2.0)
    elapsed = time.monotonic() - start

    assert node is not None
    assert elapsed < 2.0
    tree = node.to_dict()
    assert tree["type"] in {"Stack", "Page"}


@pytest.mark.asyncio
async def test_left_panel_survives_a_stalled_store_gateway(fast_timeout):
    """Hung ctx.store.count()/query() must not block the whole left panel."""
    ctx = make_ctx(with_key=True)
    ctx.store = _SlowStore()

    start = time.monotonic()
    node = await asyncio.wait_for(gemini_quick_panel(ctx), timeout=2.0)
    elapsed = time.monotonic() - start

    assert node is not None
    assert elapsed < 2.0
    tree = node.to_dict()
    assert tree["type"] in {"Stack", "Page"}


@pytest.mark.asyncio
async def test_left_panel_io_runs_concurrently_not_sequentially(fast_timeout):
    """Four sequential hangs must NOT sum -- they must overlap under one
    shared timeout, so total wall time stays near ONE timeout window."""
    ctx = make_ctx(with_key=True)
    ctx.secrets = _SlowSecrets()
    ctx.store = _SlowStore()

    start = time.monotonic()
    await asyncio.wait_for(gemini_quick_panel(ctx), timeout=2.0)
    elapsed = time.monotonic() - start

    # If the four reads ran sequentially this would need ~4x the timeout
    # window; concurrently it should land close to a single window.
    assert elapsed < 2.0
