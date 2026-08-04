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
timeout window, never the sum of several.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from handlers.panel import _QUICK_PANEL_IO_TIMEOUT_S, gemini_quick_panel
from tests.fixtures import make_ctx


class _SlowSecrets:
    """A secrets client that hangs forever -- simulates a stalled auth-gw
    connection, not a clean error."""

    async def get(self, name: str):
        await asyncio.sleep(3600)
        return "unreachable"


class _SlowStore:
    """A store client whose every method hangs forever -- simulates a
    stalled store gateway connection under load."""

    async def count(self, *args, **kwargs):
        await asyncio.sleep(3600)
        return 999

    async def query(self, *args, **kwargs):
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_left_panel_survives_a_stalled_secrets_gateway():
    """A hung ctx.secrets.get() must not block the whole left panel forever."""
    ctx = make_ctx(with_key=True)
    ctx.secrets = _SlowSecrets()

    start = time.monotonic()
    node = await asyncio.wait_for(
        gemini_quick_panel(ctx), timeout=_QUICK_PANEL_IO_TIMEOUT_S + 5,
    )
    elapsed = time.monotonic() - start

    assert node is not None
    assert elapsed < _QUICK_PANEL_IO_TIMEOUT_S + 5
    tree = node.to_dict()
    assert tree["type"] in {"Stack", "Page"}


@pytest.mark.asyncio
async def test_left_panel_survives_a_stalled_store_gateway():
    """Hung ctx.store.count()/query() must not block the whole left panel."""
    ctx = make_ctx(with_key=True)
    ctx.store = _SlowStore()

    start = time.monotonic()
    node = await asyncio.wait_for(
        gemini_quick_panel(ctx), timeout=_QUICK_PANEL_IO_TIMEOUT_S + 5,
    )
    elapsed = time.monotonic() - start

    assert node is not None
    assert elapsed < _QUICK_PANEL_IO_TIMEOUT_S + 5
    tree = node.to_dict()
    assert tree["type"] in {"Stack", "Page"}


@pytest.mark.asyncio
async def test_left_panel_io_runs_concurrently_not_sequentially():
    """Four sequential 3600s hangs must NOT sum -- they must overlap under one
    shared timeout, so total wall time stays near ONE timeout window."""
    ctx = make_ctx(with_key=True)
    ctx.secrets = _SlowSecrets()
    ctx.store = _SlowStore()

    start = time.monotonic()
    await asyncio.wait_for(
        gemini_quick_panel(ctx), timeout=_QUICK_PANEL_IO_TIMEOUT_S + 5,
    )
    elapsed = time.monotonic() - start

    # If the four reads ran sequentially this would need ~4x the timeout
    # window; concurrently it should land close to a single window.
    assert elapsed < _QUICK_PANEL_IO_TIMEOUT_S + 5
