"""Tests for the Gemini Studio panel handler."""
from __future__ import annotations

import pytest

from handlers.panel import gemini_studio_panel, _quick_stats_panel
from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx


def _find_types(node: dict, acc: list[str]) -> None:
    """Walk a serialized UINode tree, collecting all 'type' fields."""
    if isinstance(node, dict):
        if "type" in node and isinstance(node["type"], str):
            acc.append(node["type"])
        for v in node.values():
            _find_types(v, acc)
    elif isinstance(node, list):
        for item in node:
            _find_types(item, acc)


@pytest.mark.asyncio
async def test_panel_renders_without_key():
    ctx = make_ctx(with_key=False)

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Alert" in types
    assert "Form" in types
    assert tree["type"] == "Page"


@pytest.mark.asyncio
async def test_panel_renders_history_with_key_and_generations():
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image",
        "prompt": "a cat astronaut", "model": "gemini-3-pro-image",
        "url": "https://storage.example.com/gemini/image/abc.png", "created_at": "2026-07-18T00:00:00Z",
    })

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Image" in types  # history entry with a url renders as an Image preview
    assert "Card" in types


@pytest.mark.asyncio
async def test_panel_empty_history():
    ctx = make_ctx(with_key=True)

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Empty" in types


@pytest.mark.asyncio
async def test_quick_stats_open_button_uses_kernel_extension_id():
    # Regression test: the "Open Gemini Studio" button must navigate using
    # the kernel-authoritative ctx._extension_id, not the Python-runtime
    # ext.app_id -- that drift is the exact class of bug that broke the
    # deployed app_id ("gemini" vs registered "gemeni") once already.
    ctx = make_ctx(with_key=True)
    ctx._extension_id = "gemeni"

    result = await _quick_stats_panel(ctx)
    tree = result["ui"]

    def _find_button_path(node):
        if isinstance(node, dict):
            if node.get("type") == "Button":
                return node.get("props", {}).get("on_click", {}).get("path")
            for v in node.values():
                found = _find_button_path(v)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _find_button_path(item)
                if found:
                    return found
        return None

    path = _find_button_path(tree)
    assert path == "/ext/gemeni/gemini_studio"
