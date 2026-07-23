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
async def test_quick_stats_open_button_uses_panel_call_action():
    # Regression test: the "Open Gemini Studio" button must use
    # ui.Call("__panel__gemini_studio") -- panels are fetched via the /call
    # endpoint as __panel__{panel_id} (see ext.panel()'s docstring), there
    # is no frontend route for a raw /ext/<app>/<panel_id> URL path. An
    # earlier version of this button used ui.Navigate(path=...) instead,
    # which 404'd in the panel host -- this is the actual root cause of
    # the reported "Open Gemini AI opens a 404" bug.
    ctx = make_ctx(with_key=True)

    result = await _quick_stats_panel(ctx)
    tree = result["ui"]

    def _find_button_on_click(node):
        if isinstance(node, dict):
            if node.get("type") == "Button":
                return node.get("props", {}).get("on_click", {})
            for v in node.values():
                found = _find_button_on_click(v)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _find_button_on_click(item)
                if found:
                    return found
        return None

    on_click = _find_button_on_click(tree)
    assert on_click is not None
    assert on_click.get("action") == "call"
    assert on_click.get("function") == "__panel__gemini_studio"


@pytest.mark.asyncio
async def test_panel_image_form_has_model_select_with_all_choices():
    from gemini_config import IMAGE_MODEL_CHOICES, MODEL_IMAGE

    ctx = make_ctx(with_key=True)
    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    def _find_select(n):
        if isinstance(n, dict):
            if n.get("type") == "Select":
                return n.get("props", {})
            for v in n.values():
                found = _find_select(v)
                if found:
                    return found
        elif isinstance(n, list):
            for item in n:
                found = _find_select(item)
                if found:
                    return found
        return None

    select_props = _find_select(tree)
    assert select_props is not None
    assert select_props["value"] == MODEL_IMAGE
    option_values = {opt["value"] for opt in select_props["options"]}
    assert option_values == set(IMAGE_MODEL_CHOICES)


@pytest.mark.asyncio
async def test_panel_history_prefers_storage_download_over_signed_url():
    # ctx.storage.upload() returns a *signed* URL (see the SDK's own
    # FileInfo schema docstring: "storage path, size, MIME, and signed
    # URL") -- it can expire, which is the actual root cause of "image
    # unavailable" reports for generations that worked right after
    # creation. The panel must re-download the saved bytes via the stable
    # storage_path and embed them as a data: URI instead of trusting the
    # (possibly stale/expired) stored url.
    import base64
    ctx = make_ctx(with_key=True)
    fake_png_bytes = b"fake-png-bytes-for-panel-test"
    await ctx.storage.upload("gemini/image/fresh123.png", fake_png_bytes, content_type="image/png")
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "a generation whose signed url may have expired",
        "model": "gemini-3-pro-image",
        "url": "https://storage.example.com/gemini/image/fresh123.png?sig=maybe-expired",
        "storage_path": "gemini/image/fresh123.png",
        "mime_type": "image/png",
        "created_at": "2026-07-22T00:00:00+00:00",
    })

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    def _find_image_src(n):
        if isinstance(n, dict):
            if n.get("type") == "Image":
                return n.get("props", {}).get("src")
            for v in n.values():
                found = _find_image_src(v)
                if found:
                    return found
        elif isinstance(n, list):
            for item in n:
                found = _find_image_src(item)
                if found:
                    return found
        return None

    src = _find_image_src(tree)
    assert src is not None
    assert src.startswith("data:image/png;base64,")
    assert base64.b64decode(src.split(",", 1)[1]) == fake_png_bytes


@pytest.mark.asyncio
async def test_panel_history_normalizes_legacy_relative_url():
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "an old pre-fix generation",
        "model": "gemini-3-pro-image",
        "url": "/storage/default/gemeni/legacy123.jpg",
        "storage_path": "gemini/image/legacy123.jpg",
        "mime_type": "image/jpeg",
        "created_at": "2026-07-19T00:00:00+00:00",
    })

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    def _find_image_src(n):
        if isinstance(n, dict):
            if n.get("type") == "Image":
                return n.get("props", {}).get("src")
            for v in n.values():
                found = _find_image_src(v)
                if found:
                    return found
        elif isinstance(n, list):
            for item in n:
                found = _find_image_src(item)
                if found:
                    return found
        return None

    src = _find_image_src(tree)
    assert src is not None
    assert src.startswith("https://")


@pytest.mark.asyncio
async def test_panel_history_slow_download_times_out_instead_of_hanging():
    # Regression test for "Open Gemini Studio now opens the panel but it
    # loads forever": _history_section used to `await` ctx.storage.download()
    # for every image ONE AT A TIME in a loop -- a single slow/hanging
    # download (the real storage client has a 60s timeout) made the whole
    # panel render (and therefore the whole panel open) hang. Verifies that
    # a slow download for one item times out (via _PREVIEW_DOWNLOAD_TIMEOUT_S)
    # and falls back to the stored url instead of blocking panel render.
    import asyncio
    import handlers.panel as panel_mod

    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "a generation whose storage download hangs",
        "model": "gemini-3-pro-image",
        "url": "https://storage.example.com/gemini/image/slow123.png?sig=abc",
        "storage_path": "gemini/image/slow123.png",
        "mime_type": "image/png",
        "created_at": "2026-07-23T00:00:00+00:00",
    })

    real_download = ctx.storage.download

    async def _hanging_download(path):
        if path == "gemini/image/slow123.png":
            await asyncio.sleep(5)  # longer than the patched timeout below
        return await real_download(path)

    ctx.storage.download = _hanging_download

    original_timeout = panel_mod._PREVIEW_DOWNLOAD_TIMEOUT_S
    panel_mod._PREVIEW_DOWNLOAD_TIMEOUT_S = 0.05
    try:
        node = await asyncio.wait_for(gemini_studio_panel(ctx), timeout=2.0)
    finally:
        panel_mod._PREVIEW_DOWNLOAD_TIMEOUT_S = original_timeout

    tree = node.to_dict()

    def _find_image_src(n):
        if isinstance(n, dict):
            if n.get("type") == "Image":
                return n.get("props", {}).get("src")
            for v in n.values():
                found = _find_image_src(v)
                if found:
                    return found
        elif isinstance(n, list):
            for item in n:
                found = _find_image_src(item)
                if found:
                    return found
        return None

    src = _find_image_src(tree)
    assert src is not None
    # Timed out -> falls back to the normalized stored url, not a data: URI.
    assert src.startswith("https://")
