"""Tests for gemini_studio (the centre panel): its default landing view and
the history list it renders there.

Split out of ``tests/test_panel.py`` to stay under the 300-line file limit
the deploy validator warns about -- ``test_panel.py`` keeps the
``gemini_quick`` (left sidebar) tests, this file keeps everything that
renders through ``gemini_studio_panel``. Both share the tree-walking
helpers in ``tests/panel_helpers.py`` rather than each defining their own.
"""
from __future__ import annotations

import pytest

from handlers.panel import gemini_studio_panel
from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx
from tests.panel_helpers import _count_type, _find_image_src, _find_types


@pytest.mark.asyncio
async def test_panel_renders_history_with_key_and_generations():
    ctx = make_ctx(with_key=True)
    await ctx.storage.upload("gemini/image/abc.png", b"abc-png-bytes", content_type="image/png")
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image",
        "prompt": "a cat astronaut", "model": "gemini-3-pro-image",
        "url": "https://panel.imperal.io/storage/default/gemini/abc.png",
        "storage_path": "gemini/image/abc.png",
        "mime_type": "image/png", "created_at": "2026-07-18T00:00:00Z",
    })

    # History renders only in gemini_studio now -- checking gemini_quick here
    # would be vacuous (it always has a Card for the API key field regardless
    # of any generation existing).
    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Card" in types
    # The image is reachable via an on-demand "View image" button, NOT inlined
    # into the list payload.
    assert "Button" in types


@pytest.mark.asyncio
async def test_studio_default_renders_history_without_needing_left_panel():
    """Studio's entry state must be usable even if the left sidebar is hidden."""
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "a studio landing image",
        "model": "gemini-3-pro-image",
        "storage_path": "gemini/image/studio.png",
        "mime_type": "image/png",
        "created_at": "2026-07-29T00:00:00Z",
    })

    tree = (await gemini_studio_panel(ctx)).to_dict()

    assert tree["type"] == "Page"
    assert _count_type(tree, "Card") == 1
    assert "Nothing open yet" not in str(tree)


@pytest.mark.asyncio
async def test_panel_empty_history():
    # History lives only in gemini_studio now -- gemini_quick renders no
    # history section at all, so the "Empty" state is asserted there.
    ctx = make_ctx(with_key=True)

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Empty" in types


@pytest.mark.asyncio
async def test_history_list_does_zero_storage_reads():
    # THE regression test for "panel loads forever", which recurred twice.
    # Fetching media while rendering the list is the cause -- even 4 downloads
    # bounded by a 3 MB budget reproduced the hang. So the list must perform
    # NO storage reads at all: any download during render fails this test.
    # History renders only in gemini_studio now.
    ctx = make_ctx(with_key=True)
    total = 12
    for i in range(total):
        await ctx.storage.upload(f"gemini/image/img{i}.png", b"x" * 64, content_type="image/png")
        await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": "image",
            "prompt": f"generation {i}",
            "model": "gemini-3-pro-image",
            "url": f"https://panel.imperal.io/storage/default/gemini/img{i}.png",
            "storage_path": f"gemini/image/img{i}.png",
            "mime_type": "image/png",
            "created_at": "2026-07-22T00:00:00+00:00",
        })

    # BaseException (not Exception) on purpose: _image_data_uri catches
    # Exception, which would silently swallow a plain AssertionError and make
    # this guard pass even while the list downloads. This must escape.
    class _ForbiddenDownload(BaseException):
        pass

    async def _forbidden_download(path):
        raise _ForbiddenDownload(
            f"history render must not download media (attempted {path!r})"
        )

    ctx.storage.download = _forbidden_download

    try:
        node = await gemini_studio_panel(ctx)
    except _ForbiddenDownload as e:
        raise AssertionError(str(e)) from None
    tree = node.to_dict()

    # Nothing inlined -> no Image nodes and no data: URIs in the payload.
    assert _count_type(tree, "Image") == 0
    assert _find_image_src(tree) is None
    # Every generation is still listed, each with its own on-demand button.
    assert _count_type(tree, "Card") >= total
