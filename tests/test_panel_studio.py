"""Tests for gemini_studio (the centre panel): its default landing view and
the history list it renders there.

Split out of ``tests/test_panel.py`` to stay under the 300-line file limit
the deploy validator warns about -- ``test_panel.py`` keeps the
``gemini_quick`` (left sidebar) tests, this file keeps everything that
renders through ``gemini_studio_panel``. Both share the tree-walking
helpers in ``tests/panel_helpers.py`` rather than each defining their own.
"""
from __future__ import annotations

import base64

import pytest

from handlers.panel import gemini_studio_panel
from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx
from tests.panel_helpers import _count_type, _find_image_src, _find_types, _walk


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

    assert tree["type"] == "Stack"
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
async def test_history_payload_stays_under_reply_cap_with_many_large_previews():
    # THE regression test for the real production bug: 28 real generations,
    # each carrying a preview near build_preview's own ceiling
    # (PREVIEW_BUDGET_CHARS = 110,000 base64 chars, see core/preview.py),
    # silently blew the panel reply's ~256 KB hard cap. The panel then never
    # rendered at all -- no exception, no console error, just a permanent
    # spinner, because the client had no ui tree to mount. PANEL_HISTORY_LIMIT
    # bounds row COUNT, not cumulative payload size, which is what actually
    # broke. This proves the fix: cumulative base64 across all rendered cards
    # must stay well under the reply cap, no matter how many generations
    # exist, and the panel must say plainly that some are hidden rather than
    # silently dropping them (or the whole panel) from the reply.
    ctx = make_ctx(with_key=True)
    total = 28
    big_preview = base64.b64encode(b"x" * 82_000).decode()  # ~109,336 base64 chars
    for i in range(total):
        await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": "image",
            "prompt": f"generation {i}",
            "model": "gemini-3-pro-image",
            "storage_path": f"gemini/image/img{i}.png",
            "mime_type": "image/png",
            "created_at": "2026-07-22T00:00:00+00:00",
            "preview_b64": big_preview,
            "preview_mime": "image/png",
        })

    tree = (await gemini_studio_panel(ctx)).to_dict()

    total_preview_chars = sum(
        len(props.get("src", ""))
        for t, props in _walk(tree) if t == "Image"
    )
    # Cumulative preview payload must stay well under the measured 256 KB
    # reply cap -- this is the number that actually overflowed in production.
    assert total_preview_chars < 200_000
    # Not every generation can fit -- the panel must say so honestly instead
    # of silently truncating the whole reply.
    assert "Showing" in str(tree)
    assert f"of {total}" in str(tree)


@pytest.mark.asyncio
async def test_history_load_more_paginates_past_the_first_page():
    """Older generations beyond PANEL_HISTORY_LIMIT must be reachable via
    "Load more", not silently unreachable forever -- see this module's
    other test for why 'no zero-storage-reads' alone is not enough: a list
    that only ever shows its first page is just as useless for a heavy
    user as one that hangs.
    """
    from gemini_config import PANEL_HISTORY_LIMIT

    ctx = make_ctx(with_key=True)
    total = PANEL_HISTORY_LIMIT + 5
    for i in range(total):
        await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": "image",
            "prompt": f"generation {i}",
            "model": "gemini-3-pro-image",
            "storage_path": f"gemini/image/img{i}.png",
            "mime_type": "image/png",
            # Descending so index 0 is newest -- newest_first() then keeps
            # this exact order, making page membership easy to assert on.
            "created_at": f"2026-07-22T00:{total - i:02d}:00+00:00",
        })

    first_page = (await gemini_studio_panel(ctx)).to_dict()
    assert f"1-{PANEL_HISTORY_LIMIT}" in str(first_page)
    assert _count_type(first_page, "Card") == PANEL_HISTORY_LIMIT
    # A page that is NOT the true end of history must offer a way to see more.
    assert "Load more" in str(first_page)

    second_page = (await gemini_studio_panel(
        ctx, history_offset=str(PANEL_HISTORY_LIMIT),
    )).to_dict()
    # The remaining 5 generations are what's left, and the true end is
    # reached -- no further "Load more"/"Next" button.
    assert _count_type(second_page, "Card") == 5
    assert "Load more" not in str(second_page)
    assert "Next" not in str(second_page)
    assert f"{PANEL_HISTORY_LIMIT + 1}-{total}" in str(second_page)
    # But there IS a way back -- this was the user's exact complaint: "Load
    # more" was a one-way ratchet with no return path once you had scrolled
    # past the newest page.
    assert "Previous" in str(second_page)

    # Clicking Previous must land back on page 1, unchanged.
    back_to_first = (await gemini_studio_panel(ctx, history_offset="0")).to_dict()
    assert _count_type(back_to_first, "Card") == PANEL_HISTORY_LIMIT
    assert f"1-{PANEL_HISTORY_LIMIT}" in str(back_to_first)
    assert "Previous" not in str(back_to_first)


@pytest.mark.asyncio
async def test_history_load_more_tolerates_a_bad_offset_param():
    """A tampered/garbage history_offset must degrade to page 1, not crash."""
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image",
        "prompt": "a cat astronaut", "model": "gemini-3-pro-image",
        "storage_path": "gemini/image/abc.png",
        "mime_type": "image/png", "created_at": "2026-07-18T00:00:00Z",
    })

    tree = (await gemini_studio_panel(ctx, history_offset="not-a-number")).to_dict()
    assert _count_type(tree, "Card") == 1

    tree = (await gemini_studio_panel(ctx, history_offset="-5")).to_dict()
    assert _count_type(tree, "Card") == 1


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
