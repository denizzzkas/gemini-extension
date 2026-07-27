"""Preview caching must actually happen -- on the REAL call pattern.

The bug this file exists for
----------------------------
``_cache_preview`` used to find the record id with
``doc_data.get("id") or doc_data.get("_id")``. But the store keeps the id on
the document WRAPPER (``doc.id``), not inside ``doc.data``, and every caller in
the app passes ``doc.data``. So the lookup always failed, the function always
returned early, and the cache was NEVER written. A large image paid the full
multi-second pure-Python JPEG decode on every single view.

The suite did not catch it, and the reason matters: the one test that touched
caching called ``_load_image(ctx, {**doc.data, "id": doc.id})`` -- it built a
dict that no production caller ever builds, so it exercised a path that only
existed in the test. A green suite therefore proved nothing about the real
call. These tests deliberately pass ``doc.data`` exactly as the panel does.

Why this is worth its own file: caching is invisible when it works and
invisible when it does not. Nothing about the rendered panel differs -- the
same image appears either way. Only the clock tells them apart, so it has to be
asserted directly.
"""
from __future__ import annotations

import base64


import pytest

from core import png
from gemini_config import GENERATION_LOG_COLLECTION, MODEL_IMAGE
from handlers.image_loader import (
    PREVIEW_FIELD, PREVIEW_MIME_FIELD, FAIL_NONE, _load_image,
)
from tests.fixtures import make_ctx


def _oversized_png(width: int = 420, height: int = 420) -> bytes:
    """A PNG too big to inline, but one a preview can actually be built from.

    Two competing constraints meet here, and both bit during development:

    * a FLAT colour deflates to almost nothing, never crosses the inline
      ceiling, and the preview branch is never reached -- the test would pass
      while proving nothing;
    * pure RANDOM noise is incompressible at every candidate size, so
      ``build_preview`` gives up and returns None, and there is no preview to
      cache at all.

    So this is structured detail: smooth gradients with a fine pattern over
    them. Big enough to need a preview, compressible enough to have one.
    """
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row += bytes((
                (x * 3 + y) % 256,
                (y * 2 + (x // 7)) % 256,
                220 if ((x // 3 + y // 3) % 2) else 40,
            ))
        rows.append(bytes(row))
    return png.encode_rgb(rows, width, height)


async def _record(ctx, raw: bytes, mime: str = "image/png"):
    """Store bytes plus a generation record that has NO cached preview."""
    path = "gemini/image/cache-test.png"
    await ctx.storage.upload(path, raw, content_type=mime)
    return await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "a record predating preview caching",
        "model": MODEL_IMAGE,
        "storage_path": path,
        "mime_type": mime,
        "created_at": "2026-01-01T10:00:00Z",
        "source": "generated",
    })


@pytest.mark.asyncio
async def test_a_built_preview_is_persisted_on_the_record():
    """The exact bug: build a preview, and it must survive on the document.

    Called the way the panel calls it -- ``doc.data`` and ``doc.id`` as two
    separate arguments, because that is the shape the real code has.
    """
    ctx = make_ctx(with_key=True)
    raw = _oversized_png()
    assert len(base64.b64encode(raw)) > 127_000, \
        "fixture must exceed the inline ceiling or the preview branch is skipped"

    doc = await _record(ctx, raw)
    assert not doc.data.get(PREVIEW_FIELD), "must start with no cached preview"

    src, reason = await _load_image(ctx, doc.data, doc.id)
    assert reason == FAIL_NONE
    assert src.startswith("data:image/")

    fresh = await ctx.store.get(GENERATION_LOG_COLLECTION, doc.id)
    cached = fresh.data.get(PREVIEW_FIELD)
    assert cached, \
        "the built preview must be written back, or every view re-decodes it"
    assert fresh.data.get(PREVIEW_MIME_FIELD), "the preview's mime must be stored too"
    assert src.endswith(cached), "the cached bytes must be the ones just served"


@pytest.mark.asyncio
async def test_the_second_view_does_not_touch_storage_at_all():
    """A cached preview must make the next view a pure document read.

    This is the property the user actually feels. Asserting on elapsed time
    would be flaky, so it is asserted structurally instead: any download at all
    on the second view is a failure.
    """
    ctx = make_ctx(with_key=True)
    doc = await _record(ctx, _oversized_png())

    first_src, _ = await _load_image(ctx, doc.data, doc.id)

    fresh = await ctx.store.get(GENERATION_LOG_COLLECTION, doc.id)

    async def _forbidden(_path):
        raise AssertionError(
            "second view downloaded the original despite a cached preview"
        )

    ctx.storage.download = _forbidden

    second_src, reason = await _load_image(ctx, fresh.data, fresh.id)
    assert reason == FAIL_NONE
    assert second_src == first_src, "the cached view must show the same image"


@pytest.mark.asyncio
async def test_caching_survives_the_panel_call_path_end_to_end():
    """Guards the integration, not just the loader.

    The bug was a mismatch BETWEEN layers: the loader looked in one place and
    every caller supplied another. Testing the loader alone would not have
    caught it, so this drives the real panel entry point instead.
    """
    from handlers.panel_detail import load_detail

    ctx = make_ctx(with_key=True)
    doc = await _record(ctx, _oversized_png())

    await load_detail(ctx, doc)

    fresh = await ctx.store.get(GENERATION_LOG_COLLECTION, doc.id)
    assert fresh.data.get(PREVIEW_FIELD), \
        "opening an image in the panel must leave a cached preview behind"


@pytest.mark.asyncio
async def test_a_missing_id_does_not_crash_the_view():
    """No id means no caching -- but the image must still be shown.

    Degrading to a slow view is acceptable; raising is not, since that would
    turn a caching miss into a blank panel.
    """
    ctx = make_ctx(with_key=True)
    doc = await _record(ctx, _oversized_png())

    src, reason = await _load_image(ctx, doc.data, None)
    assert reason == FAIL_NONE
    assert src.startswith("data:image/"), "the image must render even uncached"
