"""Size-related tests for inlining a generated image into a panel.

These cover the actual reason one generation opened and another did not: a real
render is orders of magnitude larger than the tiny test images used elsewhere.
Measured in production: ~90k and ~127k base64 chars display, while ~954k and
~1.25M do not.

IMPORTANT: nothing here may require Pillow. The deploy validator runs the suite
in an environment without third-party dependencies, and a test that imported
PIL at module scope failed the deploy (18/19) and rolled production back. The
shrink path under test is deliberately pure-stdlib (core/png.py uses zlib), so
these tests exercise the code that ACTUALLY runs in production.
"""
from __future__ import annotations

import base64

import pytest

from tests.fixtures import make_ctx

pytestmark = pytest.mark.asyncio


def _incompressible(n_bytes: int) -> bytes:
    """Bytes that cannot be compressed away, standing in for a big render."""
    import random

    rnd = random.Random(7)
    return bytes(rnd.getrandbits(8) for _ in range(n_bytes))


def _photo_png(w: int, h: int) -> bytes:
    """A photo-like PNG: gradients plus noise, so it does not compress away.

    A flat colour would be misleading -- that is exactly how a 68KB white
    square "passed" while real renders failed.
    """
    import random

    from core import png

    rnd = random.Random(11)
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            row += bytes((
                (x * 255 // w + rnd.randint(-20, 20)) & 0xFF,
                (y * 255 // h + rnd.randint(-20, 20)) & 0xFF,
                ((x + y) * 255 // (w + h) + rnd.randint(-20, 20)) & 0xFF,
            ))
        rows.append(bytes(row))
    return png.encode_rgb(rows, w, h)


async def test_small_image_is_inlined_untouched():
    """Under the proven-good size, the original must be served verbatim."""
    from handlers.panel_viewer import FAIL_NONE, _load_image

    ctx = make_ctx(with_key=True)
    raw = b"small-image-bytes"
    await ctx.storage.upload("gemini/image/small.png", raw, content_type="image/png")

    src, reason = await _load_image(
        ctx, {"storage_path": "gemini/image/small.png", "mime_type": "image/png"},
    )

    assert reason == FAIL_NONE
    assert src == f"data:image/png;base64,{base64.b64encode(raw).decode()}"


async def test_oversized_png_is_shrunk_under_the_proven_ceiling():
    """THE regression test for "one generation opens, another does not".

    A real render inlines to ~1.25M base64 chars and does not display. It must
    come back as a preview small enough to be in the range measured to work --
    without Pillow, which production does not have.
    """
    from core.preview import PROVEN_GOOD_CHARS
    from handlers.panel_viewer import FAIL_NONE, _load_image

    ctx = make_ctx(with_key=True)
    raw = _photo_png(600, 600)
    assert len(base64.b64encode(raw)) > PROVEN_GOOD_CHARS  # guard the premise

    await ctx.storage.upload("gemini/image/big.png", raw, content_type="image/png")

    src, reason = await _load_image(
        ctx, {"storage_path": "gemini/image/big.png", "mime_type": "image/png"},
    )

    assert reason == FAIL_NONE
    assert src.startswith("data:image/png;base64,")
    payload = src.split(",", 1)[1]
    assert len(payload) < PROVEN_GOOD_CHARS, (
        f"preview is {len(payload)} chars, above the size proven to display"
    )
    # It must still be a decodable image, not a truncated blob.
    from core import png

    rows, w, h = png.decode(base64.b64decode(payload))
    assert w > 0 and h > 0 and len(rows) == h


async def test_built_preview_is_cached_on_the_record():
    """Building a preview costs ~1s of pure Python, so it must happen once."""
    from gemini_config import GENERATION_LOG_COLLECTION
    from handlers.panel_viewer import PREVIEW_FIELD, PREVIEW_MIME_FIELD, _load_image

    ctx = make_ctx(with_key=True)
    raw = _photo_png(600, 600)
    await ctx.storage.upload("gemini/image/big.png", raw, content_type="image/png")
    doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "storage_path": "gemini/image/big.png",
        "mime_type": "image/png",
    })

    src, _ = await _load_image(ctx, {"id": doc.id, **doc.data})

    stored = await ctx.store.get(GENERATION_LOG_COLLECTION, doc.id)
    assert stored is not None
    assert stored.data.get(PREVIEW_FIELD), "preview was not cached"
    assert stored.data.get(PREVIEW_MIME_FIELD) == "image/png"
    assert stored.data[PREVIEW_FIELD] == src.split(",", 1)[1]


async def test_cached_preview_skips_the_download_entirely():
    """A cached preview must be served without touching storage."""
    from handlers.panel_viewer import FAIL_NONE, PREVIEW_FIELD, PREVIEW_MIME_FIELD, _load_image

    ctx = make_ctx(with_key=True)
    cached = base64.b64encode(b"cached-preview-bytes").decode()

    async def _explode(_path):  # pragma: no cover - must never be called
        raise AssertionError("storage.download called despite a cached preview")

    ctx.storage.download = _explode  # type: ignore[method-assign]

    src, reason = await _load_image(ctx, {
        "storage_path": "gemini/image/whatever.png",
        PREVIEW_FIELD: cached,
        PREVIEW_MIME_FIELD: "image/png",
    })

    assert reason == FAIL_NONE
    assert src == f"data:image/png;base64,{cached}"


async def test_unshrinkable_payload_is_served_rather_than_refused():
    """JPEG cannot be decoded without third-party libs -- serve it anyway.

    Refusing on a self-invented ceiling is what caused the original bug: an
    unverified payload may still display, a refusal never does.
    """
    from core.preview import PROVEN_GOOD_CHARS
    from handlers.panel_viewer import FAIL_NONE, _load_image

    ctx = make_ctx(with_key=True)
    raw = _incompressible(PROVEN_GOOD_CHARS)  # not a decodable image
    await ctx.storage.upload("gemini/image/big.jpg", raw, content_type="image/jpeg")

    src, reason = await _load_image(
        ctx, {"storage_path": "gemini/image/big.jpg", "mime_type": "image/jpeg"},
    )

    assert reason == FAIL_NONE
    assert src == f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"


async def test_undecodable_bytes_do_not_raise():
    """Garbage bytes must degrade to a served payload, never an exception."""
    from handlers.panel_viewer import FAIL_NONE, _load_image

    ctx = make_ctx(with_key=True)
    raw = b"this is definitely not a png" * 8000
    await ctx.storage.upload("gemini/image/junk.png", raw, content_type="image/png")

    src, reason = await _load_image(
        ctx, {"storage_path": "gemini/image/junk.png", "mime_type": "image/png"},
    )

    assert reason == FAIL_NONE
    assert src.startswith("data:image/png;base64,")


async def test_generation_stores_a_preview_and_the_panel_uses_it():
    """End-to-end: the fix only works if BOTH halves connect.

    Generation must attach a preview, and the panel must serve that preview
    instead of downloading the original. Tested together because either half
    alone leaves the user looking at a blank image -- which is exactly how
    this bug survived several fixes.
    """
    import base64 as _b64

    from handlers.image_tools import ModelImageParams, fn_generate_image_pro
    from handlers.image_loader import FAIL_NONE, _load_image
    from tests.fixtures import INTERACTIONS_URL

    big_png = _photo_png(700, 700)
    assert len(_b64.b64encode(big_png)) > 127_000  # guard: must exceed the proven size

    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, {
        "id": "i1", "status": "completed", "model": "gemini-3-pro-image",
        "steps": [{"type": "model_output", "content": [
            {"type": "image", "data": _b64.b64encode(big_png).decode(),
             "mime_type": "image/png"},
        ]}],
    }, status=200)

    result = await fn_generate_image_pro(ctx, ModelImageParams(prompt="a big render"))
    assert result.status == "success"

    doc = await ctx.store.get("gm_generations", result.data.generation_id)
    assert doc is not None
    cached = doc.data.get("preview_b64")
    assert cached, "generation must attach a preview for the panel to use"
    assert len(cached) < 127_000

    # The panel must now serve it WITHOUT touching storage.
    async def _fail_download(_path):
        raise AssertionError("panel downloaded the original despite a cached preview")

    ctx.storage.download = _fail_download
    # Pass doc.data and doc.id the way the panel really does. This line used to
    # read ``{**doc.data, "id": doc.id}`` -- a dict no production caller ever
    # builds -- which is precisely why it stayed green while preview caching
    # was broken for every real view.
    src, reason = await _load_image(ctx, doc.data, doc.id)
    assert reason == FAIL_NONE
    assert src.startswith("data:image/png;base64,")
    assert len(src) < 127_000
