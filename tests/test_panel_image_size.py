"""Size-related tests for inlining a generated image into a panel.

These cover the actual reason one generation opened and another did not: a real
render is orders of magnitude larger than the tiny test images used elsewhere.

IMPORTANT: nothing here may require Pillow. The deploy validator runs the suite
in an environment without third-party dependencies, and a test that imports PIL
at module scope failed the deploy (18/19) and rolled production back. Shrinking
is an optional optimisation, so the tests assert the behaviour that must hold
WITH or WITHOUT it.
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


async def test_large_render_still_opens_without_pillow():
    """THE regression test for "one generation opens, another does not".

    Small test images (a few KB) always opened; a real Nano Banana render is
    megabytes and never did. Measured locally with a representative JPEG:
    2048x1152 -> ~1.80M base64 chars, 4096x2304 -> ~7.19M.

    A large image must still be served. Refusing it on a self-invented ceiling
    would recreate the original bug, so this must pass even when Pillow is
    absent and no shrinking can happen.
    """
    from handlers.panel_viewer import _INLINE_SOFT_MAX, FAIL_NONE, _load_image

    ctx = make_ctx(with_key=True)
    # Comfortably past the soft mark once base64-expanded (4/3 growth).
    big = _incompressible(int(_INLINE_SOFT_MAX * 0.9))
    assert len(base64.b64encode(big)) > _INLINE_SOFT_MAX  # guard the premise

    await ctx.storage.upload("gemini/image/big.jpg", big, content_type="image/jpeg")
    src, reason = await _load_image(ctx, {
        "storage_path": "gemini/image/big.jpg", "mime_type": "image/jpeg",
    })

    assert reason == FAIL_NONE, f"a large render must still open (got {reason!r})"
    assert src.startswith("data:image/")
    assert src.split(",", 1)[1], "the data: URI must actually carry bytes"


async def test_shrinking_is_optional_and_never_breaks_the_panel():
    """Without Pillow, _shrink must pass the bytes through untouched."""
    import handlers.panel_viewer as viewer

    original = viewer._PILImage
    viewer._PILImage = None
    try:
        raw = b"not-a-real-image"
        out, mime = viewer._shrink(raw, "image/jpeg")
    finally:
        viewer._PILImage = original

    assert (out, mime) == (raw, "image/jpeg")


async def test_undecodable_bytes_do_not_raise():
    """Garbage must degrade to a pass-through, not an exception."""
    from handlers.panel_viewer import _shrink

    raw = b"\x00\x01\x02 definitely not an image"
    out, mime = _shrink(raw, "image/png")
    assert out == raw and mime == "image/png"


async def test_shrink_reduces_a_real_image_when_pillow_is_available():
    """When Pillow IS present, a big image should get materially smaller."""
    import handlers.panel_viewer as viewer

    if viewer._PILImage is None:
        pytest.skip("Pillow not installed in this environment")

    import io
    import random

    Image = viewer._PILImage

    from handlers.panel_viewer import _PREVIEW_MAX_DIM, _shrink

    rnd = random.Random(7)
    width, height = 2048, 1152
    im = Image.new("RGB", (width, height))
    im.putdata([
        (rnd.randint(60, 200), rnd.randint(60, 200), rnd.randint(60, 200))
        for _ in range(width * height)
    ])
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    raw = buf.getvalue()

    out, mime = _shrink(raw, "image/jpeg")

    assert mime == "image/jpeg"
    assert len(out) < len(raw), "shrinking should reduce the payload"
    with Image.open(io.BytesIO(out)) as shrunk:
        assert max(shrunk.size) <= _PREVIEW_MAX_DIM


async def test_pathological_payload_is_refused_with_its_own_reason():
    """A hard cap still exists as a safety net, with an honest reason."""
    import handlers.panel_viewer as viewer

    ctx = make_ctx(with_key=True)
    raw = b"x" * 4096
    await ctx.storage.upload("gemini/image/huge.jpg", raw, content_type="image/jpeg")

    original_soft = viewer._INLINE_SOFT_MAX
    original_hard = viewer._INLINE_HARD_MAX
    viewer._INLINE_SOFT_MAX = 10
    viewer._INLINE_HARD_MAX = 20
    try:
        src, reason = await viewer._load_image(ctx, {
            "storage_path": "gemini/image/huge.jpg", "mime_type": "image/jpeg",
        })
    finally:
        viewer._INLINE_SOFT_MAX = original_soft
        viewer._INLINE_HARD_MAX = original_hard

    assert src == ""
    assert reason == viewer.FAIL_TOO_LARGE
    assert viewer._failure_message(reason) != viewer._failure_message(viewer.FAIL_ERROR)
