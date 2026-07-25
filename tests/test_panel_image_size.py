"""Size-related tests for inlining a generated image into a panel.

Split from tests/test_panel_viewer.py to stay under the 300-line limit the
deploy validator enforces. These cover the actual reason one generation opened
and another did not: a real render is orders of magnitude larger than the tiny
test images, and inlining it verbatim does not fit in one panel response.
"""
from __future__ import annotations

import pytest

from tests.fixtures import make_ctx


def _make_jpeg(width: int, height: int) -> bytes:
    """A real, non-flat JPEG -- flat colour would compress the problem away."""
    import io
    import random

    from PIL import Image

    random.seed(7)
    im = Image.new("RGB", (width, height))
    blocks = [
        (random.randint(60, 200), random.randint(60, 200), random.randint(60, 200))
        for _ in range((width // 4 + 1) * (height // 4 + 1))
    ]
    im.putdata([
        blocks[(y // 4) * (width // 4 + 1) + (x // 4)]
        for y in range(height) for x in range(width)
    ])
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_large_render_is_shrunk_so_it_can_actually_open():
    # THE regression test for "one generation opens, another does not".
    # Small test images (a few KB) always worked; a real Nano Banana render is
    # ~1.3 MB -> ~1.8M base64 chars, which is too big to inline in one panel
    # response. Measured locally: 2048x1152 => 1798k chars, 4096x2304 => 7190k.
    # It must be downscaled and still open, not silently fail.
    from handlers.panel_viewer import _INLINE_MAX_B64_CHARS, _load_image, FAIL_NONE

    ctx = make_ctx(with_key=True)
    big = _make_jpeg(2048, 1152)
    # Guard the premise: the raw image really is over the ceiling.
    import base64 as _b64
    assert len(_b64.b64encode(big)) > _INLINE_MAX_B64_CHARS

    await ctx.storage.upload("gemini/image/big.jpg", big, content_type="image/jpeg")
    src, reason = await _load_image(ctx, {
        "storage_path": "gemini/image/big.jpg", "mime_type": "image/jpeg",
    })

    assert reason == FAIL_NONE
    assert src.startswith("data:image/jpeg;base64,")
    payload = src.split(",", 1)[1]
    assert len(payload) <= _INLINE_MAX_B64_CHARS


@pytest.mark.asyncio
async def test_small_image_is_inlined_verbatim_without_reencoding():
    # The path that always worked must stay byte-exact: no needless re-encode.
    import base64
    from handlers.panel_viewer import _load_image, FAIL_NONE

    ctx = make_ctx(with_key=True)
    png = b"small-png-bytes"
    await ctx.storage.upload("gemini/image/small.png", png, content_type="image/png")
    src, reason = await _load_image(ctx, {
        "storage_path": "gemini/image/small.png", "mime_type": "image/png",
    })

    assert reason == FAIL_NONE
    assert src == f"data:image/png;base64,{base64.b64encode(png).decode()}"


@pytest.mark.asyncio
async def test_load_failures_are_distinguishable_not_one_blank_message():
    # Collapsing every cause into "" is why the real cause stayed hidden.
    import asyncio
    from handlers.panel_viewer import (
        FAIL_ERROR, FAIL_NO_FILE, FAIL_TIMEOUT, _failure_message, _load_image,
    )

    ctx = make_ctx(with_key=True)

    # 1) record without a stored file
    src, reason = await _load_image(ctx, {"storage_path": "", "mime_type": "image/png"})
    assert (src, reason) == ("", FAIL_NO_FILE)

    # 2) storage read raises
    async def _boom(path):
        raise RuntimeError("storage exploded")

    ctx.storage.download = _boom
    src, reason = await _load_image(ctx, {
        "storage_path": "gemini/image/x.png", "mime_type": "image/png",
    })
    assert (src, reason) == ("", FAIL_ERROR)

    # 3) storage read hangs -> timeout, reported as its own cause
    async def _hang(path):
        await asyncio.sleep(3600)

    ctx.storage.download = _hang
    import handlers.panel_viewer as viewer_mod
    original = viewer_mod._IMAGE_DOWNLOAD_TIMEOUT_S
    viewer_mod._IMAGE_DOWNLOAD_TIMEOUT_S = 0.01
    started = asyncio.get_event_loop().time()
    try:
        src, reason = await _load_image(ctx, {
            "storage_path": "gemini/image/x.png", "mime_type": "image/png",
        })
    finally:
        viewer_mod._IMAGE_DOWNLOAD_TIMEOUT_S = original
    elapsed = asyncio.get_event_loop().time() - started
    assert (src, reason) == ("", FAIL_TIMEOUT)
    # The cap must be READ at call time, not captured at import: a hardcoded
    # timeout would still report FAIL_TIMEOUT here and pass silently.
    assert elapsed < 1.0, f"timeout constant not honoured (took {elapsed:.2f}s)"

    # Every reason must produce its own human message.
    messages = {
        _failure_message(r) for r in (FAIL_NO_FILE, FAIL_ERROR, FAIL_TIMEOUT)
    }
    assert len(messages) == 3
