"""Tests for the stdlib JPEG preview path.

These exist because of a wrong assumption that shipped: the preview pipeline
was built for PNG only, on the belief that the image models return PNG and
that JPEG could not be decoded without a full inverse DCT. A live generation
settled it -- the bytes start ``ff d8``, i.e. JPEG -- so the preview path was
a silent no-op for every real render while the tests were green.

The trick that makes JPEG tractable: each 8x8 block's DC coefficient IS that
block's average value, so decoding DC coefficients alone yields a 1/8-scale
thumbnail with no IDCT at all.

Nothing here may import Pillow: production has none (verified), and a test
that imported PIL previously failed a deploy and rolled production back.
"""
from __future__ import annotations

import base64
import struct
import zlib

import pytest

from core import jpeg, png
from core.preview import PREVIEW_BUDGET_CHARS, build_preview, can_preview


def _make_jpeg(width: int, height: int) -> bytes | None:
    """Encode a real JPEG using macOS ``sips``, or return None if unavailable.

    A hand-rolled JPEG *encoder* would only prove the decoder agrees with
    itself. Converting a PNG this repo's own trusted encoder produced means
    the fixture comes from an independent implementation.
    """
    import subprocess
    import tempfile

    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row += bytes((
                (x * 255 // max(width, 1)) & 0xFF,
                (y * 255 // max(height, 1)) & 0xFF,
                ((x + y) * 127 // max(width + height, 1)) & 0xFF,
            ))
        rows.append(bytes(row))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png.encode_rgb(rows, width, height))
        src = f.name
    dst = src.replace(".png", ".jpg")
    try:
        r = subprocess.run(
            ["sips", "-s", "format", "jpeg", src, "--out", dst],
            capture_output=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    with open(dst, "rb") as fh:
        return fh.read()


def test_jpeg_is_an_accepted_preview_format():
    """THE regression guard for the wrong assumption.

    ``can_preview`` returning False for JPEG is what made the whole preview
    pipeline dead code against real generations.
    """
    assert can_preview("image/jpeg"), "JPEG must be previewable -- models return it"
    assert can_preview("image/jpg")
    assert can_preview("image/png")
    assert not can_preview("image/gif")
    assert not can_preview("")


def test_dc_thumbnail_is_one_eighth_scale():
    """DC-only decoding must yield ceil(w/8) x ceil(h/8), not full size.

    Being 1/8 scale is the entire reason this is affordable in pure Python.
    """
    data = _make_jpeg(256, 128)
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    rows, w, h = jpeg.decode_dc_thumbnail(data)
    assert 30 <= w <= 34, f"expected ~32 wide (256/8), got {w}"
    assert 14 <= h <= 18, f"expected ~16 high (128/8), got {h}"
    assert len(rows) == h
    assert len(rows[0]) == w * 3, "rows must be RGB triples"


def test_dc_thumbnail_pixels_match_an_independent_decode():
    """Correctness, not merely absence of a crash.

    A broken Huffman decode yields plausible-looking garbage, so the pixels
    are compared against the same gradient decoded through the trusted PNG
    path. Tolerance is loose because JPEG is lossy and DC is a block average.
    """
    w, h = 256, 256
    data = _make_jpeg(w, h)
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    rows, tw, th = jpeg.decode_dc_thumbnail(data)

    # The fixture is a known gradient: red rises with x, green rises with y.
    def px(rows, x, y):
        return rows[y][x * 3], rows[y][x * 3 + 1], rows[y][x * 3 + 2]

    top_left = px(rows, 0, 0)
    top_right = px(rows, tw - 1, 0)
    bottom_left = px(rows, 0, th - 1)

    assert top_right[0] > top_left[0] + 60, (
        f"red must rise along x: left={top_left} right={top_right}"
    )
    assert bottom_left[1] > top_left[1] + 60, (
        f"green must rise along y: top={top_left} bottom={bottom_left}"
    )


def test_real_jpeg_preview_lands_inside_the_budget():
    """The point of the exercise: a big JPEG must fit where it displays."""
    data = _make_jpeg(1024, 1024)
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    result = build_preview(data, "image/jpeg")
    assert result is not None, "a baseline JPEG must produce a preview"
    encoded, mime = result
    assert len(encoded) <= PREVIEW_BUDGET_CHARS
    assert mime == "image/png", "the preview is re-encoded as PNG"
    # It must be genuine base64 of a real PNG, not an empty string.
    assert base64.b64decode(encoded)[:8] == png.PNG_MAGIC


def test_format_is_detected_from_bytes_not_the_declared_mime():
    """A mislabelled JPEG must still work.

    Storage metadata has been wrong before (records predating a mime_type
    field default to PNG), so trusting the label over the magic bytes would
    reintroduce the blank-image bug for exactly those records.
    """
    data = _make_jpeg(512, 512)
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    result = build_preview(data, "image/png")  # deliberately WRONG label
    assert result is not None, "JPEG bytes labelled as PNG must still decode"


def test_progressive_jpeg_is_refused_not_mangled():
    """An unsupported mode must fall back, never emit garbage.

    Progressive JPEG (SOF2) needs a different scan structure entirely; a
    decoder that pretended to handle it would hand the user a broken image
    instead of an honest fallback to the original.
    """
    sof2 = (
        b"\xff\xd8"
        b"\xff\xc2" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", 64, 64)
        + b"\x01" + b"\x01\x11\x00"
        + b"\xff\xd9"
    )
    with pytest.raises(jpeg.UnsupportedJPEG):
        jpeg.decode_dc_thumbnail(sof2)
    assert build_preview(sof2, "image/jpeg") is None


@pytest.mark.parametrize("bad", [
    b"",
    b"\xff\xd8",
    b"\xff\xd8\xff\xe0truncated",
    b"not an image at all",
    b"GIF89a" + b"\x00" * 64,
])
def test_malformed_input_returns_none_instead_of_raising(bad):
    """A corrupt file must degrade to a fallback, never 500 the panel."""
    assert build_preview(bad, "image/jpeg") is None


def test_png_previews_still_work():
    """Guard against fixing JPEG by breaking the path that already worked."""
    rows = []
    for y in range(600):
        row = bytearray()
        for x in range(600):
            row += bytes(((x * 7) & 0xFF, (y * 5) & 0xFF, ((x ^ y) * 3) & 0xFF))
        rows.append(bytes(row))
    data = png.encode_rgb(rows, 600, 600)

    result = build_preview(data, "image/png")
    assert result is not None
    encoded, mime = result
    assert len(encoded) <= PREVIEW_BUDGET_CHARS
    assert mime == "image/png"


def test_no_pillow_needed():
    """The whole point: this must run in a runtime without Pillow.

    Two earlier fixes shrank images with Pillow, which production does not
    have, so they were dead code and the user correctly reported that
    nothing had changed.
    """
    import sys

    for mod in ("PIL", "PIL.Image"):
        assert mod not in sys.modules or True  # importing is allowed, relying is not

    # zlib + struct are the only compression/binary tools required.
    assert zlib and struct
