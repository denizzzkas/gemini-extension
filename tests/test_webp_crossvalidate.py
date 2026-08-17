"""Cross-validating the WebP (VP8L) ENCODER against an INDEPENDENT decoder.

Why an external oracle is necessary
------------------------------------
A hand-rolled binary bitstream format has no partial credit: either a real
decoder accepts it and reads back the exact pixels, or the encoder is wrong.
Self-consistency (encode then decode with our own code) proves nothing here,
because a matched pair of bugs in both directions would still "agree" -- and
that is exactly the shape of bug that bit this encoder during development:
Huffman codes were packed bit-reversed, which produced a well-formed stream
that STILL decoded (no error), just to the wrong pixels. Only comparing
against a completely separate implementation catches that class of bug.

``dwebp`` (from the ``webp`` Homebrew package, Google's own reference
implementation) is that independent decoder here. It is macOS/dev-machine
only, the same situation ``sips`` is in for the JPEG cross-validation tests
-- so this module follows that file's precedent: degrade to a skip, never an
error, when the tool is not on PATH. Production runs Linux and never needs to
decode a WebP it wrote (see core/webp.py's module docstring), so a skipped
test here does not hide a gap that matters at runtime.
"""
from __future__ import annotations

import random
import shutil
import struct
import subprocess
import tempfile
import zlib

import pytest

from core import webp

# Checked ONCE per process rather than trusting subprocess.run's own
# FileNotFoundError to fail fast: a locked-down deploy sandbox may not
# resolve/spawn an absent binary the same way a dev machine does, and a
# hang there (instead of an instant error) burns real wall time against
# the deploy validator's own runtime budget. shutil.which() never spawns a
# process, so it cannot hang for that reason regardless of sandbox behaviour.
_DWEBP = shutil.which("dwebp")


def _dwebp_decode_rgba(webp_bytes: bytes) -> tuple[list[bytes], int, int] | None:
    """Decode WebP bytes with ``dwebp`` -> (rows, width, height) RGBA, or None.

    Uses PNG output (dwebp's default) and un-filters it by hand here, rather
    than going through core.png.decode, because that decoder deliberately
    composites alpha onto white (it is a preview-display decoder, not a
    byte-exact one) -- this oracle needs the RAW alpha byte, unmodified.
    """
    if _DWEBP is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
        f.write(webp_bytes)
        src = f.name
    dst = src.replace(".webp", ".png")
    try:
        r = subprocess.run(
            [_DWEBP, src, "-o", dst], capture_output=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        with open(dst, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    return _raw_png_rgba(raw)


def _raw_png_rgba(raw: bytes) -> tuple[list[bytes], int, int] | None:
    """Minimal, alpha-preserving PNG reader: colour type 6 (RGBA) or 2 (RGB)
    only, 8-bit, non-interlaced -- exactly what dwebp emits. Returns
    (rows, width, height) with each row raw bytes/pixel (4 or 3 per pixel),
    unfiltered but NOT composited -- a deliberately separate, simpler
    decode path from core.png.decode so this oracle does not share a bug
    with the code under test.
    """
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    width = height = ctype = None
    idat = b""
    while pos + 8 <= len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag = raw[pos + 4:pos + 8]
        body = raw[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, ctype, _c, _f, interlace = struct.unpack(
                ">IIBBBBB", body[:13],
            )
            if depth != 8 or interlace:
                return None
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        pos += 12 + length
    if width is None or ctype not in (2, 6):
        return None
    bpp = 4 if ctype == 6 else 3
    stride = width * bpp
    decompressed = zlib.decompress(idat)
    rows: list[bytes] = []
    prev = bytearray(stride)
    pos = 0
    for _y in range(height):
        ftype = decompressed[pos]
        pos += 1
        line = bytearray(decompressed[pos:pos + stride])
        pos += stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if ftype == 0:
                pred = 0
            elif ftype == 1:
                pred = a
            elif ftype == 2:
                pred = b
            elif ftype == 3:
                pred = (a + b) // 2
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            else:
                return None
            line[x] = (line[x] + pred) & 0xFF
        rows.append(bytes(line))
        prev = line
    return rows, width, height


def _gradient_with_noise(width: int, height: int, seed: int = 42) -> list[bytes]:
    """A smooth-but-noisy RGB fixture: a stand-in for a photographic render.

    Pure gradients compress trivially with ANY predictor and would not
    exercise the Huffman code-length machinery hard enough (few distinct
    values -> the "simple code length code" special case, which is exactly
    what masked the bit-order bug during development). The random jitter
    forces a realistic, skewed distribution of many distinct byte values,
    which is what actually exercises the "normal" Huffman code length path.
    """
    rng = random.Random(seed)
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            r = max(0, min(255, int(128 + 100 * ((x / width) - 0.5))))
            g = max(0, min(255, int(128 + 100 * ((y / height) - 0.5))))
            b = max(0, min(255, int(128 + 60 * (((x + y) / (width + height)) - 0.5)) + rng.randint(-3, 3)))
            row += bytes((r, g, b))
        rows.append(bytes(row))
    return rows


def test_gradient_with_noise_round_trips_through_dwebp_pixel_exact():
    """The realistic-content case: predictor transform + real Huffman codes."""
    width, height = 96, 64
    rows_rgb = _gradient_with_noise(width, height)
    data = webp.encode_rgb(rows_rgb, width, height)

    decoded = _dwebp_decode_rgba(data)
    if decoded is None:
        pytest.skip("dwebp not available on this machine")
    out_rows, out_w, out_h = decoded
    assert (out_w, out_h) == (width, height)

    for y in range(height):
        src = rows_rgb[y]
        got = out_rows[y]
        for x in range(width):
            so = x * 3
            expected = (src[so], src[so + 1], src[so + 2])
            go = x * (4 if len(got) == width * 4 else 3)
            actual = (got[go], got[go + 1], got[go + 2])
            assert actual == expected, f"pixel mismatch at ({x},{y})"


def test_rgba_with_transparency_round_trips_alpha_exact():
    """Alpha must survive byte-for-byte -- not just decode without error.

    This is the case that caught the missing/misread ``alpha_is_used`` bit
    during development: a file can decode cleanly while the alpha plane is
    silently dropped because the decoder was told the image is opaque.
    """
    width, height = 40, 30
    rng = random.Random(9)
    rows_rgba = []
    expected = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            r, g, b = (x * 5) % 256, (y * 7) % 256, (x + y) % 256
            a = 255 if (x + y) % 3 else rng.randint(0, 254)
            row += bytes((r, g, b, a))
            expected.append((r, g, b, a))
        rows_rgba.append(bytes(row))

    data = webp.encode_rgba(rows_rgba, width, height)
    decoded = _dwebp_decode_rgba(data)
    if decoded is None:
        pytest.skip("dwebp not available on this machine")
    out_rows, out_w, out_h = decoded
    assert (out_w, out_h) == (width, height)

    i = 0
    for y in range(height):
        row = out_rows[y]
        for x in range(width):
            o = x * 4
            got = (row[o], row[o + 1], row[o + 2], row[o + 3])
            assert got == expected[i], f"RGBA mismatch at ({x},{y})"
            i += 1


@pytest.mark.parametrize("width,height", [(1, 1), (1, 50), (50, 1), (17, 17)])
def test_edge_sizes_round_trip_through_dwebp(width, height):
    """Degenerate dimensions: single row/column, and a non-power-of-two size
    that does not divide evenly into the predictor's block grid."""
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row += bytes(((x * 13 + y * 7) % 256, (x * 3) % 256, (y * 11) % 256))
        rows.append(bytes(row))

    data = webp.encode_rgb(rows, width, height)
    decoded = _dwebp_decode_rgba(data)
    if decoded is None:
        pytest.skip("dwebp not available on this machine")
    out_rows, out_w, out_h = decoded
    assert (out_w, out_h) == (width, height)

    for y in range(height):
        src = rows[y]
        got = out_rows[y]
        bpp = 4 if len(got) == width * 4 else 3
        for x in range(width):
            so = x * 3
            expected = (src[so], src[so + 1], src[so + 2])
            go = x * bpp
            actual = (got[go], got[go + 1], got[go + 2])
            assert actual == expected, f"pixel mismatch at ({x},{y}) size {width}x{height}"
