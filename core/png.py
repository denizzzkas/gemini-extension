"""Pure-stdlib PNG decode / downscale / re-encode — no Pillow, no third party.

Why this module exists
----------------------
Generated renders are far too heavy to inline into a panel response
(measured in production: ~940KB PNG -> ~1.25M base64 chars, which does not
display, while ~127k chars does). Every earlier attempt to shrink them
used Pillow -- which the production runtime does NOT have
(``pillow_available: false``, verified), so that code never executed and
the bug survived several "fixes".

A PNG's pixel data is just zlib, and ``zlib`` IS in the standard library.
So the whole decode -> downscale -> re-encode round trip is expressible
with ``zlib`` + ``struct`` alone, which means it actually RUNS in
production.

Scope, deliberately narrow
--------------------------
* 8-bit, non-interlaced PNG (colour types 0/2/3/4/6). That covers what
  the image models emit and ordinary screenshots. Anything else raises
  ``UnsupportedPNG`` so the caller can fall back instead of guessing.
* No JPEG. Decoding JPEG needs a DCT/Huffman implementation -- far too
  much surface to hand-roll. Callers who need a preview must therefore
  request PNG from the model.
"""
from __future__ import annotations

import struct
import zlib

__all__ = ["UnsupportedPNG", "decode", "downscale", "encode_rgb", "encode_palette"]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Samples per axis when averaging a source block into one output pixel.
# 1 would be nearest-neighbour (visibly aliased on text); the cost is
# ``_MAX_SAMPLES**2`` operations per output pixel, so this is a quality /
# CPU trade-off, not an arbitrary constant.
_MAX_SAMPLES = 3

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class UnsupportedPNG(Exception):
    """The PNG uses a feature this minimal decoder does not implement."""


def _crc_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _unfilter(raw: bytes, width: int, height: int, bpp: int) -> list[bytes]:
    """Reverse the per-row PNG filters. Returns one bytes object per row."""
    stride = width * bpp
    rows: list[bytes] = []
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        if pos >= len(raw):
            raise UnsupportedPNG(f"truncated pixel data at row {y}")
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        if len(line) != stride:
            raise UnsupportedPNG(f"truncated row {y}")
        pos += stride

        if ftype == 0:
            pass
        elif ftype == 1:  # Sub
            for k in range(bpp, stride):
                line[k] = (line[k] + line[k - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for k in range(stride):
                line[k] = (line[k] + prev[k]) & 0xFF
        elif ftype == 3:  # Average
            for k in range(stride):
                left = line[k - bpp] if k >= bpp else 0
                line[k] = (line[k] + ((left + prev[k]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for k in range(stride):
                left = line[k - bpp] if k >= bpp else 0
                upleft = prev[k - bpp] if k >= bpp else 0
                up = prev[k]
                p = left + up - upleft
                pa = p - left
                if pa < 0:
                    pa = -pa
                pb = p - up
                if pb < 0:
                    pb = -pb
                pc = p - upleft
                if pc < 0:
                    pc = -pc
                if pa <= pb and pa <= pc:
                    pred = left
                elif pb <= pc:
                    pred = up
                else:
                    pred = upleft
                line[k] = (line[k] + pred) & 0xFF
        else:
            raise UnsupportedPNG(f"unknown row filter {ftype}")

        rows.append(bytes(line))
        prev = line
    return rows


def decode(data: bytes) -> tuple[list[bytes], int, int]:
    """Decode a PNG to RGB rows.

    Returns ``(rows, width, height)`` where each row is ``width * 3`` bytes.
    Alpha is composited onto white -- a preview is shown on a light surface
    and dropping alpha outright would turn transparent regions black.

    Raises :class:`UnsupportedPNG` for anything outside the narrow scope
    (16-bit, interlaced, unknown filters, truncated data).
    """
    if data[:8] != PNG_MAGIC:
        raise UnsupportedPNG("not a PNG (bad signature)")

    width = height = ctype = None
    idat: list[bytes] = []
    palette = b""
    pos = 8
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, ctype, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body[:13],
            )
            if depth != 8:
                raise UnsupportedPNG(f"bit depth {depth} (only 8 supported)")
            if interlace:
                raise UnsupportedPNG("interlaced PNG")
            if ctype not in _CHANNELS:
                raise UnsupportedPNG(f"colour type {ctype}")
        elif tag == b"PLTE":
            palette = body
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + length

    if width is None:
        raise UnsupportedPNG("no IHDR chunk")
    if not idat:
        raise UnsupportedPNG("no IDAT chunk")
    if ctype == 3 and not palette:
        raise UnsupportedPNG("indexed PNG without PLTE")

    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as e:
        raise UnsupportedPNG(f"corrupt pixel stream: {e}") from e

    bpp = _CHANNELS[ctype]
    rows = _unfilter(raw, width, height, bpp)

    # Normalise every colour type to plain RGB.
    if ctype == 2:
        return rows, width, height

    out: list[bytes] = []
    for line in rows:
        rgb = bytearray(width * 3)
        if ctype == 0:  # greyscale
            for x in range(width):
                v = line[x]
                rgb[x * 3] = rgb[x * 3 + 1] = rgb[x * 3 + 2] = v
        elif ctype == 4:  # greyscale + alpha
            for x in range(width):
                v, a = line[x * 2], line[x * 2 + 1]
                v = (v * a + 255 * (255 - a)) // 255
                rgb[x * 3] = rgb[x * 3 + 1] = rgb[x * 3 + 2] = v
        elif ctype == 6:  # RGBA
            for x in range(width):
                r, g, b, a = line[x * 4:x * 4 + 4]
                if a == 255:
                    rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2] = r, g, b
                else:
                    inv = 255 - a
                    rgb[x * 3] = (r * a + 255 * inv) // 255
                    rgb[x * 3 + 1] = (g * a + 255 * inv) // 255
                    rgb[x * 3 + 2] = (b * a + 255 * inv) // 255
        else:  # ctype == 3, indexed
            n = len(palette) // 3
            for x in range(width):
                i = line[x]
                if i >= n:
                    i = 0
                rgb[x * 3] = palette[i * 3]
                rgb[x * 3 + 1] = palette[i * 3 + 1]
                rgb[x * 3 + 2] = palette[i * 3 + 2]
        out.append(bytes(rgb))
    return out, width, height


def downscale(
    rows: list[bytes], width: int, height: int, max_dim: int,
) -> tuple[list[bytes], int, int]:
    """Box-average ``rows`` down so neither side exceeds ``max_dim``.

    Averaging (rather than nearest-neighbour sampling) matters because a
    preview of a screenshot or a detailed render turns into aliased noise
    when pixels are merely dropped. Sampling is capped at
    ``_MAX_SAMPLES`` per axis to keep the cost bounded in pure Python.
    """
    if width <= max_dim and height <= max_dim:
        return rows, width, height

    scale = max_dim / float(max(width, height))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    x_step = max(1, (width // new_w) // _MAX_SAMPLES or 1)
    y_step = max(1, (height // new_h) // _MAX_SAMPLES or 1)

    out: list[bytes] = []
    for ny in range(new_h):
        y0 = ny * height // new_h
        y1 = max(y0 + 1, (ny + 1) * height // new_h)
        src_rows = [rows[y] for y in range(y0, min(y1, height), y_step)] or [rows[y0]]
        line = bytearray(new_w * 3)
        for nx in range(new_w):
            x0 = nx * width // new_w
            x1 = max(x0 + 1, (nx + 1) * width // new_w)
            xs = range(x0, min(x1, width), x_step)
            r = g = b = n = 0
            for src in src_rows:
                for x in xs:
                    o = x * 3
                    r += src[o]
                    g += src[o + 1]
                    b += src[o + 2]
                    n += 1
            if n:
                o = nx * 3
                line[o] = r // n
                line[o + 1] = g // n
                line[o + 2] = b // n
        out.append(bytes(line))
    return out, new_w, new_h


def encode_rgb(rows: list[bytes], width: int, height: int, level: int = 9) -> bytes:
    """Re-encode RGB rows as a truecolour PNG (no colour loss)."""
    raw = b"".join(b"\x00" + row for row in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        PNG_MAGIC
        + _crc_chunk(b"IHDR", ihdr)
        + _crc_chunk(b"IDAT", zlib.compress(raw, level))
        + _crc_chunk(b"IEND", b"")
    )


def encode_palette(rows: list[bytes], width: int, height: int) -> bytes:
    """Re-encode as a 256-colour indexed PNG — 1 byte per pixel instead of 3.

    Uses a fixed 3-3-2 bit palette, so no clustering pass is needed. Lossy
    on colour, but roughly a third the size of :func:`encode_rgb`; the
    caller picks it only when truecolour will not fit the payload budget.
    """
    palette = bytearray()
    for i in range(256):
        palette += bytes((
            ((i >> 5) & 0x7) * 255 // 7,
            ((i >> 2) & 0x7) * 255 // 7,
            (i & 0x3) * 255 // 3,
        ))

    indexed: list[bytes] = []
    for row in rows:
        line = bytearray(width)
        for x in range(width):
            o = x * 3
            line[x] = ((row[o] >> 5) << 5) | ((row[o + 1] >> 5) << 2) | (row[o + 2] >> 6)
        indexed.append(bytes(line))

    raw = b"".join(b"\x00" + row for row in indexed)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    return (
        PNG_MAGIC
        + _crc_chunk(b"IHDR", ihdr)
        + _crc_chunk(b"PLTE", bytes(palette))
        + _crc_chunk(b"IDAT", zlib.compress(raw, 9))
        + _crc_chunk(b"IEND", b"")
    )
