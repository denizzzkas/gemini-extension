"""Pure-stdlib WebP (lossless/VP8L) ENCODER — no Pillow, no third party.

Why this module exists
-----------------------
:mod:`core.png` already re-encodes previews without Pillow, because the
production runtime does not have it (``pillow_available: false``,
verified). WebP lossless (VP8L) is the next step: on photographic/smooth
content it is typically 20-50% smaller than PNG for the SAME lossless
quality, which buys headroom under :data:`core.preview.PREVIEW_BUDGET_CHARS`
without giving up any pixels. This is an encoder only -- nothing in this
extension ever needs to decode a ``.webp`` back, since a preview is written
once and handed straight to the browser as a ``data:`` URI.

The shared bit-writer / Huffman / image-data plumbing lives in
:mod:`core.webp_bits` (split out to stay under the deploy validator's
300-line-per-file cap -- same reason :mod:`core.jpeg_headers` exists).

How this was built and verified
--------------------------------
There is no way to "mostly" get a binary bitstream format right: either a
real decoder accepts it or it does not. Every stage below was cross-checked
against Google's own reference decoder (``dwebp``, from the ``webp``
Homebrew package) on this machine, not just visual inspection:

1. A minimal image (no transform) round-tripped through ``dwebp`` and was
   compared PIXEL BY PIXEL against the input.
2. The predictor transform was added and re-verified the same way on both
   a uniform image and a noisy gradient (photographic proxy).
3. A real, costly bug surfaced only at step 2's scale: Huffman CODES were
   being packed with the same bit-writer used for plain numeric fields --
   see :class:`core.webp_bits.BitWriter`'s docstring for the mechanism.
   Small/uniform test images never exposed this (bit order does not matter
   when every code in use has the same length), so it silently produced
   garbage pixels while still "successfully" decoding on skewed, realistic
   data.
See ``tests/test_webp_crossvalidate.py`` for the automated form of all of
the above, run against ``dwebp`` whenever it is on PATH -- it is what
caught a stale call site here (mismatched argument count) the moment this
module was split from a single file into this one plus
:mod:`core.webp_bits`, before it could reach production.

Scope, deliberately narrow
---------------------------
* RGB and RGBA, 8-bit, arbitrary width/height up to VP8L's 14-bit limit
  (16384 px/side) -- matches what :mod:`core.png` already hands this module.
* Transforms used: SUBTRACT_GREEN is NOT used; PREDICTOR_TRANSFORM is, with
  a single mode chosen for the WHOLE image (the general format allows a
  different mode per block; one block covering the image is a valid
  special case and is what libwebp itself falls back to for small images).
* NOT implemented: LZ77 backward references and the colour-cache
  transform. Both would shrink the output further (this is closer to
  "DEFLATE with only Huffman, no LZ77 matches" than to a tuned encoder),
  but each is a meaningful chunk of additional bitstream-format surface,
  and the predictor transform alone already beats :func:`core.png.encode_rgb`
  on measured gradient content. Adding them is a further optimisation, not
  a correctness requirement -- tracked as future work, not a hidden gap.
"""
from __future__ import annotations

import struct

from core.webp_bits import BitWriter, UnsupportedForWebP, encode_image_data

__all__ = ["UnsupportedForWebP", "encode_rgb", "encode_rgba"]

_SIGNATURE = 0x2F


# -- Predictor transform (spec 4.1) -----------------------------------------
#
# 14 prediction modes, selected ONCE for the whole image (a single block
# covering the full transform grid -- a valid special case of the general
# per-block format). All arithmetic operates on (a, r, g, b) tuples, alpha
# first, matching the spec's ARGB convention.

def _avg2(a: int, b: int) -> int:
    return (a + b) // 2


def _clamp(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


def _clamp_add_sub_full(l: tuple, t: tuple, tl: tuple) -> tuple:
    return tuple(_clamp(l[i] + t[i] - tl[i]) for i in range(4))


def _clamp_add_sub_half(avg: tuple, tl: tuple) -> tuple:
    return tuple(_clamp(avg[i] + (avg[i] - tl[i]) // 2) for i in range(4))


def _select(l: tuple, t: tuple, tl: tuple) -> tuple:
    cost_l = cost_t = 0
    for i in range(4):
        p = l[i] + t[i] - tl[i]
        cost_l += abs(p - l[i])
        cost_t += abs(p - t[i])
    return l if cost_l < cost_t else t


def _predict(mode: int, l: tuple, t: tuple, tl: tuple, tr: tuple) -> tuple:
    if mode == 0:
        return (255, 0, 0, 0)  # opaque black
    if mode == 1:
        return l
    if mode == 2:
        return t
    if mode == 3:
        return tr
    if mode == 4:
        return tl
    if mode == 5:
        avg_l_tr = tuple(_avg2(l[i], tr[i]) for i in range(4))
        return tuple(_avg2(avg_l_tr[i], t[i]) for i in range(4))
    if mode == 6:
        return tuple(_avg2(l[i], tl[i]) for i in range(4))
    if mode == 7:
        return tuple(_avg2(l[i], t[i]) for i in range(4))
    if mode == 8:
        return tuple(_avg2(tl[i], t[i]) for i in range(4))
    if mode == 9:
        return tuple(_avg2(t[i], tr[i]) for i in range(4))
    if mode == 10:
        a = tuple(_avg2(l[i], tl[i]) for i in range(4))
        b = tuple(_avg2(t[i], tr[i]) for i in range(4))
        return tuple(_avg2(a[i], b[i]) for i in range(4))
    if mode == 11:
        return _select(l, t, tl)
    if mode == 12:
        return _clamp_add_sub_full(l, t, tl)
    if mode == 13:
        avg = tuple(_avg2(l[i], t[i]) for i in range(4))
        return _clamp_add_sub_half(avg, tl)
    raise UnsupportedForWebP(f"predictor mode {mode} out of range 0..13")


def _apply_predictor(width: int, height: int, pixels_argb: list[tuple], mode: int) -> list[tuple]:
    """Return per-pixel residuals ``(actual - predicted) mod 256`` for
    ``mode``. Border pixels follow the spec's special cases: the top-left
    pixel predicts as opaque black, the whole top row predicts from its
    left neighbour, and the whole left column predicts from its top
    neighbour, regardless of ``mode``.
    """
    out: list[tuple] = [None] * (width * height)  # type: ignore[list-item]
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x == 0 and y == 0:
                pred = (255, 0, 0, 0)
            elif y == 0:
                pred = pixels_argb[idx - 1]
            elif x == 0:
                pred = pixels_argb[idx - width]
            else:
                l = pixels_argb[idx - 1]
                t = pixels_argb[idx - width]
                tl = pixels_argb[idx - width - 1]
                tr = pixels_argb[idx - width + 1] if x + 1 < width else t
                pred = _predict(mode, l, t, tl, tr)
            cur = pixels_argb[idx]
            out[idx] = tuple((cur[i] - pred[i]) % 256 for i in range(4))
    return out


def _choose_predictor_mode(width: int, height: int, pixels_argb: list[tuple]) -> tuple[int, list[tuple]]:
    """Try all 14 modes, keep the one with the smallest sum of residual
    magnitudes (signed-distance-to-zero) as a cheap proxy for entropy.

    An exact choice would build each mode's full Huffman histogram and
    compare total encoded bits; this proxy is far cheaper and tracks it
    closely for smooth/photographic content, which is what previews are.
    """
    best_mode, best_cost, best_residual = 0, None, None
    for mode in range(14):
        residual = _apply_predictor(width, height, pixels_argb, mode)
        cost = sum(min(v, 256 - v) for pixel in residual for v in pixel)
        if best_cost is None or cost < best_cost:
            best_mode, best_cost, best_residual = mode, cost, residual
    return best_mode, best_residual


def _encode_vp8l(width: int, height: int, pixels_rgba: list[tuple[int, int, int, int]]) -> bytes:
    if width <= 0 or height <= 0:
        raise UnsupportedForWebP(f"empty image {width}x{height}")
    if width > 16384 or height > 16384:
        raise UnsupportedForWebP(f"{width}x{height} exceeds VP8L's 14-bit (16384px) side limit")

    pixels_argb = [(a, r, g, b) for (r, g, b, a) in pixels_rgba]
    mode, residual_argb = _choose_predictor_mode(width, height, pixels_argb)
    residual_rgba = [(r, g, b, a) for (a, r, g, b) in residual_argb]

    bw = BitWriter()
    bw.write(_SIGNATURE, 8)
    bw.write(width - 1, 14)
    bw.write(height - 1, 14)
    alpha_is_used = 1 if any(a != 255 for _, _, _, a in pixels_rgba) else 0
    bw.write(alpha_is_used, 1)  # NOT purely informational: libwebp uses this
    # to decide whether to even emit an alpha plane on decode.
    bw.write(0, 3)  # version_number, fixed at 0

    # -- PREDICTOR_TRANSFORM, one block covering the whole image --
    bw.write(1, 1)  # a transform is present
    bw.write(0, 2)  # TransformType.PREDICTOR_TRANSFORM
    size_bits = 0
    while (1 << (size_bits + 2)) < max(width, height):
        size_bits += 1
    bw.write(size_bits, 3)
    block_size = 1 << (size_bits + 2)
    transform_width = (width + block_size - 1) // block_size
    transform_height = (height + block_size - 1) // block_size
    # The prediction-mode grid is itself a tiny VP8L image (green channel =
    # mode); with one block that is a single ARGB pixel.
    encode_image_data(
        bw,
        [(0, mode, 0, 255)] * (transform_width * transform_height),
        spatial=False,
    )
    bw.write(0, 1)  # no further transforms

    encode_image_data(bw, residual_rgba, spatial=True)

    payload = bw.to_bytes()
    vp8l_chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        vp8l_chunk += b"\x00"  # RIFF chunks are word-aligned
    riff_size = 4 + len(vp8l_chunk)  # "WEBP" + chunk
    return b"RIFF" + struct.pack("<I", riff_size) + b"WEBP" + vp8l_chunk


def encode_rgb(rows: list[bytes], width: int, height: int) -> bytes:
    """Encode opaque RGB rows (3 bytes/pixel, as produced by
    :func:`core.png.decode`/:func:`core.png.downscale`) as a lossless WebP.
    """
    pixels = []
    for row in rows:
        for x in range(width):
            o = x * 3
            pixels.append((row[o], row[o + 1], row[o + 2], 255))
    return _encode_vp8l(width, height, pixels)


def encode_rgba(rows: list[bytes], width: int, height: int) -> bytes:
    """Encode RGBA rows (4 bytes/pixel) as a lossless WebP, alpha preserved."""
    pixels = []
    for row in rows:
        for x in range(width):
            o = x * 4
            pixels.append((row[o], row[o + 1], row[o + 2], row[o + 3]))
    return _encode_vp8l(width, height, pixels)
