"""Baseline JPEG -> RGB at N/8 scale: a partial inverse DCT, not just DC.

The bug this fixes
------------------
:func:`core.jpeg.decode_dc_thumbnail` reads ONE coefficient per 8x8 block (the
DC term = that block's average), so it decodes at exactly 1/8 scale: a
1024x1024 render becomes 128x128. The panel then stretches that across a ~380px
column, so previews were an UPSCALED thumbnail -- that, not JPEG artefacts and
not the payload budget, is why they looked mushy. The size ladder could never
fix it, because ``core.png.downscale`` (rightly) refuses to upscale: the detail
was never decoded.

This decoder keeps the top-left NxN coefficients per block and evaluates a
partial IDCT at N positions per axis, giving N/8 scale -- so the preview is a
DOWNSCALE of real detail. :func:`choose_scale` picks the cheapest N that still
reaches :data:`TARGET_LONG_EDGE`, which keeps a big source on the cheap path
(it has plenty of detail at 1/8 already) and spends the work on small ones.

The basis weights
-----------------
Each output sample is the AVERAGE of the true 8-point IDCT over the pixels it
covers::

    M[j][u] = mean over x in group j of  C(u)/2 * cos((2x+1) u pi / 16)

with ``C(0) = 1/sqrt(2)``, ``C(u>0) = 1``, computed at import with :mod:`math`
rather than pasted in as magic numbers. That is what makes it checkable:
``M[j][0]`` is identical for every j, so a DC-only block decodes perfectly flat
and the N outputs average back to exactly the DC-only value. The tests assert
this against :func:`core.jpeg.decode_dc_thumbnail` on real JPEGs.

Cost: no NumPy, no Pillow (verified absent in production). The transform is
separable, runs once per image at preview-build time, and the result is cached
on the record. Above :data:`MAX_SCALED_PIXELS` it declines so a pathological
input degrades to the DC path instead of stalling a request.

Scope: baseline sequential Huffman JPEG (SOF0/SOF1), 8-bit, 1-3 components,
restart intervals. Anything else raises
:class:`core.jpeg_headers.UnsupportedJPEG` and the caller falls back.
"""
from __future__ import annotations

import math

from core.jpeg_headers import (
    UnsupportedJPEG, BitReader, ZIGZAG, clamp, decode_symbol, extend,
    parse_headers,
)

__all__ = [
    "decode_scaled", "SCALE_STEPS", "TARGET_LONG_EDGE", "MAX_SCALED_PIXELS",
    "choose_scale",
]

# Samples per 8-pixel block per axis. Each must divide 8, and n=1 is exactly the
# DC-only case, which is what makes this a superset of the old decoder.
SCALE_STEPS = (1, 2, 4, 8)

# The long edge the decode aims for: the smallest n whose output reaches this
# wins. The preview ladder in core.preview tops out at 640px and the panel
# column is ~380px, so decoding past this produces detail that is averaged away
# moments later. 440 is chosen so that a 4K render (3840px) is satisfied by the
# CHEAPEST setting, n=1 -> 480px, instead of paying 4x the work for a 960px
# intermediate that gets downscaled anyway.
TARGET_LONG_EDGE = 440

# Ceiling on SOURCE pixels this decoder will attempt at all; above it the caller
# falls back to DC-only rather than spending many seconds in pure Python.
MAX_SCALED_PIXELS = 40_000_000


def choose_scale(width: int, height: int) -> int:
    """Pick the CHEAPEST n whose output still reaches ``TARGET_LONG_EDGE``.

    This is what keeps the sharper decode affordable. Cost scales with the
    number of output samples, i.e. with n^2, so spending n=4 on an 8MP photo
    costs ~10s to produce a 2048px image that is then averaged down to 640 --
    all of it thrown away. A big source already has plenty of detail at 1/8
    scale; it is the SMALL source that needs the extra coefficients.

        4096px source -> n=1 (512px out)  cheap, and already enough
        1024px source -> n=4 (512px out)  the case that looked blurry
         256px source -> n=8 (256px out)  full detail, still tiny
    """
    long_edge = max(width, height)
    if long_edge <= 0:
        return 1
    for n in SCALE_STEPS:
        if long_edge * n // 8 >= TARGET_LONG_EDGE:
            return n
    return SCALE_STEPS[-1]


def _basis(n: int) -> list[list[float]]:
    """``M[j][u]``: weight of coefficient u in output sample j.

    The mean of the 8-point IDCT basis over the pixels that output sample j
    covers. Computed, not hardcoded, so the DC column is provably flat.
    """
    group = 8 // n
    m: list[list[float]] = []
    for j in range(n):
        row: list[float] = []
        for u in range(n):
            cu = (1.0 / math.sqrt(2.0)) if u == 0 else 1.0
            acc = 0.0
            for x in range(j * group, (j + 1) * group):
                acc += math.cos((2 * x + 1) * u * math.pi / 16.0)
            row.append(cu * 0.5 * acc / group)
        m.append(row)
    return m


def _keep_map(n: int) -> dict[int, tuple[int, int]]:
    """Zig-zag index -> (row, col), for coefficients this scale actually uses.

    A coefficient matters only when BOTH its row and col are < n; everything
    else is a frequency the output grid cannot represent.
    """
    keep: dict[int, tuple[int, int]] = {}
    for k, pos in enumerate(ZIGZAG):
        r, c = divmod(pos, 8)
        if r < n and c < n:
            keep[k] = (r, c)
    return keep


# Built once per scale on first use rather than at import: only the scales the
# user's actual images need get computed.
_BASIS_CACHE: dict[int, list[list[float]]] = {}
_KEEP_CACHE: dict[int, dict[int, tuple[int, int]]] = {}


def _idct_block(
    coef: list[list[float]], n: int, m: list[list[float]],
) -> list[list[float]]:
    """Separable partial IDCT of the top-left ``n x n`` coefficients."""
    # Rows first: for each coefficient row v, turn n horizontal coefficients
    # into n horizontal samples.
    tmp = [[0.0] * n for _ in range(n)]
    for v in range(n):
        cv = coef[v]
        # Skip an all-zero row -- common, and this is the hot loop.
        if not any(cv):
            continue
        out = tmp[v]
        for j in range(n):
            mj = m[j]
            acc = 0.0
            for u in range(n):
                if cv[u]:
                    acc += cv[u] * mj[u]
            out[j] = acc
    # Then columns.
    res = [[0.0] * n for _ in range(n)]
    for j in range(n):
        col = [tmp[v][j] for v in range(n)]
        if not any(col):
            continue
        for i in range(n):
            mi = m[i]
            acc = 0.0
            for v in range(n):
                if col[v]:
                    acc += col[v] * mi[v]
            res[i][j] = acc
    return res


def decode_scaled(data: bytes, scale: int = 0) -> tuple[list[bytes], int, int]:
    """Decode a baseline JPEG at ``scale``/8 of its size.

    Returns ``(rows, width, height)`` with each row ``width * 3`` bytes -- the
    same shape :func:`core.png.encode_rgb` consumes, identical in contract to
    :func:`core.jpeg.decode_dc_thumbnail`, just at a higher resolution.

    ``scale`` defaults to 0, meaning "choose it from the image's own size" via
    :func:`choose_scale`, which is what keeps the cost sane: only images that
    are actually small get the expensive high-coefficient treatment.
    """
    h = parse_headers(data)

    if h.width * h.height > MAX_SCALED_PIXELS:
        raise UnsupportedJPEG(
            f"{h.width}x{h.height} is above the scaled-decode ceiling"
        )

    n = scale or choose_scale(h.width, h.height)
    if n not in SCALE_STEPS:
        raise UnsupportedJPEG(f"scale {n} does not divide 8")

    m = _BASIS_CACHE.get(n)
    if m is None:
        m = _BASIS_CACHE[n] = _basis(n)
    keep_map = _KEEP_CACHE.get(n)
    if keep_map is None:
        keep_map = _KEEP_CACHE[n] = _keep_map(n)

    hmax, vmax = h.hmax, h.vmax
    mcus_x, mcus_y = h.mcu_grid()

    for c in h.comps:
        c["bw"] = mcus_x * c["h"]
        c["bh"] = mcus_y * c["v"]
        c["sw"] = c["bw"] * n
        c["sh"] = c["bh"] * n
        c["s"] = [[0.0] * c["sw"] for _ in range(c["sh"])]
        c["pred"] = 0

    br = BitReader(data, h.scan_pos)
    mcu_count = 0
    for my in range(mcus_y):
        for mx in range(mcus_x):
            if h.restart_interval and mcu_count and mcu_count % h.restart_interval == 0:
                if br.skip_restart_marker():
                    for c in h.comps:
                        c["pred"] = 0
            for c in h.comps:
                dc_tbl, ac_tbl = h.tables_for(c)
                q = h.quant_for(c)
                for by in range(c["v"]):
                    for bx in range(c["h"]):
                        coef = [[0.0] * n for _ in range(n)]

                        t = decode_symbol(br, dc_tbl)
                        c["pred"] += extend(br.bits(t), t)
                        coef[0][0] = c["pred"] * q[0]

                        k = 1
                        while k < 64:
                            rs = decode_symbol(br, ac_tbl)
                            run, size = rs >> 4, rs & 0x0F
                            if size == 0:
                                if run != 15:
                                    break       # EOB
                                k += 16
                                continue
                            k += run
                            if k >= 64:
                                break
                            raw = extend(br.bits(size), size)
                            keep = keep_map.get(k)
                            if keep is not None and raw:
                                # Quant tables are stored in zig-zag order, so
                                # index the table by k, not by (row, col).
                                r, cc = keep
                                coef[r][cc] = raw * q[k]
                            k += 1

                        # DC is handled separately from the AC scaling above so
                        # the level shift is added exactly once.
                        block = _idct_block(coef, n, m)

                        sy = (my * c["v"] + by) * n
                        sx = (mx * c["h"] + bx) * n
                        grid = c["s"]
                        for i in range(n):
                            grow = grid[sy + i]
                            brow = block[i]
                            for j in range(n):
                                grow[sx + j] = brow[j] + 128.0
            mcu_count += 1

    # The true output size is the source size scaled by n/8. The decoded sample
    # grid is larger than that whenever the image does not end on an exact MCU
    # boundary (JPEG pads to whole MCUs), so the grid is an upper bound to clamp
    # against -- never a substitute for the real dimensions. Getting this wrong
    # returned an image 8/n times too large on any file without chroma
    # subsampling, because the old clamp compared luma samples against luma
    # samples and therefore never clamped at all.
    y_c = h.comps[0]
    out_w = max(1, min(h.width * n // 8, y_c["sw"] * hmax // y_c["h"]))
    out_h = max(1, min(h.height * n // 8, y_c["sh"] * vmax // y_c["v"]))

    chroma = h.comps[1:] if len(h.comps) == 3 else []

    rows: list[bytes] = []
    for oy in range(out_h):
        row = bytearray()
        yy_row = y_c["s"][min(oy * y_c["v"] // vmax, y_c["sh"] - 1)]
        cb_row = cr_row = None
        if chroma:
            cb_c, cr_c = chroma[0], chroma[1]
            cb_row = cb_c["s"][min(oy * cb_c["v"] // vmax, cb_c["sh"] - 1)]
            cr_row = cr_c["s"][min(oy * cr_c["v"] // vmax, cr_c["sh"] - 1)]
        for ox in range(out_w):
            yv = yy_row[min(ox * y_c["h"] // hmax, y_c["sw"] - 1)]
            if cb_row is None:
                v = clamp(yv)
                row += bytes((v, v, v))
                continue
            cb = cb_row[min(ox * cb_c["h"] // hmax, cb_c["sw"] - 1)] - 128.0
            cr = cr_row[min(ox * cr_c["h"] // hmax, cr_c["sw"] - 1)] - 128.0
            row += bytes((
                clamp(yv + 1.402 * cr),
                clamp(yv - 0.344136 * cb - 0.714136 * cr),
                clamp(yv + 1.772 * cb),
            ))
        rows.append(bytes(row))
    return rows, out_w, out_h
