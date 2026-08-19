"""Build a panel-sized preview of a generated image using only the stdlib.

Why a preview at all
--------------------
A panel response carries the image as a base64 ``data:`` URI, and there is
an undocumented ceiling on how big that response may be. Production
measurements:

===================  ==========
base64 chars         displays?
===================  ==========
~90,000              yes
~127,000             yes
~954,000             no
~1,250,000           no
===================  ==========

So instead of guessing where the wall is, this module targets the largest
size that is *proven* to work. A preview that fits under the proven-good
mark displays without needing the exact limit to be known.

Why stdlib only
---------------
The production runtime has no Pillow (``pillow_available: false``,
verified), and ``requirements.txt`` is not installed for extensions. Any
Pillow-based path is dead code there -- which is exactly why several
earlier "fixes" changed nothing for the user. :mod:`core.png` and
:mod:`core.jpeg` do the work with ``zlib`` + ``struct``, so this runs.

Both formats are handled, because the models return JPEG (verified on a
live generation: the bytes start ``ff d8``). A PNG-only implementation was
silently a no-op for every real render.
"""
from __future__ import annotations

import base64
import logging
import time

from core import jpeg as _jpeg
from core import jpeg_scaled as _jpeg_scaled
from core import png as _png
from core import webp as _webp

log = logging.getLogger("gemini.preview")

__all__ = [
    "PROVEN_GOOD_CHARS", "PREVIEW_BUDGET_CHARS",
    "build_preview", "can_preview", "sniff_format",
]

# The largest payload MEASURED to display in production. Not a guess, and
# deliberately not raised without a new measurement -- the previous cap
# (1,500,000) was invented, sat above every real render, and therefore
# never triggered the shrink it was supposed to trigger.
PROVEN_GOOD_CHARS = 127_000

# Target for a generated preview, with headroom below the proven mark for
# the surrounding JSON (prompt text, layout nodes, other list rows).
PREVIEW_BUDGET_CHARS = 110_000

# Longest edge to try, in order. Each step is attempted truecolour first,
# then as a 256-colour palette; the first result inside the budget wins.
_DIMENSION_LADDER = (640, 512, 448, 384, 320)

# Pure-Python decoding costs roughly 0.3s per megapixel. A 4K render
# (~9.4MP) is therefore several seconds -- acceptable once at generation
# time, but this ceiling keeps a pathological input from stalling a
# request. Skipped images fall back to being served whole.
_MAX_PIXELS = 12_000_000

# WebP (core/webp.py) is a pure-Python, unoptimized VP8L encoder -- no
# vectorization, an O(pixels x 14 modes) predictor search (bounded by
# sampling, see core.webp._MODE_SELECTION_SAMPLE_CAP) plus a full Huffman
# pass over every pixel. It is only ever tried at the FIRST (largest)
# _DIMENSION_LADDER rung (see build_preview), and this caps how large that
# first attempt is allowed to be in pixel count, independent of the source
# image's resolution -- a defensive ceiling so one pathological input can't
# make the one WebP attempt itself expensive, on top of the existing
# per-rung bound.
_WEBP_MAX_PIXELS = 200_000


def sniff_format(raw: bytes) -> str:
    """Identify image bytes by their magic number: ``"jpeg"``/``"png"``/``""``.

    The declared mime type is not trusted anywhere in this extension, for a
    reason paid for in production: the code once assumed PNG throughout while
    every real generation was JPEG, so previews silently never built. The
    first bytes of a file cannot lie the way a label can.

    Returns ``""`` for anything unrecognised -- callers decide whether that is
    fatal, rather than being handed a wrong guess.
    """
    if raw[:2] == b"\xff\xd8":
        return "jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return ""


def can_preview(mime_type: str) -> bool:
    """Whether :func:`build_preview` can handle this format at all.

    PNG and baseline JPEG. JPEG needs no inverse DCT here: each 8x8 block's
    DC coefficient is that block's average, so reading DC only yields a
    1/8-scale thumbnail (see :mod:`core.jpeg`) -- which is what a preview
    wants anyway.
    """
    fmt = (mime_type or "").lower()
    return "png" in fmt or "jpeg" in fmt or "jpg" in fmt


def build_preview(raw: bytes, mime_type: str) -> tuple[str, str] | None:
    """Return ``(base64_str, mime_type)`` for a preview inside the budget.

    Returns ``None`` when no preview could be produced -- unsupported
    format, undecodable bytes, or nothing in the ladder fitting. Callers
    must handle that instead of assuming success, so a failure degrades to
    an honest message rather than a blank panel.
    """
    if not can_preview(mime_type):
        return None

    started = time.monotonic()
    # Trust the bytes, not the declared type (see sniff_format).
    try:
        if sniff_format(raw) == "jpeg":
            # Half scale FIRST, DC-only as the fallback. The DC path decodes at
            # 1/8 scale, which meant a 1024px render previewed at 128px and was
            # then UPSCALED by the panel to fill the column -- that was the
            # blur. Half scale is a downscale of real detail instead.
            try:
                rows, width, height = _jpeg_scaled.decode_scaled(raw)
            except _jpeg.UnsupportedJPEG:
                raise
            except Exception as e:  # noqa: BLE001
                # Never lose the preview over the sharper path: fall back to
                # the decoder that has been serving production.
                log.info("preview: half-scale decode failed (%s), using DC", e)
                rows, width, height = _jpeg.decode_dc_thumbnail(raw)
        else:
            rows, width, height = _png.decode(raw)
    except (_png.UnsupportedPNG, _jpeg.UnsupportedJPEG) as e:
        log.info("preview: cannot decode image (%s)", e)
        return None
    except Exception as e:  # noqa: BLE001 — a corrupt file must not 500 the panel
        log.warning("preview: unexpected decode failure: %s", e)
        return None

    if width * height > _MAX_PIXELS:
        log.info("preview: %dx%d exceeds the pixel ceiling, skipping", width, height)
        return None

    for rung, max_dim in enumerate(_DIMENSION_LADDER):
        small, new_w, new_h = _png.downscale(rows, width, height, max_dim)
        encoders = [
            (_png.encode_rgb, "truecolour", "image/png"),
            (_png.encode_palette, "palette", "image/png"),
        ]
        if rung == 0 and new_w * new_h <= _WEBP_MAX_PIXELS:
            # WebP (VP8L lossless) first, but ONLY at the largest rung, and
            # only when that rung is small enough to bound the worst case
            # (see _WEBP_MAX_PIXELS): same pixel-for-pixel quality as the
            # PNG encoders below, and 20-50% smaller on photographic/smooth
            # content in practice here (measured, cross-validated against
            # Google's own dwebp decoder -- see core/webp.py). Trying it
            # first means it wins the budget at a LARGER max_dim than PNG
            # would have, i.e. a sharper preview for the same character
            # budget.
            #
            # Restricted to one rung deliberately: on genuinely noisy,
            # incompressible content (measured on a synthetic worst case)
            # this pure-Python encoder pays real time -- around a second per
            # rung at up to 640px -- at EVERY rung before losing anyway to
            # the much cheaper PNG palette path, adding ~3.5s per such image
            # for no benefit. One attempt bounds that cost; if WebP cannot
            # win on the largest, least-downscaled version of the image, the
            # smaller rungs are the lossy palette path's job.
            encoders.insert(0, (_webp.encode_rgb, "webp", "image/webp"))
        for encoder, label, out_mime in encoders:
            try:
                encoded = base64.b64encode(encoder(small, new_w, new_h)).decode()
            except Exception as e:  # noqa: BLE001
                log.warning("preview: %s encode failed: %s", label, e)
                continue
            if len(encoded) <= PREVIEW_BUDGET_CHARS:
                log.info(
                    "preview: %dx%d %s -> %d base64 chars in %.2fs",
                    new_w, new_h, label, len(encoded), time.monotonic() - started,
                )
                return encoded, out_mime

    log.info("preview: nothing in the ladder fit the %d-char budget", PREVIEW_BUDGET_CHARS)
    return None
