"""Cross-validating the JPEG decoder against an INDEPENDENT implementation.

Split from ``test_jpeg_scaled.py`` to keep both files under the 300-line limit
the deploy validator enforces.

Why a separate oracle is necessary
----------------------------------
The DC-identity test in ``test_jpeg_scaled.py`` is mathematically BLIND to the
AC weights: averaging the n x n samples of a block cancels every AC basis
function by construction, so that identity holds no matter what the AC weights
are. This was verified, not assumed -- scaling every AC weight by 1.02 left it
entirely green. It pins the DC term, the zig-zag order, the dequantisation and
the level shift, and nothing more.

At ``n = 8`` nothing is discarded, so the decoder is a FULL baseline JPEG
decoder and can be compared pixel for pixel against a completely separate
implementation: Apple ImageIO, reached through ``sips``. Every AC weight has to
be right for that comparison to hold.

Both helpers degrade to ``None`` when ``sips`` is missing, and the test then
SKIPS. That is not cosmetic: production runs Linux, where an uncaught
``FileNotFoundError`` in a fixture fails the entire deploy -- which is exactly
what it did once.
"""
from __future__ import annotations

import statistics

import pytest

from core import jpeg_scaled


def _imageio_decode(jpeg_bytes: bytes):
    """Decode a JPEG with Apple ImageIO (``sips``) -> (rows, w, h), or None.

    This is the INDEPENDENT oracle. ``sips`` converts the JPEG to PNG, which
    this repo's own trusted PNG decoder then reads, so the pixels come from a
    completely separate JPEG implementation -- no shared code, no shared bug.
    """
    import subprocess
    import tempfile

    from core import png

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(jpeg_bytes)
        src = f.name
    dst = src.replace(".jpg", ".png")
    # Same reasoning as _high_detail_jpeg: no sips on Linux, so this must
    # degrade to "no oracle available" rather than raise.
    try:
        r = subprocess.run(
            ["sips", "-s", "format", "png", src, "--out", dst],
            capture_output=True, timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        with open(dst, "rb") as fh:
            return png.decode(fh.read())
    except Exception:  # noqa: BLE001
        return None


def _high_detail_jpeg(width: int = 128, height: int = 64) -> bytes | None:
    """A GREY, high-frequency fixture: maximum AC energy, no chroma subsampling.

    Both properties are deliberate.

    *High frequency* -- a smooth gradient carries almost no AC energy, so an
    error in the AC weights moves the pixels by less than a level and hides in
    rounding. A 3px checkerboard plus noise puts real energy in the high
    coefficients, which is what makes them measurable.

    *Grey* -- with flat chroma the encoder emits 4:4:4, which removes chroma
    upsampling from the comparison. That matters because this decoder upsamples
    chroma by REPLICATION while ImageIO interpolates: on a 4:2:0 fixture with
    saturated colour that difference alone is ~30 levels of RGB disagreement
    (measured) even though luma agrees to 0.55, and it would swamp exactly the
    signal this test is trying to read. See the note in the test body.
    """
    import random
    import subprocess
    import tempfile

    from core import png

    random.seed(11)
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            v = 230 if ((x // 3 + y // 3) % 2) else 25
            v = max(0, min(255, v + random.randint(-18, 18)))
            row += bytes((v, v, v))
        rows.append(bytes(row))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png.encode_rgb(rows, width, height))
        src = f.name
    dst = src.replace(".png", ".jpg")
    # FileNotFoundError is the case that matters: ``sips`` is macOS-only, and
    # production runs Linux, where an uncaught raise here fails the whole
    # deploy instead of skipping a test that cannot run. Returning None lets
    # the caller skip honestly.
    try:
        r = subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "best",
             src, "--out", dst],
            capture_output=True, timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        with open(dst, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def test_full_scale_decode_matches_an_independent_jpeg_decoder():
    """Cross-validate the AC coefficients against Apple ImageIO, pixel for pixel.

    Why this is needed ON TOP of the DC-identity test
    -------------------------------------------------
    The DC-identity test is mathematically BLIND to the AC weights: averaging
    the n x n samples of a block cancels every AC basis function by
    construction, so that identity holds regardless of what the AC weights are.
    Verified rather than assumed -- scaling every AC weight by 1.02 left it
    completely green. It pins the DC term, the zig-zag order, the dequantisation
    and the level shift, and nothing more.

    This test closes that hole. At ``n=8`` nothing is discarded, so the decoder
    is a FULL baseline JPEG decoder and can be compared pixel for pixel against
    a completely independent implementation (Apple ImageIO via ``sips``). Every
    AC weight has to be right for that to hold.

    The fixture is high-frequency AND grey on purpose -- see
    :func:`_high_detail_jpeg`. A smooth gradient has too little AC energy for a
    weight error to be visible, and a colourful 4:2:0 fixture would be dominated
    by a chroma-upsampling difference (replication here vs interpolation in
    ImageIO) that is a deliberate cost choice, not a defect: measured at ~30
    levels of RGB while luma still agreed to 0.55.
    """
    data = _high_detail_jpeg()
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    reference = _imageio_decode(data)
    if reference is None:
        pytest.skip("no independent JPEG decoder available")

    ref_rows, rw, rh = reference
    rows, w, h = jpeg_scaled.decode_scaled(data, scale=8)

    assert (w, h) == (rw, rh), (
        f"at n=8 the decode must be full size: got {w}x{h}, reference {rw}x{rh}"
    )

    deltas = [
        abs(ref_rows[y][x * 3 + c] - rows[y][x * 3 + c])
        for y in range(rh) for x in range(rw) for c in range(3)
    ]
    mean_delta = statistics.fmean(deltas)

    # Tight on purpose. Two conforming IDCT implementations may differ by about
    # a level from rounding, so anything beyond that is a real disagreement --
    # this caught a transposed zig-zag and a missing level shift, and rejects
    # a 2% error in the AC weights.
    assert mean_delta < 1.5, (
        f"decode disagrees with an independent JPEG decoder by {mean_delta:.2f} "
        f"levels on average -- the AC coefficients are wrong"
    )
    assert max(deltas) <= 12, (
        f"worst-pixel disagreement {max(deltas)} is too large for rounding"
    )
