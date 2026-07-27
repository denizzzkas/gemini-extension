"""Tests for the partial-IDCT JPEG decoder behind sharper previews.

The bug being guarded
---------------------
Previews were decoded DC-only, i.e. at 1/8 scale, and the panel then stretched
that thumbnail across its column -- so a 1024px render was shown as a 128px
image blown up ~3x. It read as "blurry previews" and no amount of tuning the
size ladder could fix it, because ``core.png.downscale`` refuses to upscale:
the detail had never been decoded.

What makes these tests worth trusting
-------------------------------------
The strongest assertion here does NOT check pixels against hand-copied numbers
(that only proves the decoder still agrees with whatever it did yesterday). It
checks the partial IDCT against an INDEPENDENT, already-trusted decoder: average
the N x N samples this decoder produces for one block, and the result must equal
the DC coefficient ``core.jpeg.decode_dc_thumbnail`` reads for that same block,
because the DC term IS the block mean by definition. Any error in the basis
weights, the zig-zag order, the dequantisation or the level shift breaks that
identity. See ``test_scaled_samples_average_back_to_the_dc_value``.

Fixtures come from an independent JPEG encoder (macOS ``sips``) rather than a
hand-rolled one, so the tests cannot pass by a decoder agreeing with a matching
bug in a local encoder. Where that encoder is unavailable the tests skip rather
than silently assert nothing. Nothing here imports Pillow: production has none
(verified), and a test that imported PIL previously failed a deploy.
"""
from __future__ import annotations

import statistics

import pytest

from core import jpeg, jpeg_scaled
from tests.test_jpeg_preview import _make_jpeg


def _luma(r: int, g: int, b: int) -> float:
    """BT.601 luma -- the Y that JPEG actually codes, recovered from RGB.

    The comparison below MUST be done in luma, not in a colour channel. Chroma
    is subsampled, so one Cb/Cr sample is shared between neighbouring luma
    blocks; R, G and B each mix that shared chroma in, and a per-block mean of
    R therefore does not have to equal the DC decoder's R for the same block.
    Measured on a real fixture, that mixing alone accounts for ~5 levels of
    apparent disagreement while luma agrees to ~0.2 -- i.e. checking R would
    have meant either a meaningless tolerance or chasing a bug that is not
    there. Converting back to Y cancels the chroma contribution exactly.
    """
    return 0.299 * r + 0.587 * g + 0.114 * b


def _block_luma_mean(rows: list[bytes], w: int, h: int,
                     bx: int, by: int, n: int) -> float:
    """Mean luma over the n x n samples belonging to block (bx, by)."""
    vals = [
        _luma(rows[y][x * 3], rows[y][x * 3 + 1], rows[y][x * 3 + 2])
        for y in range(by * n, min((by + 1) * n, h))
        for x in range(bx * n, min((bx + 1) * n, w))
    ]
    return statistics.fmean(vals) if vals else 0.0


def test_scaled_decoding_beats_dc_only_on_resolution():
    """THE regression guard: the preview must not be a 1/8-scale thumbnail.

    This is the bug the user reported as blurry previews. A 512px source
    decoded DC-only gives 64px; the panel column is ~380px wide, so it was
    displayed at roughly 6x magnification.
    """
    data = _make_jpeg(512, 512)
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    _dc_rows, dc_w, dc_h = jpeg.decode_dc_thumbnail(data)
    _sc_rows, sc_w, sc_h = jpeg_scaled.decode_scaled(data)

    assert sc_w > dc_w and sc_h > dc_h, "scaled decode must beat DC-only"
    # 512px source -> n=8 -> full scale, i.e. 8x the DC-only linear resolution.
    assert sc_w >= dc_w * 4, (
        f"expected a large resolution gain, got {dc_w}px -> {sc_w}px"
    )


@pytest.mark.parametrize("n", jpeg_scaled.SCALE_STEPS)
def test_scaled_samples_average_back_to_the_dc_value(n: int):
    """Cross-checks the partial IDCT against an INDEPENDENT decoder.

    A block's DC coefficient is by definition that block's mean, so averaging
    the n x n samples this decoder emits for a block MUST reproduce the value
    ``decode_dc_thumbnail`` reads for it. Wrong basis weights, a wrong zig-zag
    mapping, a missed dequantisation or a double level shift all break the
    identity -- which is why this is asserted at every supported scale rather
    than only the default one.

    The comparison is in LUMA for the reason documented on :func:`_luma`, which
    is also what allows the tolerance to be tight: sub-level agreement, not a
    band loose enough to hide a real error.
    """
    data = _make_jpeg(256, 128)
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    dc_rows, dc_w, dc_h = jpeg.decode_dc_thumbnail(data)
    rows, w, h = jpeg_scaled.decode_scaled(data, scale=n)

    assert len(rows) == h and all(len(r) == w * 3 for r in rows), \
        "rows must be exactly width*3 bytes -- core.png.encode_rgb assumes it"

    deltas = []
    for by in range(min(dc_h, h // n)):
        for bx in range(min(dc_w, w // n)):
            dc_y = _luma(
                dc_rows[by][bx * 3], dc_rows[by][bx * 3 + 1], dc_rows[by][bx * 3 + 2],
            )
            deltas.append(abs(_block_luma_mean(rows, w, h, bx, by, n) - dc_y))

    assert deltas, "fixture produced no comparable blocks"
    mean_delta = statistics.fmean(deltas)
    assert mean_delta < 1.0, (
        f"n={n}: block-mean luma disagrees with the DC decoder by "
        f"{mean_delta:.2f} levels on average -- the partial IDCT is wrong "
        f"(basis weights, zig-zag order, dequantisation or level shift)"
    )
    assert max(deltas) < 4.0, (
        f"n={n}: worst block is off by {max(deltas):.2f} levels"
    )


def test_dc_only_scale_reproduces_the_old_decoder_exactly():
    """``n=1`` must BE the DC-only case, not merely resemble it.

    This is what makes the new decoder a superset of the proven one: at the
    cheapest setting it is the same computation, so falling back to it can
    never change what a user sees.
    """
    data = _make_jpeg(128, 64)
    if data is None:
        pytest.skip("no JPEG encoder available to build a fixture")

    dc_rows, dc_w, dc_h = jpeg.decode_dc_thumbnail(data)
    rows, w, h = jpeg_scaled.decode_scaled(data, scale=1)

    assert (w, h) == (dc_w, dc_h), \
        f"n=1 must match DC-only geometry, got {w}x{h} vs {dc_w}x{dc_h}"

    deltas = [
        abs(rows[y][x * 3] - dc_rows[y][x * 3])
        for y in range(h) for x in range(w)
    ]
    assert max(deltas) <= 1, \
        f"n=1 must reproduce DC-only luma (max delta {max(deltas)})"


def test_output_dimensions_are_the_true_scaled_size():
    """Geometry must follow the IMAGE size, not the padded MCU grid.

    JPEG pads to whole MCUs, so the decoded sample grid is often larger than
    the picture. An earlier version clamped the output against the luma grid
    in luma units -- a comparison that never binds -- and returned an image
    8/n times too large for any file without chroma subsampling.
    """
    for src_w, src_h in ((256, 128), (200, 100), (129, 67)):
        data = _make_jpeg(src_w, src_h)
        if data is None:
            pytest.skip("no JPEG encoder available to build a fixture")
        for n in jpeg_scaled.SCALE_STEPS:
            _rows, w, h = jpeg_scaled.decode_scaled(data, scale=n)
            exp_w, exp_h = max(1, src_w * n // 8), max(1, src_h * n // 8)
            assert abs(w - exp_w) <= 1 and abs(h - exp_h) <= 1, (
                f"{src_w}x{src_h} at n={n}: expected ~{exp_w}x{exp_h}, "
                f"got {w}x{h}"
            )


def test_choose_scale_spends_effort_only_where_it_helps():
    """Cost control: a big source must NOT get the expensive treatment.

    Work scales with n^2. A 4K render already carries plenty of detail at 1/8
    scale, so paying for more produces an intermediate that is immediately
    averaged down to the preview size -- pure waste, and measured in SECONDS of
    pure-Python decoding. The small source is the one that needs the
    coefficients.
    """
    assert jpeg_scaled.choose_scale(3840, 2160) == 1, \
        "a 4K source must use the cheapest scale"
    assert jpeg_scaled.choose_scale(256, 256) == 8, \
        "a small source must use the sharpest scale"

    # Monotonic: a larger source never costs a bigger n than a smaller one.
    scales = [jpeg_scaled.choose_scale(w, w) for w in (256, 512, 1024, 2048, 3840)]
    assert scales == sorted(scales, reverse=True), \
        f"choose_scale must not increase with source size, got {scales}"

    # Whatever it picks must actually clear the target it exists to hit.
    for w in (512, 1024, 2048, 3840):
        n = jpeg_scaled.choose_scale(w, w)
        if n < jpeg_scaled.SCALE_STEPS[-1]:
            assert w * n // 8 >= jpeg_scaled.TARGET_LONG_EDGE, (
                f"{w}px -> n={n} yields {w * n // 8}px, below the target"
            )


def test_a_corrupt_jpeg_fails_loudly_rather_than_returning_garbage():
    """The preview path catches failures and falls back -- so it must SEE them.

    Returning a half-decoded image instead of raising would show the user a
    corrupted preview and call it the result.
    """
    with pytest.raises(Exception):
        jpeg_scaled.decode_scaled(b"\xff\xd8\xff\xd9")
    with pytest.raises(Exception):
        jpeg_scaled.decode_scaled(b"not a jpeg at all")


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
    r = subprocess.run(
        ["sips", "-s", "format", "png", src, "--out", dst], capture_output=True,
    )
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
    r = subprocess.run(
        ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "best",
         src, "--out", dst],
        capture_output=True,
    )
    if r.returncode != 0:
        return None
    with open(dst, "rb") as fh:
        return fh.read()


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
