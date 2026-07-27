"""Tests for accepting a user-supplied image as a generation reference.

The gap being closed: references could ONLY be past generations of your own,
so there was no way to say "use THIS photo" from the panel or from chat.

The properties worth pinning are the ones that have already cost this
extension real breakage:

  * the CACHED preview must be the shrunk one -- caching full-size base64 is
    exactly how the panel ended up unable to display anything,
  * format is decided by MAGIC BYTES, never by the declared mime (a
    mislabelled image made the whole preview path a silent no-op once),
  * an unreadable upload must SAY so rather than fail quietly,
  * an upload must land in the store in the shape the EXISTING reference
    resolver already reads, otherwise the feature only looks finished.
"""
from __future__ import annotations

import base64
import random

import pytest

from core import png
from core.preview import PREVIEW_BUDGET_CHARS
from gemini_config import GENERATION_LOG_COLLECTION
from handlers.image_loader import PREVIEW_FIELD
from handlers.uploads import (
    MAX_UPLOAD_BYTES,
    UploadReferenceParams,
    _extract_upload,
    fn_upload_reference_image,
)
from tests.fixtures import make_ctx



def _real_png(w: int = 420, h: int = 420) -> bytes:
    """A genuine, decodable PNG with incompressible-ish noise.

    Noise matters: a flat colour compresses to almost nothing and would sail
    under any budget, hiding the very size behaviour these tests check.
    """
    rnd = random.Random(7)
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            row += bytes((
                (x * 255 // w + rnd.randint(-25, 25)) & 0xFF,
                (y * 255 // h + rnd.randint(-25, 25)) & 0xFF,
                ((x + y) * 255 // (w + h) + rnd.randint(-25, 25)) & 0xFF,
            ))
        rows.append(bytes(row))
    return png.encode_rgb(rows, w, h)


def _as_upload(raw: bytes, name: str = "ref.png", mime: str = "image/png") -> dict:
    return {"name": name, "mime_type": mime, "data": base64.b64encode(raw).decode()}


# ── the wire shape is undocumented, so tolerance is a feature ──────────


def test_extract_accepts_a_plain_base64_string():
    raw = b"\x89PNG\r\n\x1a\n" + b"rest"
    got, name, mime = _extract_upload(base64.b64encode(raw).decode())
    assert got == raw


def test_extract_accepts_a_data_uri():
    raw = b"\xff\xd8\xff\xe0" + b"jpegish"
    uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    got, name, mime = _extract_upload({"name": "a.jpg", "data": uri})
    assert got == raw


def test_extract_reports_an_unrecognisable_item_instead_of_guessing():
    """Silence here would look like 'upload worked' while nothing was stored."""
    with pytest.raises(ValueError):
        _extract_upload({"unexpected": "shape"})


# ── the behaviour that actually burned the panel before ───────────────


@pytest.mark.asyncio
async def test_cached_preview_is_the_shrunk_one_not_the_original():
    """Caching full-size base64 is what made images undisplayable before.

    The stored preview must be far below the display budget even though the
    uploaded original is comfortably above it.
    """
    ctx = make_ctx()
    raw = _real_png()
    original_b64_len = len(base64.b64encode(raw).decode())

    result = await fn_upload_reference_image(
        ctx, UploadReferenceParams(files=[_as_upload(raw)])
    )
    assert result.status == "success"

    gen_id = result.data["generation_ids"][0]
    doc = await ctx.store.get(GENERATION_LOG_COLLECTION, gen_id)
    cached = doc.data.get(PREVIEW_FIELD) or ""

    assert cached, "no preview was cached at upload time"
    assert len(cached) <= PREVIEW_BUDGET_CHARS
    # The real regression guard: not merely 'small', but genuinely shrunk.
    assert len(cached) < original_b64_len


@pytest.mark.asyncio
async def test_format_comes_from_the_bytes_not_the_declared_mime():
    """A PNG mislabelled as JPEG must still be stored correctly."""
    ctx = make_ctx()
    raw = _real_png(64, 64)

    result = await fn_upload_reference_image(
        ctx,
        UploadReferenceParams(files=[_as_upload(raw, "lying.jpg", "image/jpeg")]),
    )

    assert result.status == "success"
    assert result.data["stored"][0]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_upload_is_stored_in_the_shape_the_reference_resolver_reads():
    """An upload is only useful if the EXISTING resolver can pick it up.

    _resolve_reference_images requires: owned by the caller, kind == 'image',
    and a non-empty storage_path. Miss any one and the feature silently does
    nothing at generation time.
    """
    ctx = make_ctx()

    result = await fn_upload_reference_image(
        ctx, UploadReferenceParams(files=[_as_upload(_real_png(64, 64))])
    )
    gen_id = result.data["generation_ids"][0]
    doc = await ctx.store.get(GENERATION_LOG_COLLECTION, gen_id)

    assert doc.data["user_id"] == ctx.user.imperal_id
    assert doc.data["kind"] == "image"
    assert doc.data["storage_path"]


@pytest.mark.asyncio
async def test_non_image_bytes_are_rejected_with_a_reason():
    ctx = make_ctx()

    result = await fn_upload_reference_image(
        ctx,
        UploadReferenceParams(files=[_as_upload(b"this is not an image at all")]),
    )

    assert result.status == "error"
    assert "PNG or JPEG" in result.error
    assert result.retryable is False


@pytest.mark.asyncio
async def test_oversized_upload_is_refused_before_it_is_stored():
    ctx = make_ctx()
    huge = b"\x89PNG\r\n\x1a\n" + b"x" * (MAX_UPLOAD_BYTES + 1)

    result = await fn_upload_reference_image(
        ctx, UploadReferenceParams(files=[_as_upload(huge)])
    )

    assert result.status == "error"
    assert await ctx.store.count(GENERATION_LOG_COLLECTION) == 0


@pytest.mark.asyncio
async def test_one_bad_file_does_not_discard_the_good_one():
    """Partial success must still return the usable reference."""
    ctx = make_ctx()

    result = await fn_upload_reference_image(
        ctx,
        UploadReferenceParams(files=[
            _as_upload(b"garbage", "bad.png"),
            _as_upload(_real_png(64, 64), "good.png"),
        ]),
    )

    assert result.status == "success"
    assert len(result.data["generation_ids"]) == 1
    assert result.data["skipped"], "the rejected file must be reported, not hidden"


@pytest.mark.asyncio
async def test_no_files_says_so_plainly():
    ctx = make_ctx()
    result = await fn_upload_reference_image(ctx, UploadReferenceParams(files=[]))
    assert result.status == "error"
