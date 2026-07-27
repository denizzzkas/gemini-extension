"""Tests for the centre detail view: one generation, in full.

The properties pinned here are the ones the user actually asked for, plus the
ones this extension has already been burned by:

  * the download anchor must carry the ORIGINAL bytes -- handing over the
    shrunk preview while calling it "the original" is the exact dishonesty
    this view exists to end,
  * the "shown at preview size" note must appear only when what is displayed
    really is smaller than the original,
  * "Regenerate" must call the per-model tool matching the model that made the
    image: Imperal prices a TOOL, and these models differ several-fold in
    cost, so the wrong target bills the wrong amount,
  * the model->tool map must not drift away from the really-registered tools.
"""
from __future__ import annotations

import base64
import random

import pytest

from core import png
from gemini_config import (
    GENERATION_LOG_COLLECTION, IMAGE_TOOL_FOR_MODEL, MODEL_IMAGE_FLASH,
)
from handlers.panel_detail import detail_view, load_detail
from tests.fixtures import make_ctx


def _real_png(w: int = 380, h: int = 380) -> bytes:
    """A genuine PNG with noise, so it cannot compress away to nothing."""
    rnd = random.Random(3)
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            row += bytes((
                (x * 255 // w + rnd.randint(-30, 30)) & 0xFF,
                (y * 255 // h + rnd.randint(-30, 30)) & 0xFF,
                ((x * y) % 255),
            ))
        rows.append(bytes(row))
    return png.encode_rgb(rows, w, h)


def _walk(node, acc=None):
    """Flatten a serialized UI tree into a list of dicts."""
    acc = [] if acc is None else acc
    if isinstance(node, dict):
        acc.append(node)
        for v in node.values():
            _walk(v, acc)
    elif isinstance(node, list):
        for item in node:
            _walk(item, acc)
    return acc


def _of_type(tree, type_name):
    return [n for n in _walk(tree) if n.get("type") == type_name]


async def _seed(ctx, *, raw: bytes, model: str = MODEL_IMAGE_FLASH,
                prompt: str = "a lighthouse in fog", reference_ids=None):
    path = "gemini/image/detail-test.png"
    await ctx.storage.upload(path, raw, content_type="image/png")
    doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": prompt,
        "model": model,
        "storage_path": path,
        "mime_type": "image/png",
        "source": "generated",
        "reference_ids": list(reference_ids or []),
        "created_at": "2026-07-27T10:00:00Z",
    })
    return doc


@pytest.mark.asyncio
async def test_download_anchor_carries_the_original_not_the_preview():
    """The whole point of the download: full quality, not the shrunk copy.

    Asserted by decoding the anchor's data: URI and comparing it byte-for-byte
    with what was stored -- not by checking that a button merely exists.
    """
    ctx = make_ctx()
    raw = _real_png()
    doc = await _seed(ctx, raw=raw)

    detail = await load_detail(ctx, doc)
    tree = detail_view(
        doc,
        image_src=detail["image_src"],
        fail_reason=detail["fail_reason"],
        raw_original=detail["raw_original"],
        references=detail["references"],
        is_preview=detail["is_preview"],
        # Armed: the anchor only exists after an explicit click, so that
        # opening an image never embeds a ~1M-char payload by itself.
        download_armed=True,
    ).to_dict()

    html_blocks = _of_type(tree, "Html")
    assert html_blocks, "expected a download anchor in the detail view"
    content = html_blocks[0]["props"]["content"]
    assert 'download="' in content, "anchor must have the download attribute"

    payload = content.split("base64,", 1)[1].split('"', 1)[0]
    assert base64.b64decode(payload) == raw, \
        "the download must hand over the ORIGINAL bytes, byte for byte"


@pytest.mark.asyncio
async def test_preview_notice_appears_only_when_it_really_is_a_preview():
    """An honest label: no 'preview size' note when the original is shown.

    A big noisy image gets shrunk for display, so the note belongs there. A
    tiny one is served whole, and claiming it is a preview would be a lie in
    the opposite direction -- both halves are checked.
    """
    ctx = make_ctx()

    big = await _seed(ctx, raw=_real_png(700, 700))
    big_detail = await load_detail(ctx, big)
    assert big_detail["is_preview"] is True, \
        "a large image is displayed shrunk, so it must be flagged as a preview"
    big_text = " ".join(
        n["props"].get("content", "")
        for n in _of_type(detail_view(big, **{
            k: big_detail[k] for k in
            ("image_src", "fail_reason", "raw_original", "references", "is_preview")
        }).to_dict(), "Text")
    )
    assert "preview size" in big_text

    ctx2 = make_ctx()
    small = await _seed(ctx2, raw=_real_png(24, 24))
    small_detail = await load_detail(ctx2, small)
    assert small_detail["is_preview"] is False, \
        "a small image is inlined verbatim -- calling it a preview is wrong"
    small_text = " ".join(
        n["props"].get("content", "")
        for n in _of_type(detail_view(small, **{
            k: small_detail[k] for k in
            ("image_src", "fail_reason", "raw_original", "references", "is_preview")
        }).to_dict(), "Text")
    )
    assert "preview size" not in small_text


@pytest.mark.asyncio
async def test_regenerate_targets_the_tool_priced_for_that_model():
    """Wrong tool = wrong price. The button must match the model used."""
    ctx = make_ctx()
    doc = await _seed(ctx, raw=_real_png(64, 64), model=MODEL_IMAGE_FLASH)

    detail = await load_detail(ctx, doc)
    tree = detail_view(doc, **{
        k: detail[k] for k in
        ("image_src", "fail_reason", "raw_original", "references", "is_preview")
    }).to_dict()

    buttons = [
        n for n in _of_type(tree, "Button")
        if "Regenerate" in (n["props"].get("label") or "")
    ]
    assert buttons, "expected a regenerate button"
    call = buttons[0]["props"]["on_click"]
    assert call["function"] == IMAGE_TOOL_FOR_MODEL[MODEL_IMAGE_FLASH]
    assert call["params"]["prompt"] == "a lighthouse in fog", \
        "regenerate must reuse the SAME prompt"


@pytest.mark.asyncio
async def test_the_reference_that_made_an_image_is_shown_and_reused():
    """'Made from this reference' needs the reference to be recorded AND shown."""
    ctx = make_ctx()
    ref = await _seed(ctx, raw=_real_png(48, 48), prompt="my reference photo")
    doc = await _seed(ctx, raw=_real_png(64, 64), reference_ids=[ref.id])

    detail = await load_detail(ctx, doc)
    assert [r["id"] for r in detail["references"]] == [ref.id]

    tree = detail_view(doc, **{
        k: detail[k] for k in
        ("image_src", "fail_reason", "raw_original", "references", "is_preview")
    }).to_dict()

    texts = " ".join(n["props"].get("content", "") for n in _of_type(tree, "Text"))
    assert "Made from reference" in texts
    assert "my reference photo" in texts

    regen = [
        n for n in _of_type(tree, "Button")
        if "Regenerate" in (n["props"].get("label") or "")
    ][0]
    assert regen["props"]["on_click"]["params"]["reference_generation_ids"] == [ref.id], \
        "regenerating must reuse the same reference, not drop it"


def test_every_mapped_tool_really_exists():
    """Guards the model->tool map against drift.

    A renamed tool would otherwise leave the regenerate button calling a
    function that no longer exists -- a dead button discovered by the user
    rather than by the suite.
    """
    import main  # noqa: F401 -- registers all handler modules
    from app import chat
    from gemini_config import IMAGE_MODEL_CHOICES

    for model, tool in IMAGE_TOOL_FOR_MODEL.items():
        assert tool in chat.functions, f"{tool!r} (for {model}) is not registered"
    assert set(IMAGE_TOOL_FOR_MODEL) == set(IMAGE_MODEL_CHOICES), \
        "every offered model needs a priced tool, and vice versa"


@pytest.mark.asyncio
async def test_opening_does_not_embed_the_original_until_asked():
    """Opening a generation must stay cheap.

    Measured in production, an original inlines to 571k-1005k base64 chars,
    while ~954k was proven NOT to render. Embedding it on every open would
    therefore risk killing the entire panel as a side effect of looking at an
    image, so the heavy payload is attached only after an explicit click.
    """
    ctx = make_ctx(with_key=True)
    raw = _real_png()
    await ctx.storage.upload("gemini/image/big.png", raw, content_type="image/png")
    doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image", "prompt": "big one",
        "model": MODEL_IMAGE_FLASH, "storage_path": "gemini/image/big.png",
        "mime_type": "image/png", "created_at": "2026-07-27T10:00:00Z",
        "source": "generated",
    })
    detail = await load_detail(ctx, doc)

    # Not armed: no data: anchor anywhere, but an affordance to get one.
    closed = detail_view(
        doc, image_src=detail["image_src"], fail_reason=detail["fail_reason"],
        raw_original=detail["raw_original"], references=detail["references"],
        is_preview=detail["is_preview"], download_armed=False,
    ).to_dict()
    html_nodes = [n for n in _walk(closed) if n.get("type") == "Html"]
    assert not html_nodes, "unarmed view must not embed the original"
    labels = [
        n.get("props", {}).get("label", "")
        for n in _walk(closed) if n.get("type") == "Button"
    ]
    assert any("download" in l.lower() for l in labels), \
        f"expected a way to request the original, got {labels}"

    # Armed: the anchor appears, carrying the real bytes.
    armed = detail_view(
        doc, image_src=detail["image_src"], fail_reason=detail["fail_reason"],
        raw_original=detail["raw_original"], references=detail["references"],
        is_preview=detail["is_preview"], download_armed=True,
    ).to_dict()
    html = [n for n in _walk(armed) if n.get("type") == "Html"]
    assert html, "armed view must embed the original"
    assert base64.b64encode(raw).decode() in html[0]["props"]["content"]
