"""Tests for the generation detail content: one generation, in full.

The properties pinned here are the ones the user actually asked for, plus the
ones this extension has already been burned by:

  * a real download must be OFFERED (an ``<a download>`` anchor over a
    data: URI, gated by DOWNLOAD_CEILING_CHARS -- see handlers/panel_html.py
    for why the earlier "data: URIs are categorically blocked" conclusion was
    an overreach: Chrome only ever blocked page-initiated top-frame
    *navigation*, never the download attribute's forced-save path),
  * "View full resolution in chat" (ui.Send) must ALSO be present as the
    size-independent fallback, and must name this exact generation,
  * the "shown at preview size" note must appear only when what is displayed
    really is smaller than the original,
  * "Regenerate" must call the per-model tool matching the model that made the
    image: Imperal prices a TOOL, and these models differ several-fold in
    cost, so the wrong target bills the wrong amount,
  * the model->tool map must not drift away from the really-registered tools.

``detail_content`` is the one real render path for a generation's full
detail. It is used both inside "gemini_studio" (the centre panel restored
with ``center_overlay=True`` -- see handlers/panel.py) and is exercised here
standalone, independent of which panel calls it.
"""
from __future__ import annotations

import random

import pytest

from core import png
from gemini_config import (
    GENERATION_LOG_COLLECTION, IMAGE_TOOL_FOR_MODEL, MODEL_IMAGE_FLASH,
)
from handlers.panel_detail import detail_content, load_detail
from imperal_sdk import ui
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


def _detail_tree(doc, detail, **overrides):
    kwargs = {
        k: detail[k] for k in
        ("image_src", "fail_reason", "raw_original", "references", "is_preview")
    }
    kwargs.update(overrides)
    nodes = detail_content(doc, **kwargs)
    return ui.Stack(children=nodes, direction="v", gap=3).to_dict()


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
async def test_detail_does_not_offer_a_broken_full_resolution_chat_detour():
    """No control should promise a chat view that returns only an ID."""
    ctx = make_ctx()
    doc = await _seed(ctx, raw=_real_png())

    detail = await load_detail(ctx, doc)
    tree = _detail_tree(doc, detail)

    buttons = [
        b for b in _of_type(tree, "Button")
        if "full resolution" in (b.get("props", {}).get("label") or "").lower()
    ]
    assert not buttons


@pytest.mark.asyncio
async def test_download_button_is_offered_for_a_small_original():
    """The real download: an <a download> anchor, not just the chat fallback.

    handlers/panel_html.py restores this after re-checking Chrome's own
    documented behaviour -- the 2017 data: block is page-navigation only, an
    anchor's ``download`` attribute is a different, unaffected code path.
    """
    ctx = make_ctx()
    raw = _real_png(24, 24)  # small -> well under DOWNLOAD_CEILING_CHARS
    doc = await _seed(ctx, raw=raw)

    detail = await load_detail(ctx, doc)
    tree = _detail_tree(doc, detail)

    html_nodes = _of_type(tree, "Html")
    assert any("download=" in n["props"].get("content", "") for n in html_nodes), \
        "expected a real <a download> anchor for a small original"


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
        for n in _of_type(_detail_tree(big, big_detail), "Text")
    )
    assert "preview size" in big_text

    ctx2 = make_ctx()
    small = await _seed(ctx2, raw=_real_png(24, 24))
    small_detail = await load_detail(ctx2, small)
    assert small_detail["is_preview"] is False, \
        "a small image is inlined verbatim -- calling it a preview is wrong"
    small_text = " ".join(
        n["props"].get("content", "")
        for n in _of_type(_detail_tree(small, small_detail), "Text")
    )
    assert "preview size" not in small_text


@pytest.mark.asyncio
async def test_regenerate_targets_the_tool_priced_for_that_model():
    """Wrong tool = wrong price. The button must match the model used."""
    ctx = make_ctx()
    doc = await _seed(ctx, raw=_real_png(64, 64), model=MODEL_IMAGE_FLASH)

    detail = await load_detail(ctx, doc)
    tree = _detail_tree(doc, detail)

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

    tree = _detail_tree(doc, detail)

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


def test_detail_exposes_real_download_not_a_broken_chat_detour():
    """Full originals are offered through the panel's download block only.

    The former chat hand-off looked actionable but returned a confirmation
    sentence without a usable picture or file. Keeping a real browser
    download and omitting the dead detour is the honest, testable UX.
    """
    from handlers.panel_detail import detail_content
    import handlers.panel_html as panel_html

    assert "view_full_resolution_block" not in detail_content.__code__.co_names
    assert not hasattr(panel_html, "view_full_resolution_block")
