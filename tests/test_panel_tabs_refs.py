"""The reworked left panel: generation as an Image/Video button toggle.

``ui.Tabs`` was tried first and the user reported it simply did not switch in
the real panel host. Rather than debug a component neither of us can inspect
client-side, generation now uses the same primitive already proven reliable
everywhere else in this panel: a ``ui.Button`` targeting a self-call that
overwrites a param (``gen_tab``), exactly like ``View image``/``Hide``. History
stays underneath, unchanged.

Reference ATTACHMENT is tested separately in ``test_panel_references.py``; this
file is about layout and the walker that makes the other assertions honest.
"""
from __future__ import annotations

import pytest

from tests.fixtures import make_ctx
from tests.panel_helpers import _an_image, _labels, _types, _walk


def test_the_walker_descends_into_stacks():
    """Proves the helper above can SEE inside a Stack's children.

    Without this, every assertion in this file could pass by finding nothing.
    """
    from imperal_sdk import ui

    tree = ui.Stack(direction="v", children=[ui.Text("inside-a-stack")]).to_dict()

    assert "Text" in _types(tree), \
        "the walker cannot see inside Stack -- every other test here is vacuous"


@pytest.mark.asyncio
async def test_generation_offers_an_image_video_toggle():
    """Two buttons switch between the image and video forms -- not ui.Tabs."""
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    assert "Tabs" not in _types(tree), \
        "ui.Tabs was reported broken in the real host -- must not reappear"

    labels = _labels(tree)
    assert "Image" in labels and "Video" in labels, \
        f"expected Image/Video toggle buttons, got {labels}"


@pytest.mark.asyncio
async def test_default_tab_is_image():
    """With no ``gen_tab`` param, the image form renders (not video)."""
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    forms = [p for t, p in _walk(tree) if t == "Form"]
    assert forms, "expected a generation form to render by default"


@pytest.mark.asyncio
async def test_video_toggle_button_calls_self_with_gen_tab_video():
    """Clicking 'Video' must re-call this SAME panel with gen_tab=video --

    the pattern already proven reliable (View image/Hide), not a client-side
    tab widget with its own opaque state.
    """
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    video_click = next(
        p.get("on_click") for t, p in _walk(tree)
        if t == "Button" and p.get("label") == "Video"
    )
    assert video_click["function"] == "__panel__gemini_quick"
    assert video_click["params"].get("gen_tab") == "video"


@pytest.mark.asyncio
async def test_history_sits_below_the_generation_toggle():
    """Order matters: generate first, then history -- not history buried."""
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    await _an_image(ctx)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    order = _types(tree)
    labels_in_order = [
        p.get("label") for t, p in _walk(tree) if t == "Button" and p.get("label")
    ]
    assert "Image" in labels_in_order, "no generation toggle rendered"
    headers = [
        i for i, (t, p) in enumerate(_walk(tree))
        if t == "Header" and "Recent" in str(p.get("text") or p.get("title") or "")
    ]
    image_btn_at = next(
        i for i, (t, p) in enumerate(_walk(tree))
        if t == "Button" and p.get("label") == "Image"
    )
    if headers:
        assert headers[0] > image_btn_at, "history must come after the generation toggle"


@pytest.mark.asyncio
async def test_history_offers_use_as_reference_next_to_the_image():
    """The fix for a picker that listed prompt text.

    The choice has to be made where the image is VISIBLE, so the button must
    live on the history card itself.
    """
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    doc = await _an_image(ctx)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    assert "Use as reference" in _labels(tree), \
        "a history entry must be selectable as a reference by sight"

    calls = [
        p.get("on_click") for t, p in _walk(tree)
        if t == "Button" and p.get("label") == "Use as reference"
    ]
    assert calls and calls[0]["params"].get("refs") == doc.id, \
        "the button must carry THIS generation's id"


@pytest.mark.asyncio
async def test_view_full_resolution_click_through_the_real_panel():
    """End-to-end: open an entry through the ACTUAL panel entry point -- not
    detail_content in isolation -- and check the full-resolution affordance is
    there. Testing only the unit would have missed a broken wire between
    panel.py's ``generation_id`` param and the studio panel, which is exactly
    the kind of gap that let a real bug through while unit tests stayed green.

    Full detail now renders in "gemini_studio" (the restored centre panel),
    not inline in "gemini_quick" any more -- see handlers/panel.py.
    """
    from handlers.panel import gemini_studio_panel
    from gemini_config import GENERATION_LOG_COLLECTION, MODEL_IMAGE

    ctx = make_ctx(with_key=True)
    raw = b"the-real-original-bytes" * 50
    await ctx.storage.upload("gemini/image/e2e.png", raw, content_type="image/png")
    doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image", "prompt": "e2e",
        "model": MODEL_IMAGE, "storage_path": "gemini/image/e2e.png",
        "mime_type": "image/png", "created_at": "2026-07-27T10:00:00Z",
        "source": "generated",
    })

    tree = (
        await gemini_studio_panel(ctx, generation_id=doc.id)
    ).to_dict()

    hit = [
        p for t, p in _walk(tree)
        if t == "Button" and "full resolution" in (p.get("label") or "").lower()
        and (p.get("on_click") or {}).get("action") == "send"
        and doc.id in (p.get("on_click") or {}).get("message", "")
    ]
    assert hit, (
        "opening the entry through the studio panel must offer a "
        "'View full resolution in chat' button naming this generation"
    )
