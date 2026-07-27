"""The reworked left panel: generation in TABS, history underneath.

The column used to stack the image form, the video form and the history all at
once, though only one form is ever in use -- the video form pushed history off
screen. Generation now lives in ``ui.Tabs`` (Image / Video) with history below.

Reference ATTACHMENT is tested separately in ``test_panel_references.py``; this
file is about layout and the walker that makes the other assertions honest.
"""
from __future__ import annotations

import pytest

from tests.fixtures import make_ctx
from tests.panel_helpers import _an_image, _labels, _types, _walk


def test_the_walker_descends_into_tabs():
    """Proves the helper above can SEE tab contents.

    Without this, every assertion in this file could pass by finding nothing.
    """
    from imperal_sdk import ui

    tree = ui.Tabs(tabs=[
        {"label": "Image", "content": ui.Text("inside-a-tab")},
    ]).to_dict()

    assert "Text" in _types(tree), \
        "the walker cannot see inside Tabs -- every other test here is vacuous"


@pytest.mark.asyncio
async def test_generation_is_split_into_image_and_video_tabs():
    """The two forms must be TABS, not two stacked forms in one column."""
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    tabs = [p for t, p in _walk(tree) if t == "Tabs"]
    assert tabs, "the panel must offer tabs for generation"
    labels = [tab.get("label") for tab in tabs[0].get("tabs", [])]
    assert labels == ["Image", "Video"], f"expected Image/Video tabs, got {labels}"

    # Exactly one form per tab: two visible at once is the layout being fixed.
    for tab in tabs[0]["tabs"]:
        forms = [t for t, _ in _walk(tab["content"]) if t == "Form"]
        assert len(forms) == 1, f"tab {tab['label']} has {len(forms)} forms"


@pytest.mark.asyncio
async def test_history_sits_below_the_generation_tabs():
    """Order matters: generate first, then history -- not history buried."""
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    await _an_image(ctx)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    order = _types(tree)
    assert "Tabs" in order, "no generation tabs rendered"
    headers = [
        i for i, (t, p) in enumerate(_walk(tree))
        if t == "Header" and "Recent" in str(p.get("text") or p.get("title") or "")
    ]
    tabs_at = order.index("Tabs")
    if headers:
        assert headers[0] > tabs_at, "history must come after the generation tabs"


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
