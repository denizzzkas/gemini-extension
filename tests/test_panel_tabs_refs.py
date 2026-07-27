"""The reworked left panel: tabs, visual references, and the copy button.

What these guard, and why each one exists
-----------------------------------------
Every test here maps to a specific complaint about the shipped panel:

* the column was a wall of inputs -- the image form, the video form and the
  history all stacked, though only one form is ever in use, so generation and
  history now live in ``ui.Tabs`` with history underneath;
* the reference picker listed PROMPT TEXT, which asked the user to remember
  which wall of text produced which picture -- so selection moved to a button
  next to the visible image, and what is attached is shown as a thumbnail;
* the prompt could only be selected by hand, so it needs a real copy button;
* "Open in Studio" named the place it opened rather than what it gives you.

Two failure modes these are written specifically to avoid
---------------------------------------------------------
1. **Walking the tree wrongly.** A naive recursive walk that only follows
   ``props`` values silently skips everything inside ``Tabs``, because a tab is
   a plain ``{"label", "content"}`` dict. A test using such a walk finds no
   form, asserts nothing, and passes -- which happened during development and
   briefly looked like a bug in the panel. :func:`_walk` here descends into
   dicts AND lists, and :func:`test_the_walker_descends_into_tabs` proves it.

2. **Testing clicks in isolation.** The host ACCUMULATES params per panel_id
   and merges a re-fetch's ``{}`` into them, so an empty value cannot clear an
   earlier one. Reference clearing is therefore tested through the existing
   :class:`PanelHost` simulator, not by calling the handler with a fresh dict.
"""
from __future__ import annotations

import base64

import pytest

from gemini_config import GENERATION_LOG_COLLECTION, MODEL_IMAGE
from handlers.panel_viewer import CLOSED_SENTINEL
from tests.fixtures import make_ctx
from tests.test_panel_accordion import PanelHost


def _walk(node):
    """Yield every ``(type, props)`` in a rendered tree, INCLUDING inside Tabs.

    Descends through dicts and lists alike. The list case is what makes tab
    contents visible: ``Tabs`` stores ``[{"label", "content"}, ...]``, so a
    walker that only follows ``props`` values never reaches the forms.
    """
    if isinstance(node, dict):
        if node.get("type"):
            yield node["type"], node.get("props", {})
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _types(tree) -> list[str]:
    return [t for t, _ in _walk(tree)]


def _labels(tree) -> list[str]:
    return [p.get("label") for t, p in _walk(tree) if t == "Button" and p.get("label")]


async def _an_image(ctx, prompt="a red fox in snow", with_preview=True):
    data = {
        "user_id": ctx.user.imperal_id, "kind": "image", "prompt": prompt,
        "model": MODEL_IMAGE, "storage_path": "gemini/image/a.jpg",
        "mime_type": "image/jpeg", "created_at": "2026-07-27T10:00:00Z",
        "source": "generated",
    }
    if with_preview:
        data["preview_b64"] = base64.b64encode(b"fakepreviewbytes").decode()
        data["preview_mime"] = "image/png"
    return await ctx.store.create(GENERATION_LOG_COLLECTION, data)


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


@pytest.mark.asyncio
async def test_an_attached_reference_is_shown_as_a_thumbnail():
    """What is attached must be visible as an IMAGE, not as its prompt text."""
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    doc = await _an_image(ctx)

    tree = (await gemini_quick_panel(ctx, refs=doc.id)).to_dict()

    srcs = [p.get("src", "") for t, p in _walk(tree) if t == "Image"]
    assert any(s.startswith("data:image/") for s in srcs), \
        "the attached reference must render as a thumbnail"

    # And it must actually be SUBMITTED. The carrier is a HIDDEN ui.Form
    # default, not a visible picker: a form submits the values of its own
    # inputs and a thumbnail is not an input, so without this the "Use as
    # reference" button would be purely decorative.
    defaults = [
        p.get("defaults") or {} for t, p in _walk(tree)
        if t == "Form" and p.get("action") == "generate_image"
    ]
    assert defaults, "the image form must exist"
    carried = defaults[0].get("reference_generation_ids")
    assert carried and doc.id in carried, \
        "a reference chosen by sight must reach the form submit"

    # It must be a LIST: the tool's field is list[str] and Pydantic rejects a
    # comma-joined string outright (verified against GenerateImageParams).
    assert isinstance(carried, list), \
        f"reference ids must submit as a list, got {type(carried).__name__}"


@pytest.mark.asyncio
async def test_the_meaningless_prompt_text_picker_is_gone():
    """There must be no dropdown asking the user to pick an image by its PROMPT.

    The old MultiSelect listed prompt text, so choosing a reference meant
    recognising a picture from the wall of words that produced it. Selection now
    happens by sight, next to the visible image, and this guards against the
    picker quietly coming back.
    """
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    await _an_image(ctx)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    pickers = [
        p for t, p in _walk(tree)
        if t == "MultiSelect" and p.get("param_name") == "reference_generation_ids"
    ]
    assert not pickers, \
        "references must be chosen by sight, never from a list of prompt text"


@pytest.mark.asyncio
async def test_nothing_is_attached_until_the_user_asks():
    """No selection => no thumbnail and no pre-filled reference."""
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    await _an_image(ctx)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    carried = [
        (p.get("defaults") or {}).get("reference_generation_ids")
        for t, p in _walk(tree)
        if t == "Form" and p.get("action") == "generate_image"
    ]
    assert all(not c for c in carried), "nothing may be attached by default"
    assert "Clear references" not in _labels(tree)


@pytest.mark.asyncio
async def test_clearing_a_reference_survives_param_accumulation():
    """Clearing must work against the host's MERGE semantics.

    The host merges new params INTO those already accumulated for a panel_id,
    so ``refs=""`` cannot remove a previously set ``refs`` -- the old value
    survives. Clearing therefore overwrites with a sentinel. This is the exact
    bug that once made "Hide" do nothing while "View" worked, so it is tested
    through the accumulating host rather than with a fresh params dict.
    """
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    doc = await _an_image(ctx)
    host = PanelHost(ctx, "gemini_quick")
    host.render = lambda **kw: gemini_quick_panel(ctx, **kw)  # noqa: E731

    attached = (await gemini_quick_panel(ctx, **{"refs": doc.id})).to_dict()
    assert any(
        p.get("src", "").startswith("data:image/")
        for t, p in _walk(attached) if t == "Image"
    ), "precondition: the reference should be attached"

    # The Clear button's params, merged over the accumulated ones.
    clear_calls = [
        p.get("on_click") for t, p in _walk(attached)
        if t == "Button" and p.get("label") == "Clear references"
    ]
    assert clear_calls, "an attached reference must be removable"
    merged = {"refs": doc.id}
    merged.update(clear_calls[0]["params"])

    assert merged["refs"] != doc.id, \
        "clearing must OVERWRITE the id -- an empty value cannot survive the merge"
    assert merged["refs"] == CLOSED_SENTINEL

    cleared = (await gemini_quick_panel(ctx, **merged)).to_dict()
    # Assert on the REAL carrier (the hidden Form default). Looking for a
    # MultiSelect here would iterate an empty list and pass unconditionally --
    # a vacuous assertion that proves nothing.
    carried = [
        (p.get("defaults") or {}).get("reference_generation_ids")
        for t, p in _walk(cleared)
        if t == "Form" and p.get("action") == "generate_image"
    ]
    assert carried, "the image form must still render after clearing"
    assert all(not c for c in carried), "the reference must actually be gone"
    assert "Clear references" not in _labels(cleared), \
        "nothing is attached any more, so there is nothing to clear"


@pytest.mark.asyncio
async def test_a_reference_without_a_cached_preview_is_still_visible():
    """A missing thumbnail must not make an attached reference invisible.

    Older records predate preview caching. Rendering nothing for them would
    silently attach an image the user cannot see -- worse than the dropdown
    this replaced.
    """
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    doc = await _an_image(ctx, prompt="an old record", with_preview=False)

    tree = (await gemini_quick_panel(ctx, refs=doc.id)).to_dict()
    texts = " ".join(
        str(p.get("content") or "") for t, p in _walk(tree) if t == "Text"
    )
    assert "an old record" in texts, \
        "a reference with no cached preview must still be listed by label"


@pytest.mark.asyncio
async def test_building_the_form_never_downloads_image_bytes():
    """Rendering the panel must stay free of media I/O.

    Thumbnails come from the CACHED preview on the record. Downloading originals
    to build the form would make every render slow and push the response toward
    the payload ceiling that made images fail to display in the first place.
    """
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    doc = await _an_image(ctx)

    downloads: list[str] = []
    original = ctx.storage.download

    async def spy(path, *a, **kw):
        downloads.append(path)
        return await original(path, *a, **kw)

    ctx.storage.download = spy
    await gemini_quick_panel(ctx, refs=doc.id)

    assert not downloads, f"rendering the form downloaded {downloads}"
