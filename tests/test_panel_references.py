"""Attaching a reference image BY SIGHT, not by its prompt text.

The old picker was a ``MultiSelect`` listing prompt text, so choosing a
reference meant recognising a picture from the wall of words that produced it.
Selection now happens next to the visible image ("Use as reference"), what is
attached is shown as a THUMBNAIL, and the ids ride to the tool in a hidden
``ui.Form(defaults=...)`` -- a form submits the values of its own inputs, and a
thumbnail is not an input, so without that carrier the button would be purely
decorative.

Two failure modes these tests are written to avoid
--------------------------------------------------
1. **Vacuous assertions.** After the picker was deleted, a check that iterated
   "every MultiSelect named reference_generation_ids" passed unconditionally by
   iterating nothing. The assertions here target the carrier that actually
   exists, and ``test_the_meaningless_prompt_text_picker_is_gone`` pins the
   removal directly.

2. **Testing clicks in isolation.** The host ACCUMULATES params per panel_id and
   merges a re-fetch's ``{}`` into them, so an empty value cannot clear an
   earlier one -- the exact bug that once made "Hide" do nothing while "View"
   worked. Clearing is therefore driven through the ``PanelHost`` simulator.
"""
from __future__ import annotations

import pytest

from handlers.panel_viewer import CLOSED_SENTINEL
from tests.fixtures import make_ctx
from tests.panel_helpers import _an_image, _labels, _walk
from tests.test_panel_accordion import PanelHost


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
