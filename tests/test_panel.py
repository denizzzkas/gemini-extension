"""Tests for gemini_quick (the left-sidebar panel): generation controls,
the per-user API-key field, and the entry point into gemini_studio.

gemini_studio's OWN tests (its default landing view + the history list it
renders) live in ``tests/test_panel_studio.py`` -- split out so neither file
grows past the 300-line limit the deploy validator warns about. Both share
the tree-walking helpers in ``tests/panel_helpers.py``.
"""
from __future__ import annotations

import pytest

from handlers.panel import gemini_quick_panel
from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx
from tests.panel_helpers import _count_type, _find_image_src, _find_types


@pytest.mark.asyncio
async def test_panel_renders_without_key():
    ctx = make_ctx(with_key=False)

    node = await gemini_quick_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Alert" in types
    assert "Form" in types
    # The LEFT panel is a column, not a Page: a "permanent" slot renders as a
    # sidebar column, and only the centre surface is a Page. Asserting Page
    # here would be asserting the OLD layout, in which the centre duplicated
    # the generation form.
    assert tree["type"] == "Stack"


@pytest.mark.asyncio
async def test_panel_has_api_key_field_above_generation_form_when_unconfigured():
    """gemini_api_key is per-user now (I-KEY-PER-USER) -- the left panel must
    carry its own inline key field, positioned ABOVE the generation form, not
    just a Navigate-to-Secrets alert."""
    ctx = make_ctx(with_key=False)

    tree = (await gemini_quick_panel(ctx)).to_dict()

    children = tree["props"]["children"]
    key_card_idx = next(
        i for i, c in enumerate(children)
        if isinstance(c, dict) and c.get("type") == "Card"
        and "API key" in str(c.get("props", {}).get("title", ""))
    )
    # The key Card must come before the generation tabs/history section.
    # ui.Password serializes as an Input with type="password", not a
    # distinct "Password" node type.
    key_card = children[key_card_idx]
    all_types = []
    _find_types(key_card, all_types)
    assert "Input" in all_types
    input_types = []
    def _collect_input_type_props(node):
        if isinstance(node, dict):
            if node.get("type") == "Input":
                input_types.append(node.get("props", {}).get("type"))
            for v in node.values():
                _collect_input_type_props(v)
        elif isinstance(node, list):
            for item in node:
                _collect_input_type_props(item)
    _collect_input_type_props(key_card)
    assert "password" in input_types

    # Confirm ordering: key Card index is earlier than the child that
    # contains the generation-tabs Form -- gemini_quick renders no
    # history/Header at all any more (that moved to gemini_studio), so the
    # Form is the next real landmark below the key card.
    def _contains_form(node) -> bool:
        t = []
        _find_types(node, t)
        return "Form" in t

    form_idx = next(
        i for i, c in enumerate(children)
        if i != key_card_idx and _contains_form(c)
    )
    assert key_card_idx < form_idx


@pytest.mark.asyncio
async def test_panel_has_api_key_field_when_already_configured():
    """The field stays present (to allow replacing the key) even once a key
    is already set -- it must not disappear once "Connected"."""
    ctx = make_ctx(with_key=True)

    tree = (await gemini_quick_panel(ctx)).to_dict()

    children = tree["props"]["children"]
    key_card_idx = next(
        i for i, c in enumerate(children)
        if isinstance(c, dict) and c.get("type") == "Card"
        and "API key" in str(c.get("props", {}).get("title", ""))
    )
    types = []
    _find_types(children[key_card_idx], types)
    assert "Input" in types


@pytest.mark.asyncio
async def test_quick_panel_has_a_reachable_entry_point_into_studio():
    """Regression: once history-cards moved OUT of gemini_quick, the "Image
    info"/"View image" buttons that used to open gemini_studio moved with
    them -- leaving gemini_studio declared in the manifest but unreachable,
    which for the user is indistinguishable from "there is no centre panel
    and no history at all". gemini_quick must always render its OWN
    explicit call into gemini_studio."""
    ctx = make_ctx(with_key=True)

    tree = (await gemini_quick_panel(ctx)).to_dict()

    def _buttons(n):
        if isinstance(n, dict):
            if n.get("type") == "Button":
                yield n.get("props", {})
            for v in n.values():
                yield from _buttons(v)
        elif isinstance(n, list):
            for item in n:
                yield from _buttons(item)

    targets = [
        b["on_click"]["function"] for b in _buttons(tree)
        if b.get("on_click", {}).get("function") == "__panel__gemini_studio"
    ]
    assert targets, "gemini_quick has no button that opens gemini_studio -- the centre panel is unreachable"


@pytest.mark.asyncio
async def test_left_panel_has_no_unwanted_startup_dispatch():
    """The permanent sidebar must render independently of Studio routing."""
    ctx = make_ctx(with_key=True)

    tree = (await gemini_quick_panel(ctx)).to_dict()

    assert "auto_action" not in tree["props"]


@pytest.mark.asyncio
async def test_panel_image_form_has_model_toggle_with_all_choices():
    """Model is chosen via a button row (like Image/Video), not a ui.Select.

    A Select's value never actually changed which tool the Form posted to --
    action was a fixed string chosen at render time -- so every submission hit
    the same tool regardless of which model was "selected". The picker is now
    buttons that re-render the panel with the model baked into the Form's
    action, so what's highlighted is really what runs.
    """
    from gemini_config import IMAGE_MODEL_CHOICES, IMAGE_TOOL_FOR_MODEL, MODEL_IMAGE

    ctx = make_ctx(with_key=True)
    node = await gemini_quick_panel(ctx)
    tree = node.to_dict()

    def _buttons(n):
        if isinstance(n, dict):
            if n.get("type") == "Button":
                yield n.get("props", {})
            for v in n.values():
                yield from _buttons(v)
        elif isinstance(n, list):
            for item in n:
                yield from _buttons(item)

    labels = {b.get("label") for b in _buttons(tree)}
    assert set(info["label"] for info in IMAGE_MODEL_CHOICES.values()) <= labels

    def _find_forms(n):
        if isinstance(n, dict):
            if n.get("type") == "Form":
                yield n.get("props", {})
            for v in n.values():
                yield from _find_forms(v)
        elif isinstance(n, list):
            for item in n:
                yield from _find_forms(item)

    actions = {f.get("action") for f in _find_forms(tree)}
    # Default (no model picked yet) submits to the Pro tool.
    assert IMAGE_TOOL_FOR_MODEL[MODEL_IMAGE] in actions


@pytest.mark.asyncio
async def test_panel_renders_no_image_when_bytes_are_gone():
    # A legacy record whose bytes are no longer in storage must NOT fall back
    # to its stored url: that url 404s in a browser, so it would render as a
    # broken-image icon (the original "image unavailable" complaint). Honest
    # behaviour is a text-only card and no <Image> node at all.
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "an old pre-fix generation",
        "model": "gemini-3-pro-image",
        "url": "/storage/default/gemeni/legacy123.jpg",
        "storage_path": "gemini/image/legacy123.jpg",  # bytes never uploaded
        "mime_type": "image/jpeg",
        "created_at": "2026-07-19T00:00:00+00:00",
    })

    node = await gemini_quick_panel(ctx)
    tree = node.to_dict()

    assert _find_image_src(tree) is None  # no fake/broken link
    assert _count_type(tree, "Card") >= 1  # still listed


# The production validator enforces an OLDER ui.FileUpload signature than the
# locally-installed SDK: title, hint and show_previews exist here but are
# rejected there, and using them made the whole extension undeployable
# (deploy 84b3f132, rolled back). Local pytest could not see it, because
# locally the kwargs are perfectly valid -- so this pins the intersection.
#
# Re-verified against SDK 5.9.13: the gap is NOT closed by upgrading. That
# release still exposes title/hint/show_previews (and adds variant), while the
# production validator accepts none of them -- so this test stays load-bearing
# rather than being a leftover from one bad deploy.
_PROD_SAFE_FILEUPLOAD_KWARGS = {
    "accept", "blocked_extensions", "max_files", "max_size_mb",
    "max_total_mb", "multiple", "on_upload", "param_name",
}


@pytest.mark.asyncio
async def test_fileupload_uses_only_kwargs_production_accepts():
    """A newer local SDK must not tempt us into an undeployable panel."""
    from handlers.panel_forms import _reference_controls

    for node in _reference_controls([{"value": "a", "label": "a"}]):
        d = node.to_dict()
        if d.get("type") != "FileUpload":
            continue
        extra = set(d.get("props", {})) - _PROD_SAFE_FILEUPLOAD_KWARGS
        assert not extra, (
            f"ui.FileUpload uses {sorted(extra)}, which the production "
            "validator rejects -- the deploy will be refused even though the "
            "local SDK accepts them"
        )
