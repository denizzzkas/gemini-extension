"""Tests for the Gemini panel handler (gemini_quick, the only panel this
extension declares -- gemini_studio was removed, see handlers/panel.py)."""
from __future__ import annotations

import pytest

from handlers.panel import gemini_quick_panel, gemini_studio_panel
from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx


def _find_types(node: dict, acc: list[str]) -> None:
    """Walk a serialized UINode tree, collecting all 'type' fields."""
    if isinstance(node, dict):
        if "type" in node and isinstance(node["type"], str):
            acc.append(node["type"])
        for v in node.values():
            _find_types(v, acc)
    elif isinstance(node, list):
        for item in node:
            _find_types(item, acc)


def _find_image_src(node):
    """Return the src of the first Image node in a serialized tree, or None."""
    if isinstance(node, dict):
        if node.get("type") == "Image":
            return node.get("props", {}).get("src")
        for v in node.values():
            found = _find_image_src(v)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_image_src(item)
            if found:
                return found
    return None


def _count_type(node, target: str) -> int:
    """Count nodes of a given type in a serialized tree."""
    n = 0
    if isinstance(node, dict):
        if node.get("type") == target:
            n += 1
        for v in node.values():
            n += _count_type(v, target)
    elif isinstance(node, list):
        for item in node:
            n += _count_type(item, target)
    return n


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
async def test_panel_renders_history_with_key_and_generations():
    ctx = make_ctx(with_key=True)
    await ctx.storage.upload("gemini/image/abc.png", b"abc-png-bytes", content_type="image/png")
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image",
        "prompt": "a cat astronaut", "model": "gemini-3-pro-image",
        "url": "https://panel.imperal.io/storage/default/gemini/abc.png",
        "storage_path": "gemini/image/abc.png",
        "mime_type": "image/png", "created_at": "2026-07-18T00:00:00Z",
    })

    node = await gemini_quick_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Card" in types
    # The image is reachable via an on-demand "View image" button, NOT inlined
    # into the list payload.
    assert "Button" in types


@pytest.mark.asyncio
async def test_left_panel_auto_opens_studio_after_discovery():
    """The host reads auto_action only from the left panel's root node."""
    ctx = make_ctx(with_key=True)

    tree = (await gemini_quick_panel(ctx)).to_dict()

    action = tree["props"].get("auto_action")
    assert action == {
        "action": "call",
        "function": "__panel__gemini_studio",
        "params": {},
    }


@pytest.mark.asyncio
async def test_studio_default_renders_history_without_needing_left_panel():
    """Studio's entry state must be usable even if the left sidebar is hidden."""
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "a studio landing image",
        "model": "gemini-3-pro-image",
        "storage_path": "gemini/image/studio.png",
        "mime_type": "image/png",
        "created_at": "2026-07-29T00:00:00Z",
    })

    tree = (await gemini_studio_panel(ctx)).to_dict()

    assert tree["type"] == "Page"
    assert _count_type(tree, "Card") == 1
    assert "Nothing open yet" not in str(tree)


@pytest.mark.asyncio
async def test_panel_empty_history():
    ctx = make_ctx(with_key=True)

    node = await gemini_quick_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Empty" in types


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
async def test_history_list_does_zero_storage_reads():
    # THE regression test for "panel loads forever", which recurred twice.
    # Fetching media while rendering the list is the cause -- even 4 downloads
    # bounded by a 3 MB budget reproduced the hang. So the list must perform
    # NO storage reads at all: any download during render fails this test.
    ctx = make_ctx(with_key=True)
    total = 12
    for i in range(total):
        await ctx.storage.upload(f"gemini/image/img{i}.png", b"x" * 64, content_type="image/png")
        await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": "image",
            "prompt": f"generation {i}",
            "model": "gemini-3-pro-image",
            "url": f"https://panel.imperal.io/storage/default/gemini/img{i}.png",
            "storage_path": f"gemini/image/img{i}.png",
            "mime_type": "image/png",
            "created_at": "2026-07-22T00:00:00+00:00",
        })

    # BaseException (not Exception) on purpose: _image_data_uri catches
    # Exception, which would silently swallow a plain AssertionError and make
    # this guard pass even while the list downloads. This must escape.
    class _ForbiddenDownload(BaseException):
        pass

    async def _forbidden_download(path):
        raise _ForbiddenDownload(
            f"history render must not download media (attempted {path!r})"
        )

    ctx.storage.download = _forbidden_download

    try:
        node = await gemini_quick_panel(ctx)
    except _ForbiddenDownload as e:
        raise AssertionError(str(e)) from None
    tree = node.to_dict()

    # Nothing inlined -> no Image nodes and no data: URIs in the payload.
    assert _count_type(tree, "Image") == 0
    assert _find_image_src(tree) is None
    # Every generation is still listed, each with its own on-demand button.
    assert _count_type(tree, "Card") >= total


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
