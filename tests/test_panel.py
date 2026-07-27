"""Tests for the Gemini Studio panel handler."""
from __future__ import annotations

import pytest

from handlers.panel import gemini_studio_panel, gemini_quick_panel
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
async def test_panel_empty_history():
    ctx = make_ctx(with_key=True)

    node = await gemini_quick_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Empty" in types


@pytest.mark.asyncio
async def test_panel_image_form_has_model_select_with_all_choices():
    from gemini_config import IMAGE_MODEL_CHOICES, MODEL_IMAGE

    ctx = make_ctx(with_key=True)
    node = await gemini_quick_panel(ctx)
    tree = node.to_dict()

    def _find_select(n):
        if isinstance(n, dict):
            if n.get("type") == "Select":
                return n.get("props", {})
            for v in n.values():
                found = _find_select(v)
                if found:
                    return found
        elif isinstance(n, list):
            for item in n:
                found = _find_select(item)
                if found:
                    return found
        return None

    select_props = _find_select(tree)
    assert select_props is not None
    assert select_props["value"] == MODEL_IMAGE
    option_values = {opt["value"] for opt in select_props["options"]}
    assert option_values == set(IMAGE_MODEL_CHOICES)


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
