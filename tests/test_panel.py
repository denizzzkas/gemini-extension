"""Tests for the Gemini Studio panel handler."""
from __future__ import annotations

import pytest

from handlers.panel import gemini_studio_panel, _quick_stats_panel
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

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Alert" in types
    assert "Form" in types
    assert tree["type"] == "Page"


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

    node = await gemini_studio_panel(ctx)
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

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    types = []
    _find_types(tree, types)
    assert "Empty" in types


@pytest.mark.asyncio
async def test_quick_stats_open_button_uses_panel_call_action():
    # Regression test: the "Open Gemini Studio" button must use
    # ui.Call("__panel__gemini_studio") -- panels are fetched via the /call
    # endpoint as __panel__{panel_id} (see ext.panel()'s docstring), there
    # is no frontend route for a raw /ext/<app>/<panel_id> URL path. An
    # earlier version of this button used ui.Navigate(path=...) instead,
    # which 404'd in the panel host -- this is the actual root cause of
    # the reported "Open Gemini AI opens a 404" bug.
    ctx = make_ctx(with_key=True)

    result = await _quick_stats_panel(ctx)
    tree = result["ui"]

    def _find_button_on_click(node):
        if isinstance(node, dict):
            if node.get("type") == "Button":
                return node.get("props", {}).get("on_click", {})
            for v in node.values():
                found = _find_button_on_click(v)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _find_button_on_click(item)
                if found:
                    return found
        return None

    on_click = _find_button_on_click(tree)
    assert on_click is not None
    assert on_click.get("action") == "call"
    assert on_click.get("function") == "__panel__gemini_studio"


@pytest.mark.asyncio
async def test_panel_image_form_has_model_select_with_all_choices():
    from gemini_config import IMAGE_MODEL_CHOICES, MODEL_IMAGE

    ctx = make_ctx(with_key=True)
    node = await gemini_studio_panel(ctx)
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
async def test_image_viewer_panel_returns_bytes_as_data_uri():
    # Verified against production: the stored /storage/<tenant>/<ext>/<file>
    # path is NOT publicly served -- GET returns HTTP 404 with the panel's
    # HTML shell, even seconds after creation. So a url-based <Image> can only
    # ever render broken; the bytes must be shipped as a data: URI. That now
    # happens in the dedicated one-image viewer, not in the history list.
    import base64
    from handlers.panel import _image_viewer_panel

    ctx = make_ctx(with_key=True)
    png = b"fake-png-bytes-for-panel-test"
    await ctx.storage.upload("gemini/image/fresh123.png", png, content_type="image/png")
    doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "a normal generation",
        "model": "gemini-3-pro-image",
        "url": "https://panel.imperal.io/storage/default/gemini/fresh123.png",
        "storage_path": "gemini/image/fresh123.png",
        "mime_type": "image/png",
        "created_at": "2026-07-22T00:00:00+00:00",
    })

    result = await _image_viewer_panel(ctx, generation_id=doc.id)

    src = _find_image_src(result["ui"])
    assert src is not None
    assert src.startswith("data:image/png;base64,")
    assert base64.b64decode(src.split(",", 1)[1]) == png


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
        node = await gemini_studio_panel(ctx)
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

    node = await gemini_studio_panel(ctx)
    tree = node.to_dict()

    assert _find_image_src(tree) is None  # no fake/broken link
    assert _count_type(tree, "Card") >= 1  # still listed


# ─── viewer lookup: production-realistic store semantics ──────────────────── #
#
# These exist because the real bug shipped GREEN through the suite above.
# MockStore.get() is a plain dict lookup that ignores user_id, but the real
# StoreClient.get() does NOT send user_id to the gateway at all (only
# extension/tenant), while StoreClient.query() DOES. So a mock can never
# reproduce "the list shows it, the viewer says Not found". The stores below
# deliberately model that asymmetry.

class _ScopedStore:
    """Store whose get() is unscoped (404s like the gateway) but query() is scoped."""

    def __init__(self, rows: dict, *, get_returns_none=True, query_raises=False):
        self._rows = rows
        self._get_returns_none = get_returns_none
        self._query_raises = query_raises
        self.get_calls = 0
        self.query_calls = 0

    async def get(self, collection, doc_id=None):
        self.get_calls += 1
        if self._get_returns_none:
            return None
        from imperal_sdk.types.models import Document
        data = self._rows.get(doc_id)
        return None if data is None else Document(id=doc_id, collection=collection, data=dict(data))

    async def query(self, collection, where=None, order_by=None, limit=100, cursor=None):
        self.query_calls += 1
        if self._query_raises:
            raise RuntimeError("gateway unreachable")
        from imperal_sdk.types.models import Document
        from imperal_sdk.types.pagination import Page
        wanted_user = (where or {}).get("user_id")
        docs = [
            Document(id=doc_id, collection=collection, data=dict(data))
            for doc_id, data in self._rows.items()
            if wanted_user is None or data.get("user_id") == wanted_user
        ]
        return Page(data=docs[:limit], has_more=False)


def _row(user_id="test_user"):
    return {
        "user_id": user_id,
        "kind": "image",
        "prompt": "rooftop antagonist",
        "model": "gemini-3-pro-image",
        "storage_path": "gemini/image/x.png",
        "mime_type": "image/png",
        "created_at": "2026-07-24T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_viewer_finds_generation_even_when_store_get_is_unscoped():
    """THE regression: get() returns None (as in production) -> must still resolve."""
    from handlers.panel import _image_viewer_panel

    ctx = make_ctx(with_key=True)
    ctx.store = _ScopedStore({"gen-1": _row()}, get_returns_none=True)
    await ctx.storage.upload("gemini/image/x.png", b"\x89PNG\r\n\x1a\n" + b"bytes" * 10,
                             content_type="image/png")

    result = await _image_viewer_panel(ctx, generation_id="gen-1")

    src = _find_image_src(result["ui"])
    assert src and src.startswith("data:image/png;base64,"), \
        "viewer must resolve the record via the user-scoped query path"


@pytest.mark.asyncio
async def test_viewer_accepts_nested_params_envelope():
    """ui.Call params may arrive nested; reading only the flat key looks like 'missing'."""
    from handlers.panel import _image_viewer_panel

    ctx = make_ctx(with_key=True)
    ctx.store = _ScopedStore({"gen-1": _row()}, get_returns_none=True)
    await ctx.storage.upload("gemini/image/x.png", b"\x89PNG\r\n\x1a\n" + b"bytes" * 10,
                             content_type="image/png")

    result = await _image_viewer_panel(ctx, params={"generation_id": "gen-1"})

    assert _find_image_src(result["ui"]), "nested params envelope must be understood"


@pytest.mark.asyncio
async def test_viewer_distinguishes_missing_id_from_missing_record():
    """The three failure modes must not share one vague message."""
    from handlers.panel import _image_viewer_panel
    import json

    ctx = make_ctx(with_key=True)
    ctx.store = _ScopedStore({"gen-1": _row()}, get_returns_none=True)

    no_id = json.dumps(await _image_viewer_panel(ctx))
    missing = json.dumps(await _image_viewer_panel(ctx, generation_id="does-not-exist"))

    assert "Nothing to show" in no_id
    assert "no longer in your generation history" in missing
    assert no_id != missing


@pytest.mark.asyncio
async def test_viewer_reports_storage_error_separately_from_missing():
    """A failing store must not be reported as 'that entry does not exist'."""
    from handlers.panel import _image_viewer_panel
    import json

    ctx = make_ctx(with_key=True)
    ctx.store = _ScopedStore({"gen-1": _row()}, query_raises=True)

    body = json.dumps(await _image_viewer_panel(ctx, generation_id="gen-1"))
    assert "try again" in body


@pytest.mark.asyncio
async def test_viewer_refuses_another_users_generation():
    """Ownership is still enforced after switching to the query path."""
    from handlers.panel import _image_viewer_panel
    import json

    ctx = make_ctx(with_key=True)
    ctx.store = _ScopedStore({"gen-1": _row(user_id="someone_else")}, get_returns_none=False)

    body = json.dumps(await _image_viewer_panel(ctx, generation_id="gen-1"))
    assert "data:image" not in body
