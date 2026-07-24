"""Tests for the single-image viewer panel (``gemini_image``).

Split out of tests/test_panel.py to keep both files under the 300-line limit
the deploy validator enforces, mirroring the handlers/panel_viewer.py split.
"""
from __future__ import annotations

import pytest

from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx


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
