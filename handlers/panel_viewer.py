"""Single-image viewer panel (``gemini_image``).

Split out of ``handlers/panel.py`` to stay under the 300-line file limit the
deploy validator enforces.

Why this is its own panel at all: the history list must never pay for media
I/O. Downloading bytes while rendering the list is what made the Studio panel
load forever (twice), so the list is strictly zero-I/O and each entry gets a
"View image" button that opens THIS panel, which fetches exactly one file.
A slow or failed storage read then costs one image instead of the whole panel.
"""
from __future__ import annotations

import asyncio
import base64
import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION

log = logging.getLogger("gemini.panel")

_IMAGE_DOWNLOAD_TIMEOUT_S = 8.0

# How many of the user's recent generations the viewer scans to resolve one id.
# The history list only shows DEFAULT_HISTORY_LIMIT, so anything the user can
# actually click is inside this window; the ceiling keeps the lookup bounded.
MAX_LOOKUP_SCAN = 200


async def _image_data_uri(ctx, doc_data: dict) -> str:
    """Download ONE generation's bytes and return a ``data:`` URI.

    Extension storage is readable only through the gateway's authenticated
    internal endpoint; no public host serves ``/storage/<tenant>/<ext>/...``
    (that path returns the panel's HTML shell / 404s, verified seconds after
    upload -- so it is not link expiry). A url-based ``<Image>`` can therefore
    only ever render broken, which is why the bytes are inlined as a ``data:``
    URI instead (the panel CSP allows ``img-src data:``).

    Returns ``""`` on failure/timeout so the caller can show an honest message.
    Deliberately does NOT fall back to ``doc_data["url"]``.
    """
    storage_path = doc_data.get("storage_path")
    if not storage_path:
        return ""
    try:
        raw = await asyncio.wait_for(
            ctx.storage.download(storage_path), timeout=_IMAGE_DOWNLOAD_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001  (includes asyncio.TimeoutError)
        log.warning("panel: image download failed for %r: %s", storage_path, e)
        return ""
    mime_type = doc_data.get("mime_type") or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"


def _param(params: dict, name: str) -> str:
    """Read a panel param, tolerating a nested ``params`` envelope.

    ``ui.Call(fn, **kw)`` serializes to ``{"action","function","params":{...}}``,
    so depending on how the host forwards a panel action the handler may be
    invoked as ``fn(ctx, generation_id=...)`` OR as ``fn(ctx, params={...})``.
    Reading only the flat key silently yields None in the second shape, which
    is indistinguishable from a genuinely missing record downstream.
    """
    value = params.get(name)
    if value:
        return str(value)
    nested = params.get("params")
    if isinstance(nested, dict) and nested.get(name):
        return str(nested[name])
    return ""


async def _find_generation(ctx, generation_id: str):
    """Resolve one generation for THIS user. Returns (doc_or_None, lookup_failed).

    Deliberately resolves via ``ctx.store.query`` rather than
    ``ctx.store.get(collection, id)``: ``get`` does not send ``user_id`` to the
    gateway (only extension/tenant), whereas ``query`` does -- and ``query`` is
    the call the history list already uses successfully. Using the same scoped
    path here removes the whole class of "the list shows it but the viewer
    cannot find it" mismatch that produced a hard "Not found" in production.
    Ownership is still re-checked below, so this never widens access.

    ``lookup_failed`` distinguishes "storage errored" from "no such row" so the
    UI can tell the user which one actually happened.
    """
    try:
        page = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id},
            limit=MAX_LOOKUP_SCAN,
        )
    except Exception as e:  # noqa: BLE001
        log.error("image panel: lookup failed for %r: %s", generation_id, e)
        return None, True

    for doc in page.data:
        if doc.id == generation_id:
            # Re-assert ownership: query is user-scoped, but never rely on a
            # single layer for an authorization decision.
            if doc.data.get("user_id") != ctx.user.imperal_id:
                log.warning("image panel: ownership mismatch for %r", generation_id)
                return None, False
            return doc, False

    log.info("image panel: %r not in this user's history", generation_id)
    return None, False


def _page(node: ui.UINode) -> dict:
    tree = ui.Page(title="Generated image", children=[node])
    return {"ui": tree.to_dict(), "panel_id": "gemini_image"}


async def _image_viewer_panel(ctx, **params) -> dict:
    """Show ONE generation's image, fetched on demand."""
    generation_id = _param(params, "generation_id")

    if not generation_id:
        # The click never carried an id -- a UI/transport problem, NOT a
        # missing record. Saying "not found" here sent the last investigation
        # hunting the store for a document that was never asked for.
        log.error("image panel: no generation_id in params (keys=%s)", sorted(params))
        return _page(ui.Alert(
            title="Nothing to show",
            message=(
                "This viewer opened without a generation id, so there is no "
                "image to load. Re-open Gemini Studio and click View image "
                "on a specific entry."
            ),
            type="warn",
        ))

    doc, lookup_failed = await _find_generation(ctx, generation_id)

    if doc is None:
        return _page(ui.Alert(
            title="Could not load that generation",
            message=(
                "Reading it from storage failed just now — please try again."
                if lookup_failed else
                "That entry is no longer in your generation history."
            ),
            type="warn",
        ))

    src = await _image_data_uri(ctx, doc.data)
    prompt = doc.data.get("prompt", "")
    if not src:
        return _page(ui.Alert(
            title="Image unavailable",
            message=(
                "The stored file for this generation could not be read. "
                "Older entries created before files were kept may no longer exist."
            ),
            type="warn",
        ))

    return _page(ui.Stack(gap=2, children=[
        ui.Image(src=src, alt=prompt, width="100%", caption=prompt),
    ]))


ext.panel(
    "gemini_image", slot="center", title="Generated image", icon="Image",
    refresh="manual", center_overlay=True,
)(_image_viewer_panel)
