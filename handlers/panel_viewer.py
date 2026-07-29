"""Single-image viewer panel (``gemini_image``, UNREGISTERED) + shared lookup helpers.

NOT the primary way to view an image: full detail (image, prompt, download,
regenerate) now lives in ``gemini_studio``, the restored centre panel (see
handlers/panel.py and handlers/panel_detail.py), which correctly declares
``center_overlay=True`` and does render. This module predates that fix and
used to be registered on the SAME ``slot="center"`` as ``gemini_studio`` --
one slot shows one panel, so whichever the host opened made the other
unreachable. It is kept deliberately UNREGISTERED (see the bottom of this
file) purely so its lookup/rendering helpers (``_find_generation``,
``_load_image`` via handlers/image_loader.py, the accordion sentinel) stay
importable for the real panels and their tests, without a second panel
fighting Studio for the centre slot.
"""
from __future__ import annotations

import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION
from handlers.image_loader import (
    FAIL_ERROR, FAIL_NO_FILE, FAIL_NONE, FAIL_TIMEOUT, FAIL_TOO_LARGE,
    PREVIEW_FIELD, PREVIEW_MIME_FIELD, _cache_preview, _failure_message,
    _load_image,
)

log = logging.getLogger("gemini.panel")

# Re-exported above so existing callers (handlers/panel.py) and tests keep
# importing these from panel_viewer unchanged after the split.
__all__ = [
    "CLOSED_SENTINEL", "FAIL_ERROR", "FAIL_NO_FILE", "FAIL_NONE",
    "FAIL_TIMEOUT", "FAIL_TOO_LARGE", "PREVIEW_FIELD", "PREVIEW_MIME_FIELD",
    "_cache_preview", "_failure_message", "_find_generation", "_load_image",
]

# Sentinel value that CLOSES an open image. The host accumulates params per
# panel_id and merges a re-fetch's ``{}`` INTO them, so a param-less self-call
# can never clear an open generation_id -- which is why "Hide" did nothing
# while "View image" worked: opening ADDS a param, closing must REMOVE one,
# and removal is not expressible. Overwriting is the only reset, and the value
# must be non-empty: a falsy one risks being dropped before the merge.
CLOSED_SENTINEL = "__closed__"

# How many of the user's recent generations the viewer scans to resolve one id.
# The history list only shows DEFAULT_HISTORY_LIMIT, so anything the user can
# actually click is inside this window; the ceiling keeps the lookup bounded.
MAX_LOOKUP_SCAN = 200


async def _image_data_uri(ctx, doc_data: dict, doc_id: str | None = None) -> str:
    """Backwards-compatible wrapper around :func:`_load_image`.

    Kept so existing callers/tests that only care about the ``src`` keep
    working; new code should use ``_load_image`` and surface the reason.
    """
    src, _reason = await _load_image(ctx, doc_data, doc_id)
    return src


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


def _opened_id(params: dict) -> str:
    """Which generation is open, honouring :data:`CLOSED_SENTINEL`.

    Returns ``""`` when nothing should be expanded, so callers treat an
    explicit close exactly like a fresh render.
    """
    value = _param(params, "generation_id")
    return "" if value == CLOSED_SENTINEL else value


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

    src, reason = await _load_image(ctx, doc.data, doc.id)
    prompt = doc.data.get("prompt", "")
    if not src:
        return _page(ui.Alert(
            title="Image unavailable",
            message=_failure_message(reason),
            type="warn",
        ))

    return _page(ui.Stack(gap=2, children=[
        ui.Image(src=src, alt=prompt, width="100%", caption=prompt),
    ]))


# DELIBERATELY NOT REGISTERED AS A PANEL.
#
# It used to sit on slot="center" -- the SAME slot as gemini_studio. One slot
# shows one panel: the host opened THIS one, with no params, so the centre was
# stuck on its "Nothing to show / opened without a generation id" dead end and
# the real Studio was unreachable. Nothing calls it either -- viewing happens
# inline via a self-call in the panel the user is already looking at.
# Kept (with its tests) as documentation of the on-demand load path.
