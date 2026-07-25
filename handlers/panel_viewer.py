"""Single-image viewer panel (``gemini_image``) + shared lookup helpers.

NOT the primary way to view an image any more: the history list renders it
INLINE in whichever panel was clicked (a self-call to the same panel_id),
because routing to a *second* panel needs that panel to be granted a render
path -- for ``slot="center"`` that historically meant the host's hardcoded
isCenterOverlay allowlist, which we are not in. That is why "View image"
appeared to do nothing. Kept as a fallback surface; owns the shared helpers.
The list stays zero media I/O -- only an explicitly requested image loads.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION

log = logging.getLogger("gemini.panel")

try:  # optional: only used to shrink oversized images before inlining
    from PIL import Image as _PILImage
except Exception:  # noqa: BLE001  (missing/broken install must not break the panel)
    _PILImage = None

# A real render is far heavier than the small test images that always worked
# (2048x1152 JPEG ~= 1.8M base64 chars), and an 8s cap could not even finish
# downloading one -- half of why "one generation opens, another does not".
_IMAGE_DOWNLOAD_TIMEOUT_S = 25.0

# Two ceilings, because there is NO documented panel payload limit to cite.
# SOFT = where shrinking is worth attempting, NOT a refusal threshold:
# refusing on an invented number would itself block a good render, i.e. the
# very bug being fixed. HARD = safety net for a pathological payload (a 4K
# render is ~7.2M chars, so that still gets through).
_INLINE_SOFT_MAX = 1_500_000
_INLINE_HARD_MAX = 9_000_000

_PREVIEW_MAX_DIM = 1400
_PREVIEW_JPEG_QUALITY = 82

# Why a load failed, so the UI can say something true instead of one blank
# "could not load" that hides four different problems.
FAIL_NONE = ""
FAIL_NO_FILE = "no_file"
FAIL_TIMEOUT = "timeout"
FAIL_ERROR = "error"
FAIL_TOO_LARGE = "too_large"

_FAIL_MESSAGES = {
    FAIL_NO_FILE: (
        "This entry has no stored file — it was created before generated "
        "files were kept, so there is nothing to load."
    ),
    FAIL_TIMEOUT: (
        "Reading the stored file took too long. Large renders can be slow — "
        "try again."
    ),
    FAIL_ERROR: (
        "The stored file for this generation could not be read just now — "
        "try again."
    ),
    FAIL_TOO_LARGE: (
        "This render is too large to display in the panel, and it could not "
        "be shrunk automatically."
    ),
}


def _failure_message(reason: str) -> str:
    return _FAIL_MESSAGES.get(reason, _FAIL_MESSAGES[FAIL_ERROR])


def _shrink(raw: bytes, mime_type: str) -> tuple[bytes, str]:
    """Re-encode an oversized image down to a panel-sized preview.

    Returns the original bytes unchanged when Pillow is unavailable or the
    image cannot be decoded -- the caller still enforces the size ceiling, so
    a failure here degrades to an honest "too large" message rather than a
    broken panel.
    """
    if _PILImage is None:
        log.info("panel: Pillow unavailable, cannot shrink oversized image")
        return raw, mime_type
    try:
        with _PILImage.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            im.thumbnail((_PREVIEW_MAX_DIM, _PREVIEW_MAX_DIM))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=_PREVIEW_JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:  # noqa: BLE001
        log.warning("panel: could not shrink image: %s", e)
        return raw, mime_type


async def _load_image(ctx, doc_data: dict) -> tuple[str, str]:
    """Fetch one generation's bytes as a ``data:`` URI.

    Returns ``(data_uri, failure_reason)`` -- exactly one is meaningful. The
    reason exists because collapsing timeout / read error / missing file /
    oversized into a single empty string made it impossible to tell why some
    generations opened and others did not.
    """
    storage_path = doc_data.get("storage_path")
    if not storage_path:
        return "", FAIL_NO_FILE

    try:
        raw = await asyncio.wait_for(
            ctx.storage.download(storage_path), timeout=_IMAGE_DOWNLOAD_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log.warning(
            "panel: image download timed out after %ss for %r",
            _IMAGE_DOWNLOAD_TIMEOUT_S, storage_path,
        )
        return "", FAIL_TIMEOUT
    except Exception as e:  # noqa: BLE001
        log.warning("panel: image download failed for %r: %s", storage_path, e)
        return "", FAIL_ERROR

    mime_type = doc_data.get("mime_type") or "image/png"
    encoded = base64.b64encode(raw).decode()
    if len(encoded) <= _INLINE_SOFT_MAX:
        return f"data:{mime_type};base64,{encoded}", FAIL_NONE

    # Over the soft mark: shrinking makes the panel lighter, but it is only an
    # OPTIMISATION. Pillow is not guaranteed to exist in the runtime (the
    # deploy validator environment has no third-party deps), so the image must
    # still be served when shrinking is unavailable or fails.
    log.info(
        "panel: %r inlines to %d base64 chars, trying to shrink",
        storage_path, len(encoded),
    )
    small_raw, small_mime = _shrink(raw, mime_type)
    small_encoded = base64.b64encode(small_raw).decode()
    if len(small_encoded) < len(encoded):
        encoded, mime_type = small_encoded, small_mime

    if len(encoded) > _INLINE_HARD_MAX:
        log.warning(
            "panel: %r is %d base64 chars -- beyond the hard cap, not inlining",
            storage_path, len(encoded),
        )
        return "", FAIL_TOO_LARGE
    return f"data:{mime_type};base64,{encoded}", FAIL_NONE

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


async def _image_data_uri(ctx, doc_data: dict) -> str:
    """Backwards-compatible wrapper around :func:`_load_image`.

    Kept so existing callers/tests that only care about the ``src`` keep
    working; new code should use ``_load_image`` and surface the reason.
    """
    src, _reason = await _load_image(ctx, doc_data)
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

    src, reason = await _load_image(ctx, doc.data)
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


ext.panel(
    "gemini_image", slot="center", title="Generated image", icon="Image",
    refresh="manual", center_overlay=True,
)(_image_viewer_panel)
