"""Loading one generation's image as a panel-ready ``data:`` URI.

Split out of ``handlers/panel_viewer.py`` to keep both files under the
300-line deploy guideline (exceeding it cost a deploy point before).

This is where the "one generation opens, another does not" bug lived. The
cause was measured, not guessed: a panel response carries the image inline
as base64, and there is an undocumented ceiling on its size. In production,
~90k and ~127k base64 chars display; ~954k and ~1.25M do not. Real renders
land far above that, tiny test images far below -- hence the split
behaviour.

Two earlier fixes failed for the same reason: they shrank the image with
Pillow, which the production runtime does not have
(``pillow_available: false``), so the code never ran. The shrink now goes
through :mod:`core.preview`, which is pure stdlib and therefore real.
"""
from __future__ import annotations

import asyncio
import base64
import logging

from core.preview import PROVEN_GOOD_CHARS, build_preview
from gemini_config import GENERATION_LOG_COLLECTION

log = logging.getLogger("gemini.image_loader")

# A real render is far heavier than the small test images that always worked
# (2048x1152 JPEG ~= 1.8M base64 chars), and an 8s cap could not even finish
# downloading one -- half of why "one generation opens, another does not".
_IMAGE_DOWNLOAD_TIMEOUT_S = 25.0

# Above the size PROVEN to display, a preview is built instead of inlining the
# original (see core/preview.py for the measurements).
_INLINE_SAFE_MAX = PROVEN_GOOD_CHARS

# When no preview could be built at all (video -- can_preview() is False for
# every video mime type -- or bytes this decoder can't read), the original is
# still served if it falls in the UNVERIFIED middle ground between what is
# proven to display (~127k, _INLINE_SAFE_MAX) and what is proven NOT to
# (~954k/1.25M, both measured live -- see core/preview.py's own table): a
# refusal there would guarantee failure, while nothing has actually confirmed
# that size fails. Past the proven-bad mark, though, "might still work" stops
# being true -- refuse there instead of guaranteeing the same blown-reply
# crash a multi-MB video produced every time it went out through this
# unbounded branch. Kept a little under 954k, not equal to it, since that
# number is itself a single measured point, not a hard boundary.
_UNVERIFIED_SERVE_MAX = 900_000

# Field on the generation record holding a cached preview, so the pure-Python
# shrink runs once per image rather than on every open. Documents have no
# size limit in the SDK's StoreClient, and a preview is only ~20-80KB.
PREVIEW_FIELD = "preview_b64"
PREVIEW_MIME_FIELD = "preview_mime"

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


async def _cache_preview(
    ctx, doc_data: dict, encoded: str, mime_type: str,
    doc_id: str | None = None,
) -> None:
    """Persist a built preview on the generation record (best effort).

    Building one costs seconds of pure-Python entropy decoding on a large
    JPEG, so caching turns every subsequent open into a plain document read
    with no storage download at all. Failure here is deliberately silent: the
    preview was already produced, so the user still sees the image.

    ``doc_id`` is passed EXPLICITLY by the caller. It used to be dug out of
    ``doc_data`` with ``.get("id") or .get("_id")``, but the store keeps the id
    on the document WRAPPER (``doc.id``) and every caller passes ``doc.data``,
    which never contains it. So the lookup always failed, this function always
    returned early, and the cache was never written -- a large image paid the
    full multi-second decode on EVERY view. The ``doc_data`` fallback is kept
    for any caller that really does hand over a dict carrying its own id.
    """
    doc_id = doc_id or doc_data.get("id") or doc_data.get("_id")
    if not doc_id:
        log.info("panel: no id available, preview not cached")
        return
    try:
        await ctx.store.update(GENERATION_LOG_COLLECTION, str(doc_id), {
            PREVIEW_FIELD: encoded,
            PREVIEW_MIME_FIELD: mime_type,
        })
    except Exception as e:  # noqa: BLE001
        log.info("panel: could not cache preview: %s", e)


async def _load_image(
    ctx, doc_data: dict, doc_id: str | None = None,
) -> tuple[str, str]:
    """Fetch one generation's bytes as a ``data:`` URI.

    ``doc_id`` is optional only for backwards compatibility; pass it whenever
    you have it, or the preview built here cannot be cached and the next view
    of the same image repeats the whole multi-second decode.

    Returns ``(data_uri, failure_reason)`` -- exactly one is meaningful. The
    reason exists because collapsing timeout / read error / missing file /
    oversized into a single empty string made it impossible to tell why some
    generations opened and others did not.

    Order of preference:

    1. A preview already cached on the record -- no download, no re-encode.
    2. The original, inlined whole, when it is under the size proven to
       display.
    3. A preview built here with the stdlib (:mod:`core.preview`), then
       cached for next time.
    4. The original anyway. Unverified, but it is what the user got before,
       so a possibly-working payload beats a certain error.
    """
    cached = doc_data.get(PREVIEW_FIELD)
    if cached:
        mime = doc_data.get(PREVIEW_MIME_FIELD) or "image/png"
        return f"data:{mime};base64,{cached}", FAIL_NONE

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
    if len(encoded) <= _INLINE_SAFE_MAX:
        return f"data:{mime_type};base64,{encoded}", FAIL_NONE

    log.info(
        "panel: %r inlines to %d base64 chars (over the %d proven to display), "
        "building a preview", storage_path, len(encoded), _INLINE_SAFE_MAX,
    )
    preview = build_preview(raw, mime_type)
    if preview is not None:
        small_encoded, small_mime = preview
        await _cache_preview(ctx, doc_data, small_encoded, small_mime, doc_id)
        # Mirror onto the SAME dict the caller holds, not just the DB. A
        # preview built here is only ever persisted going forward -- the
        # caller's already-loaded doc_data would keep reading empty until
        # the NEXT view, which meant the download button's cached-preview
        # fallback (handlers/panel_html.download_block's fallback_b64) had
        # nothing to offer on exactly the first view of a large image, i.e.
        # exactly when the original is over DOWNLOAD_CEILING_CHARS and this
        # fallback is the only download available at all. doc_data is the
        # actual dict the caller passed (doc.data, not a copy), so this
        # mutation is visible to it immediately.
        doc_data[PREVIEW_FIELD] = small_encoded
        doc_data[PREVIEW_MIME_FIELD] = small_mime
        return f"data:{small_mime};base64,{small_encoded}", FAIL_NONE

    # No preview possible -- either the format can't be shrunk at all
    # (video: core.preview.can_preview() is False for every video mime type,
    # so this path is EVERY video, not an edge case), or the bytes could not
    # be decoded (a JPEG variant the decoder doesn't handle, corrupt data).
    #
    # This used to serve the original completely unconditionally. That is
    # right for the genuinely UNVERIFIED middle ground documented above this
    # function (~127k proven to display, ~954k/1.25M proven NOT to -- nothing
    # in between has ever been measured either way), which is why a moderate
    # overage still gets served on the chance it fits: refusing would
    # GUARANTEE failure, while an unverified size might not. But "serve it
    # anyway" stops being a reasonable bet once the payload is already past
    # the point genuinely PROVEN to fail (~954k) -- and it was never bounded
    # at all, so a multi-MB video (measured: several million base64 chars,
    # tens of times past even the proven-bad mark) went out through this
    # exact branch every time, deterministically blowing the whole reply.
    # This refuses only that clearly-hopeless tier, honestly, instead of
    # guaranteeing the same truncated-reply crash the unverified middle
    # ground is deliberately still allowed to risk.
    if len(encoded) <= _UNVERIFIED_SERVE_MAX:
        return f"data:{mime_type};base64,{encoded}", FAIL_NONE

    log.warning(
        "panel: no preview could be built for %r (%s) and the original is "
        "%d base64 chars -- past the %d mark proven to fail, refusing "
        "instead of guaranteeing a blown reply cap", storage_path, mime_type,
        len(encoded), _UNVERIFIED_SERVE_MAX,
    )
    return "", FAIL_TOO_LARGE
