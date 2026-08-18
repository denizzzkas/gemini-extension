"""Gemini media helpers — API key lookup, generation logging, storage save,
and reference-image resolution.

Split out of handlers/generate.py to keep both files under the 300-line
guideline; handlers.generate re-imports the names it needs, so callers/tests
that import from handlers.generate directly keep working unchanged.
"""
from __future__ import annotations

import base64
import logging
import uuid

from core.preview import build_preview
from gemini_config import GENERATION_LOG_COLLECTION
from clients.gemini_client import build_reference_image_block

log = logging.getLogger("gemini.media")

# Extension-side safety cap on reference_generation_ids, not a
# Google-documented Gemini API limit.
MAX_REFERENCE_IMAGES = 6


async def _get_api_key(ctx) -> str | None:
    try:
        return await ctx.secrets.get("gemini_api_key")
    except Exception as e:  # noqa: BLE001
        log.error("get_api_key failed: %s", e)
        return None


async def _log_generation(
    ctx, kind: str, prompt: str, model: str, *,
    url: str = "", storage_path: str = "", mime_type: str = "",
    source: str = "generated", reference_ids: list[str] | None = None,
) -> str:
    """Persist one generation log entry; returns its doc id (or '' on failure).

    ``storage_path`` is kept alongside ``url`` so a later generation can use
    THIS one as a reference image (re-downloaded via ctx.storage.download,
    not re-fetched by URL -- see _resolve_reference_images).

    ``source`` separates images you GENERATED from ones you UPLOADED as
    references -- both are images you own, but only the first are results.
    ``reference_ids`` records which images fed this generation: without it the
    viewer cannot show "made from this reference", and regenerating with the
    same inputs is impossible because the inputs were never written down.
    """
    try:
        doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": kind,
            "prompt": prompt,
            "model": model,
            "url": url,
            "storage_path": storage_path,
            "mime_type": mime_type,
            "source": source,
            "reference_ids": list(reference_ids or []),
            "created_at": getattr(ctx.time, "now_utc", "") if getattr(ctx, "time", None) else "",
        })
        return doc.id
    except Exception as e:  # noqa: BLE001
        log.error("log_generation failed: %s", e)
        return ""


def _absolute_url(url: str) -> str:
    """Normalize a storage reference to an absolute form.

    WARNING -- the result is NOT a publicly viewable link. Extension storage
    is served only from the gateway's authenticated internal endpoint
    (/v1/internal/storage/download + Bearer token, see the SDK StorageClient);
    no public host serves /storage/<tenant>/<ext>/<file>. Fetching this URL
    from a browser returns HTTP 404 with the panel's HTML shell (verified
    against panel.imperal.io and imperal.io). It is kept only as a stable,
    absolute identifier for logging/record-keeping purposes -- to actually
    SHOW media,
    re-download the bytes via ctx.storage.download(storage_path).
    """
    if not url or url.startswith(("http://", "https://")):
        return url
    import os
    host = os.environ.get("IMPERAL_PUBLIC_HOST", "panel.imperal.io")
    return f"https://{host}{url}" if url.startswith("/") else f"https://{host}/{url}"


async def _attach_preview(
    ctx, generation_id: str, data_b64: str, mime_type: str,
) -> tuple[str, str] | None:
    """Build a panel-sized preview at generation time and store it on the record.

    Returns ``(preview_b64, preview_mime)`` when a preview was built and
    stored, ``None`` otherwise -- callers that need the actual bytes (e.g. to
    put a compressed preview straight into a chat reply, not just cache it)
    get them for free instead of re-reading the record they just wrote. Best
    effort throughout: a failure here costs a slower first open (the panel
    rebuilds it on demand), never a failed generation, so nothing raises.

    Why at generation time: the bytes are already in memory, so this avoids a
    storage download plus ~0.5-1.3s of pure-Python shrinking for whoever opens
    the image first. Why stored in the document store rather than as a file:
    the panel needs the preview INSIDE its response payload, and a store read
    delivers exactly that -- an extra file would still have to be downloaded
    and base64-encoded on every open.
    """
    if not generation_id:
        return None
    try:
        raw = base64.b64decode(data_b64)
        preview = build_preview(raw, mime_type)
        if preview is None:
            log.info("preview: none built for %s (%s)", generation_id, mime_type)
            return None
        encoded, preview_mime = preview
        await ctx.store.update(GENERATION_LOG_COLLECTION, generation_id, {
            "preview_b64": encoded,
            "preview_mime": preview_mime,
        })
        log.info(
            "preview: stored %d base64 chars for %s", len(encoded), generation_id,
        )
        return encoded, preview_mime
    except Exception as e:  # noqa: BLE001
        log.warning("preview: could not attach for %s: %s", generation_id, e)
        return None


async def _save_media(ctx, kind: str, mime_type: str, data_b64: str) -> tuple[str, str]:
    """Persist generated media bytes to ctx.storage; returns (storage_path, absolute_url).

    Both are needed downstream: ``storage_path`` to re-download the exact
    bytes later (e.g. as a reference image for a follow-up generation),
    ``absolute_url`` to show/link the result to the user.
    """
    try:
        raw = base64.b64decode(data_b64)
        ext = "png" if "png" in mime_type else ("jpg" if "jpeg" in mime_type or "jpg" in mime_type else ("mp4" if kind == "video" else "bin"))
        path = f"gemini/{kind}/{uuid.uuid4().hex}.{ext}"
        info = await ctx.storage.upload(path, raw, content_type=mime_type or "application/octet-stream")
        return path, _absolute_url(info.url or "")
    except Exception as e:  # noqa: BLE001
        log.error("save_media failed: %s", e)
        return "", ""


async def _resolve_reference_images(ctx, generation_ids: list[str]) -> list[dict]:
    """Turn past-generation doc IDs into Gemini-shaped reference image blocks.

    Only this user's own logged generations are resolvable (scoped by
    user_id) -- re-downloads the exact saved bytes via ctx.storage.download()
    rather than re-fetching by URL, so the bytes can't be corrupted in transit
    (see build_reference_image_block's docstring for why that matters).
    Silently skips any ID that can't be resolved (missing doc, no stored
    path, download failure, or a video entry) -- a bad reference ID
    shouldn't hard-fail the whole generation.
    """
    blocks: list[dict] = []
    for gen_id in generation_ids[:MAX_REFERENCE_IMAGES]:
        try:
            doc = await ctx.store.get(GENERATION_LOG_COLLECTION, gen_id)
            if doc is None or doc.data.get("user_id") != ctx.user.imperal_id:
                log.warning("reference image %r not found or not owned by caller", gen_id)
                continue
            if doc.data.get("kind") != "image":
                log.warning("reference %r is not an image generation, skipping", gen_id)
                continue
            storage_path = doc.data.get("storage_path")
            if not storage_path:
                log.warning("reference %r has no stored path (predates this feature)", gen_id)
                continue
            raw = await ctx.storage.download(storage_path)
            mime_type = doc.data.get("mime_type") or "image/png"
            blocks.append(build_reference_image_block(mime_type, raw))
        except Exception as e:  # noqa: BLE001
            log.error("resolve reference image %r failed: %s", gen_id, e)
    return blocks


def newest_first(docs: list) -> list:
    """Sort generation docs newest-first by ``created_at``.

    ``ctx.store.query`` takes an ``order_by``, but its accepted syntax is not
    documented anywhere in the SDK, and a wrong value risks failing the query
    outright. Sorting here is cheap (a page of rows) and guaranteed.

    Without this the row order is whatever the backend returns, so a capped
    page could omit recent generations unpredictably -- which is exactly why
    some images "could not be seen" at all, and why the skeleton's
    ``limit=1`` "last generation" was not necessarily the last one.
    """
    return sorted(
        docs, key=lambda d: str(d.data.get("created_at") or ""), reverse=True,
    )

