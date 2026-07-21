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
) -> str:
    """Persist one generation log entry; returns its doc id (or '' on failure).

    ``storage_path`` is kept alongside ``url`` so a later generation can use
    THIS one as a reference image (re-downloaded via ctx.storage.download,
    not re-fetched by URL -- see _resolve_reference_images).
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
            "created_at": getattr(ctx.time, "now_utc", "") if getattr(ctx, "time", None) else "",
        })
        return doc.id
    except Exception as e:  # noqa: BLE001
        log.error("log_generation failed: %s", e)
        return ""


def _absolute_url(url: str) -> str:
    """Normalize a storage URL to an absolute, clickable link.

    ctx.storage.upload() can return a bare path (e.g.
    ``/storage/default/<ext>/<file>.jpg``) rather than a full URL -- pasted
    verbatim in chat that's dead text, not a link. Same IMPERAL_PUBLIC_HOST
    convention the SDK itself uses for ctx.webhook_url()/oauth_authorize_url().
    """
    if not url or url.startswith(("http://", "https://")):
        return url
    import os
    host = os.environ.get("IMPERAL_PUBLIC_HOST", "panel.imperal.io")
    return f"https://{host}{url}" if url.startswith("/") else f"https://{host}/{url}"


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
