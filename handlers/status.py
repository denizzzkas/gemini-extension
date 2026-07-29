"""Gemini status/history handlers — connection check and generation history.

Split out of handlers/generate.py to keep both files under the 300-line
guideline; these two chat functions are read-only and logically distinct
from the write-heavy generate_image/generate_video handlers.
"""
from __future__ import annotations

import base64
import logging

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import chat
from gemini_config import (
    GENERATION_LOG_COLLECTION, DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT,
)
from return_models import (
    GeminiConnectionRecord, GenerationHistoryItem, GenerationHistoryRecord,
    OriginalMediaRecord,
)
from handlers.media import _get_api_key, _absolute_url, newest_first
from handlers.panel_viewer import _find_generation

log = logging.getLogger("gemini.status")


class CheckGeminiConnectionParams(BaseModel):
    pass


class ListGenerationHistoryParams(BaseModel):
    limit: int = Field(DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT, description="Max number of past generations to return")


class GetOriginalMediaParams(BaseModel):
    generation_id: str = Field(..., description="The generation ID to fetch the untouched original for (from list_generation_history or a prior generate call)")


@chat.function(
    "check_gemini_connection",
    action_type="read",
    chain_callable=True,
    data_model=GeminiConnectionRecord,
    description="Check whether a Gemini API key is configured and whether the Gemini API is reachable.",
)
async def fn_check_gemini_connection(ctx, params: CheckGeminiConnectionParams) -> ActionResult:
    """User-facing connectivity check (distinct from the app-level health_check)."""
    api_key = await _get_api_key(ctx)
    configured = bool(api_key)
    api_reachable = False

    if configured:
        from gemini_config import GEMINI_API_BASE
        try:
            resp = await ctx.http.get(
                f"{GEMINI_API_BASE}/models",
                headers={"x-goog-api-key": api_key},
                timeout=5,
            )
            api_reachable = resp.status_code < 500
        except Exception as e:  # noqa: BLE001
            log.error("check_gemini_connection probe failed: %s", e)

    record = GeminiConnectionRecord(configured=configured, api_reachable=api_reachable)
    if not configured:
        summary = "No Gemini API key configured yet."
    elif api_reachable:
        summary = "Gemini API key is configured and reachable."
    else:
        summary = "Gemini API key is configured, but the API did not respond."
    return ActionResult.success(data=record, summary=summary)


@chat.function(
    "list_generation_history",
    action_type="read",
    chain_callable=True,
    data_model=GenerationHistoryRecord,
    description="List your recent Gemini image/video generations (prompt, model, timestamp).",
)
async def fn_list_generation_history(ctx, params: ListGenerationHistoryParams) -> ActionResult:
    """Return the caller's recent generation log entries."""
    try:
        page = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id},
            limit=params.limit,
        )
        items = [
            GenerationHistoryItem(
                id=doc.id,
                kind=doc.data.get("kind", ""),
                prompt=doc.data.get("prompt", ""),
                model=doc.data.get("model", ""),
                url=_absolute_url(doc.data.get("url", "")),
                created_at=doc.data.get("created_at", ""),
            )
            for doc in newest_first(page.data)
        ]
    except Exception as e:  # noqa: BLE001
        log.error("list_generation_history failed: %s", e)
        items = []

    record = GenerationHistoryRecord(items=items, count=len(items))
    return ActionResult.success(data=record, summary=f"Found {len(items)} recent generation(s).")


@chat.function(
    "get_original_media",
    action_type="read",
    chain_callable=True,
    data_model=OriginalMediaRecord,
    description=(
        "Fetch the UNTOUCHED original bytes of a past generation by its "
        "generation_id, so it can be shown at full resolution in chat -- use "
        "this when the panel could only show a shrunk preview or nothing at "
        "all, or whenever the user explicitly wants the original file "
        "instead of the preview."
    ),
)
async def fn_get_original_media(ctx, params: GetOriginalMediaParams) -> ActionResult:
    """Fetch the untouched original and hand it back in the SAME field name
    generate_image/generate_video already use to render inline
    (``image_base64``/``video_base64``), not a differently named field.

    That field-name mismatch (the record used to expose ``media_base64``)
    is the actual reason this button used to print a bare generation id
    instead of the picture: the LLM only renders inline media when its
    instructions name the exact field to look at, and nothing on this
    platform renders any base64 field automatically by convention.

    Chat is the channel already proven to render a full image (every
    generate_image reply does it), so this hands the SAME original bytes
    stored at generation time back through that channel.
    """
    doc, lookup_failed = await _find_generation(ctx, params.generation_id)
    if doc is None:
        msg = (
            "Could not read your generation history just now -- try again."
            if lookup_failed else
            "No generation with that id was found in your history."
        )
        return ActionResult.error(msg, retryable=lookup_failed)

    storage_path = doc.data.get("storage_path")
    if not storage_path:
        return ActionResult.error(
            "This generation has no stored file to fetch.", retryable=False,
        )

    try:
        raw = await ctx.storage.download(storage_path)
    except Exception as e:  # noqa: BLE001
        log.error("get_original_media: download failed for %r: %s", storage_path, e)
        return ActionResult.error(
            "Downloading the original from storage failed just now -- please try again.",
            retryable=True,
        )

    kind = doc.data.get("kind", "image")
    encoded = base64.b64encode(raw).decode()
    record = OriginalMediaRecord(
        generation_id=doc.id,
        kind=kind,
        prompt=doc.data.get("prompt", ""),
        model=doc.data.get("model", ""),
        mime_type=doc.data.get("mime_type", "") or "application/octet-stream",
        image_base64=encoded if kind == "image" else "",
        video_base64=encoded if kind == "video" else "",
    )
    field = "image_base64" if kind == "image" else "video_base64"
    return ActionResult.success(
        data=record,
        summary=(
            f"Here is the full-resolution original ({len(raw) // 1024} KB). "
            f"Render it inline in your reply using the returned {field} "
            f"(mime_type={record.mime_type}) exactly like a generate_{kind} "
            "reply -- this is the untouched file, not the panel preview."
        ),
    )
