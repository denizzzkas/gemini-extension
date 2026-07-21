"""Gemini status/history handlers — connection check and generation history.

Split out of handlers/generate.py to keep both files under the 300-line
guideline; these two chat functions are read-only and logically distinct
from the write-heavy generate_image/generate_video handlers.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import chat
from gemini_config import (
    GENERATION_LOG_COLLECTION, DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT,
)
from return_models import GeminiConnectionRecord, GenerationHistoryItem, GenerationHistoryRecord
from handlers.media import _get_api_key, _absolute_url

log = logging.getLogger("gemini.status")


class CheckGeminiConnectionParams(BaseModel):
    pass


class ListGenerationHistoryParams(BaseModel):
    limit: int = Field(DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT, description="Max number of past generations to return")


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
            for doc in page.data
        ]
    except Exception as e:  # noqa: BLE001
        log.error("list_generation_history failed: %s", e)
        items = []

    record = GenerationHistoryRecord(items=items, count=len(items))
    return ActionResult.success(data=record, summary=f"Found {len(items)} recent generation(s).")
