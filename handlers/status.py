"""Gemini status/history handlers — connection check and generation history.

Split out of handlers/generate.py to keep both files under the 300-line
guideline; these two chat functions are read-only and logically distinct
from the write-heavy image/video generation handlers.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import chat
from return_models import GeminiConnectionRecord, SavedSecretResult
from handlers.media import _get_api_key

log = logging.getLogger("gemini.status")


class CheckGeminiConnectionParams(BaseModel):
    pass


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


class SaveGeminiAPIKeyParams(BaseModel):
    api_key: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="The Gemini API key to store for this user (from aistudio.google.com/apikey).",
    )


@chat.function(
    "save_gemini_api_key",
    action_type="write",
    chain_callable=True,
    effects=["update:secret"],
    event="gemini.save_gemini_api_key",
    data_model=SavedSecretResult,
    description=(
        "Save the user's own Gemini API key so their generations can run. "
        "``gemini_api_key`` is scoped per-user, not per-app, so each user "
        "brings their own -- this is what the left panel's inline key field "
        "submits to."
    ),
)
async def fn_save_gemini_api_key(ctx, params: SaveGeminiAPIKeyParams) -> ActionResult:
    """Write the per-user secret from the panel's own inline field.

    Requires ``gemini_api_key`` declared with ``write_mode="both"`` (see
    app.py) -- ``write_mode="user"`` makes ``ctx.secrets.set()`` raise
    SecretWriteForbidden unconditionally, which would make this a dead button.
    """
    value = params.api_key.strip()
    if not value:
        return ActionResult.error("That key looks empty — paste your Gemini API key and try again.", retryable=True)
    try:
        await ctx.secrets.set("gemini_api_key", value)
    except Exception as e:  # noqa: BLE001
        log.error("save_gemini_api_key failed: %s", e)
        return ActionResult.error(
            "Could not save that key just now — try again.", retryable=True,
        )
    return ActionResult.success(
        data=SavedSecretResult(configured=True),
        summary="Gemini API key saved. You're connected!",
        refresh_panels=["gemini_quick"],
    )
