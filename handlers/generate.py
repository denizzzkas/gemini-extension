"""Gemini generation handlers — image and video, plus connection check."""
from __future__ import annotations

import base64
import logging
import time
import uuid

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import ext, chat
from gemini_config import (
    MODEL_IMAGE, MODEL_VIDEO, GENERATION_LOG_COLLECTION,
    MAX_PROMPT_LEN, REQUEST_TIMEOUT_IMAGE, REQUEST_TIMEOUT_VIDEO,
    DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT,
)
from clients.gemini_client import create_interaction, GeminiAPIError
from prompt_guide import image_prompt_guidance_text, video_prompt_guidance_text
from return_models import (
    GeneratedImageRecord, GeneratedVideoRecord, GeminiConnectionRecord,
    GenerationHistoryItem, GenerationHistoryRecord,
)

log = logging.getLogger("gemini.generate")


# ─── Param models ─────────────────────────────────────────────────────────── #

class GenerateImageParams(BaseModel):
    prompt: str = Field(
        ...,
        description=(
            "Fully-specified description of the image to generate or edit "
            "-- expand short/vague user requests into a Google-recommended "
            "structured prompt (subject, setting, light, camera/lens for "
            "photorealistic shots; style+medium for illustrations; explicit "
            "on-image text + font/style for text-in-image; etc.) before "
            "passing it here. See tool description for the full template set."
        ),
        min_length=1, max_length=MAX_PROMPT_LEN,
    )


class GenerateVideoParams(BaseModel):
    prompt: str = Field(
        ...,
        description=(
            "Fully-specified description of the video to generate -- expand "
            "short/vague user requests into a Google-recommended structured "
            "prompt covering subject, action, style, camera positioning/"
            "motion, composition, focus/lens and ambiance, plus quoted "
            "dialogue/SFX/ambient-noise cues if audio matters. See tool "
            "description for the full element list."
        ),
        min_length=1, max_length=MAX_PROMPT_LEN,
    )


class CheckGeminiConnectionParams(BaseModel):
    pass


class ListGenerationHistoryParams(BaseModel):
    limit: int = Field(DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT, description="Max number of past generations to return")


# ─── Helpers ──────────────────────────────────────────────────────────────── #

async def _get_api_key(ctx) -> str | None:
    try:
        return await ctx.secrets.get("gemini_api_key")
    except Exception as e:  # noqa: BLE001
        log.error("get_api_key failed: %s", e)
        return None


async def _log_generation(ctx, kind: str, prompt: str, model: str, url: str = "") -> None:
    try:
        await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": kind,
            "prompt": prompt,
            "model": model,
            "url": url,
            "created_at": getattr(ctx.time, "now_utc", "") if getattr(ctx, "time", None) else "",
        })
    except Exception as e:  # noqa: BLE001
        log.error("log_generation failed: %s", e)


async def _save_media(ctx, kind: str, mime_type: str, data_b64: str) -> str:
    """Persist generated media bytes to ctx.storage; returns a URL (or '' on failure)."""
    try:
        raw = base64.b64decode(data_b64)
        ext = "png" if "png" in mime_type else ("jpg" if "jpe" in mime_type else ("mp4" if kind == "video" else "bin"))
        path = f"gemini/{kind}/{uuid.uuid4().hex}.{ext}"
        info = await ctx.storage.upload(path, raw, content_type=mime_type or "application/octet-stream")
        return info.url or ""
    except Exception as e:  # noqa: BLE001
        log.error("save_media failed: %s", e)
        return ""


# ─── Handlers ─────────────────────────────────────────────────────────────── #

@chat.function(
    "generate_image",
    action_type="write",
    chain_callable=True,
    effects=["create:media"],
    event="gemini.image_generated",
    data_model=GeneratedImageRecord,
    description=(
        "Generate or edit an image from a text prompt using Google's Nano "
        "Banana Pro (Gemini 3 Pro Image) model. " + image_prompt_guidance_text()
    ),
)
async def fn_generate_image(ctx, params: GenerateImageParams) -> ActionResult:
    """Generate an image via the Gemini Interactions API (Nano Banana Pro)."""
    api_key = await _get_api_key(ctx)
    if not api_key:
        return ActionResult.error(
            "No Gemini API key configured. Add your key from Google AI Studio "
            "(aistudio.google.com/apikey) in the extension's Secrets panel."
        )

    try:
        result = await create_interaction(
            ctx, api_key, MODEL_IMAGE, params.prompt, timeout=REQUEST_TIMEOUT_IMAGE,
        )
    except GeminiAPIError as e:
        log.error("generate_image failed: %s", e)
        return ActionResult.error(f"Image generation failed: {e.message}", retryable=e.status_code in (429, 500, 502, 503, 504))

    image = next((m for m in result.media if m.kind == "image"), None)
    if image is None:
        return ActionResult.error("Gemini did not return an image for this prompt. Try rephrasing it.", retryable=True)

    url = await _save_media(ctx, "image", image.mime_type or "image/png", image.data_b64)
    await _log_generation(ctx, "image", params.prompt, MODEL_IMAGE, url=url)

    record = GeneratedImageRecord(
        prompt=params.prompt,
        model=MODEL_IMAGE,
        mime_type=image.mime_type or "image/png",
        image_base64=image.data_b64,
        url=url,
        text=result.text,
    )
    return ActionResult.success(data=record, summary=f"Generated an image for: \"{params.prompt}\"")


@chat.function(
    "generate_video",
    action_type="write",
    chain_callable=True,
    effects=["create:media"],
    event="gemini.video_generated",
    data_model=GeneratedVideoRecord,
    description=(
        "Generate a short video with audio from a text prompt using Google's "
        "Gemini Omni Flash model. " + video_prompt_guidance_text()
    ),
)
async def fn_generate_video(ctx, params: GenerateVideoParams) -> ActionResult:
    """Generate a video via the Gemini Interactions API (Gemini Omni Flash)."""
    api_key = await _get_api_key(ctx)
    if not api_key:
        return ActionResult.error(
            "No Gemini API key configured. Add your key from Google AI Studio "
            "(aistudio.google.com/apikey) in the extension's Secrets panel."
        )

    try:
        result = await create_interaction(
            ctx, api_key, MODEL_VIDEO, params.prompt, timeout=REQUEST_TIMEOUT_VIDEO,
        )
    except GeminiAPIError as e:
        log.error("generate_video failed: %s", e)
        return ActionResult.error(f"Video generation failed: {e.message}", retryable=e.status_code in (429, 500, 502, 503, 504))

    video = next((m for m in result.media if m.kind == "video"), None)
    if video is None:
        return ActionResult.error("Gemini did not return a video for this prompt. Try rephrasing it.", retryable=True)

    url = await _save_media(ctx, "video", video.mime_type or "video/mp4", video.data_b64)
    await _log_generation(ctx, "video", params.prompt, MODEL_VIDEO, url=url)

    record = GeneratedVideoRecord(
        prompt=params.prompt,
        model=MODEL_VIDEO,
        mime_type=video.mime_type or "video/mp4",
        video_base64=video.data_b64,
        url=url,
        text=result.text,
    )
    return ActionResult.success(data=record, summary=f"Generated a video for: \"{params.prompt}\"")


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
                kind=doc.data.get("kind", ""),
                prompt=doc.data.get("prompt", ""),
                model=doc.data.get("model", ""),
                url=doc.data.get("url", ""),
                created_at=doc.data.get("created_at", ""),
            )
            for doc in page.data
        ]
    except Exception as e:  # noqa: BLE001
        log.error("list_generation_history failed: %s", e)
        items = []

    record = GenerationHistoryRecord(items=items, count=len(items))
    return ActionResult.success(data=record, summary=f"Found {len(items)} recent generation(s).")
