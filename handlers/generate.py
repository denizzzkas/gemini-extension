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
    MODEL_IMAGE, MODEL_VIDEO, IMAGE_MODEL_CHOICES, GENERATION_LOG_COLLECTION,
    MAX_PROMPT_LEN, REQUEST_TIMEOUT_IMAGE, REQUEST_TIMEOUT_VIDEO,
    DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT,
)
from clients.gemini_client import create_interaction, build_reference_image_block, GeminiAPIError
from prompt_guide import image_prompt_guidance_text, video_prompt_guidance_text
from return_models import (
    GeneratedImageRecord, GeneratedVideoRecord, GeminiConnectionRecord,
    GenerationHistoryItem, GenerationHistoryRecord,
)

log = logging.getLogger("gemini.generate")


# ─── Param models ─────────────────────────────────────────────────────────── #

MAX_REFERENCE_IMAGES = 6  # extension-side safety cap, not a Google-documented limit

_MODEL_CHOICES_TEXT = "; ".join(
    f"{mid} ({info['label']}): {info['description']}"
    for mid, info in IMAGE_MODEL_CHOICES.items()
)


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
    model: str = Field(
        MODEL_IMAGE,
        description=(
            "Which Gemini image model to use -- defaults to Nano Banana Pro "
            "(best quality). Pick a faster/cheaper one for quick iterations "
            "or bulk generation, e.g. when the user says 'quick draft' or "
            "'don't need it perfect'. Options: " + _MODEL_CHOICES_TEXT
        ),
    )
    reference_generation_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_REFERENCE_IMAGES,
        description=(
            "Optional: IDs of this user's OWN past generations (from "
            "list_generation_history) to use as reference images for "
            "character/scene consistency -- e.g. 'use the same antagonist/"
            "rooftop as generation X'. Call list_generation_history first to "
            "get valid IDs; only this extension's own saved generations can "
            "be used as references (arbitrary external images are not "
            "supported yet)."
        ),
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
    return f"https://{host}{url if url.startswith('/') else '/' + url}"


async def _save_media(ctx, kind: str, mime_type: str, data_b64: str) -> tuple[str, str]:
    """Persist generated media bytes to ctx.storage; returns (storage_path, absolute_url).

    Both are needed downstream: ``storage_path`` to re-download the exact
    bytes later (e.g. as a reference image for a follow-up generation),
    ``absolute_url`` to show/link the result to the user.
    """
    try:
        raw = base64.b64decode(data_b64)
        ext = "png" if "png" in mime_type else ("jpg" if "jpe" in mime_type else ("mp4" if kind == "video" else "bin"))
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
        "Banana model family -- pick the model= param to trade off quality "
        "vs speed/cost (defaults to Nano Banana Pro, the best quality). "
        "Supports character/scene consistency: pass reference_generation_ids (from "
        "list_generation_history or a prior generate_image call's "
        "generation_id) to reuse the exact same character/setting from up "
        "to 6 of this user's own past image generations -- e.g. 'use the "
        "same antagonist/rooftop as generation X'. Only this extension's "
        "own saved generations work as references; arbitrary external "
        "images pasted into chat are NOT supported yet -- if the user "
        "wants to reuse an external/uploaded image, ask them to first "
        "generate or re-upload it through this extension so it gets a "
        "generation_id. " + image_prompt_guidance_text()
    ),
)
async def fn_generate_image(ctx, params: GenerateImageParams) -> ActionResult:
    """Generate an image via the Gemini Interactions API (Nano Banana Pro)."""
    if params.model not in IMAGE_MODEL_CHOICES:
        return ActionResult.error(
            f"Unknown image model {params.model!r}. Valid options: "
            f"{', '.join(IMAGE_MODEL_CHOICES)}.",
            retryable=False,
        )

    api_key = await _get_api_key(ctx)
    if not api_key:
        return ActionResult.error(
            "No Gemini API key configured. Add your key from Google AI Studio "
            "(aistudio.google.com/apikey) in the extension's Secrets panel."
        )

    reference_images = []
    if params.reference_generation_ids:
        reference_images = await _resolve_reference_images(ctx, params.reference_generation_ids)
        if not reference_images:
            return ActionResult.error(
                "None of the given reference_generation_ids could be resolved "
                "(not found, not owned by you, not an image, or predates this "
                "feature). Call list_generation_history to get valid IDs.",
                retryable=False,
            )

    try:
        result = await create_interaction(
            ctx, api_key, params.model, params.prompt,
            reference_images=reference_images or None,
            timeout=REQUEST_TIMEOUT_IMAGE,
        )
    except GeminiAPIError as e:
        log.error("generate_image failed: %s", e)
        return ActionResult.error(f"Image generation failed: {e.message}", retryable=e.status_code in (429, 500, 502, 503, 504))

    image = next((m for m in result.media if m.kind == "image"), None)
    if image is None:
        return ActionResult.error("Gemini did not return an image for this prompt. Try rephrasing it.", retryable=True)

    mime_type = image.mime_type or "image/png"
    storage_path, url = await _save_media(ctx, "image", mime_type, image.data_b64)
    generation_id = await _log_generation(ctx, "image", params.prompt, params.model, url=url, storage_path=storage_path, mime_type=mime_type)

    record = GeneratedImageRecord(
        generation_id=generation_id,
        prompt=params.prompt,
        model=params.model,
        mime_type=image.mime_type or "image/png",
        image_base64=image.data_b64,
        url=url,
        text=result.text,
    )
    return ActionResult.success(
        data=record,
        summary=(
            f"Generated an image for: \"{params.prompt}\". "
            "Show it inline in chat using the returned image_base64/mime_type "
            "(don't just paste the raw url as text -- render it as an image), "
            "and mention it's also saved in the Gemini Studio panel history."
        ),
    )


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

    mime_type = video.mime_type or "video/mp4"
    storage_path, url = await _save_media(ctx, "video", mime_type, video.data_b64)
    generation_id = await _log_generation(ctx, "video", params.prompt, MODEL_VIDEO, url=url, storage_path=storage_path, mime_type=mime_type)

    record = GeneratedVideoRecord(
        generation_id=generation_id,
        prompt=params.prompt,
        model=MODEL_VIDEO,
        mime_type=video.mime_type or "video/mp4",
        video_base64=video.data_b64,
        url=url,
        text=result.text,
    )
    return ActionResult.success(
        data=record,
        summary=(
            f"Generated a video for: \"{params.prompt}\". "
            "Show it inline in chat using the returned video_base64/mime_type "
            "(don't just paste the raw url as text -- render it as a video), "
            "and mention it's also saved in the Gemini Studio panel history."
        ),
    )


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
