"""Gemini generation handlers — image and video, plus connection check."""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import ext, chat
from gemini_config import (
    MODEL_IMAGE, MODEL_VIDEO, IMAGE_MODEL_CHOICES,
    IMAGE_SIZE_CHOICES, DEFAULT_IMAGE_SIZE,
    MAX_PROMPT_LEN, REQUEST_TIMEOUT_VIDEO,
)
from clients.gemini_client import create_interaction, GeminiAPIError
from prompt_guide import image_prompt_guidance_text, video_prompt_guidance_text
from return_models import GeneratedImageRecord, GeneratedVideoRecord
from handlers.image_core import run_image_generation
from handlers.media import (
    MAX_REFERENCE_IMAGES, _get_api_key, _log_generation, _save_media,
)

log = logging.getLogger("gemini.generate")


# ─── Param models ─────────────────────────────────────────────────────────── #

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
    image_size: str = Field(
        DEFAULT_IMAGE_SIZE,
        description=(
            "Output resolution. Defaults to 1K, and that default is "
            "deliberate: a 2K/4K render produces a payload too large to "
            "display inline, and the production runtime cannot downscale it "
            "afterwards. Only raise this if the user explicitly asks for "
            "maximum detail. Options: "
            + "; ".join(f"{k} ({v})" for k, v in IMAGE_SIZE_CHOICES.items())
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
    """Generate an image, with the model chosen by the ``model`` parameter.

    Kept for backwards compatibility (automations and saved chains may
    reference it) and because it is the right shape when the caller genuinely
    wants to pick a model at runtime. New callers should prefer the per-model
    tools in :mod:`handlers.image_tools`, which can be priced individually --
    Imperal prices a tool, not a parameter value, so this one tool cannot
    charge Pro rates for Pro and Lite rates for Lite.

    The work itself is delegated so both entry points share one code path.
    """
    return await run_image_generation(
        ctx,
        prompt=params.prompt,
        model=params.model,
        image_size=params.image_size,
        reference_generation_ids=params.reference_generation_ids,
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

