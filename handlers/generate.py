"""Gemini video generation handler.

Image generation used to also live here as one generic ``generate_image``
tool with a ``model=`` parameter. It was removed: Imperal prices a TOOL, not
a parameter value, and this repo already has four distinctly-priced per-model
tools (handlers/image_tools.py -- Pro/Flash/Flash Lite/Legacy) that cover
every model this one used to. Keeping both meant the SAME generation could be
billed at two different rates depending on which tool the caller picked, and
the panel's own form only ever called one of them anyway (see
handlers/panel_forms.py) -- the generic tool had become dead weight with a
built-in pricing footgun, not a real alternative path.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import chat
from core.preview import PROVEN_GOOD_CHARS
from gemini_config import (
    DEFAULT_VIDEO_ASPECT_RATIO, MAX_PROMPT_LEN, MODEL_VIDEO,
    REQUEST_TIMEOUT_VIDEO, VIDEO_ASPECT_RATIO_CHOICES,
)
from clients.gemini_client import create_interaction, GeminiAPIError
from prompt_guide import video_prompt_guidance_text
from return_models import GeneratedVideoRecord
from handlers.media import _get_api_key, _log_generation, _save_media
from handlers.media_link import mint_media_link

log = logging.getLogger("gemini.generate")


_VIDEO_ASPECT_TEXT = "; ".join(f"{k} ({v})" for k, v in VIDEO_ASPECT_RATIO_CHOICES.items())


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
    aspect_ratio: str = Field(
        DEFAULT_VIDEO_ASPECT_RATIO,
        description=(
            "Output aspect ratio, default 16:9 (landscape). Use 9:16 for a "
            "phone/portrait/story-style video. Options: " + _VIDEO_ASPECT_TEXT
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
    if params.aspect_ratio not in VIDEO_ASPECT_RATIO_CHOICES:
        return ActionResult.error(
            f"Unknown aspect_ratio {params.aspect_ratio!r}. Valid options: "
            f"{', '.join(VIDEO_ASPECT_RATIO_CHOICES)}.",
            retryable=False,
        )

    api_key = await _get_api_key(ctx)
    if not api_key:
        return ActionResult.error(
            "No Gemini API key configured. Add your key from Google AI Studio "
            "(aistudio.google.com/apikey) in the extension's Secrets panel."
        )

    try:
        result = await create_interaction(
            ctx, api_key, MODEL_VIDEO, params.prompt,
            # Documented response_format key for Gemini Omni Flash
            # (ai.google.dev/gemini-api/docs/omni, "Control aspect ratio",
            # verified 2026-08) -- only 16:9/9:16 are valid for this model,
            # unlike the ten-value set the image models accept.
            response_format={"type": "video", "aspect_ratio": params.aspect_ratio},
            timeout=REQUEST_TIMEOUT_VIDEO,
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

    # Same honest constraint as image generation (see core/preview and
    # handlers/image_core.py): a reply carrying media as base64 has a
    # measured, undocumented ceiling. There is no video-shrinking codec here
    # (core/preview.py only handles PNG/JPEG stills), so above the ceiling
    # the ORIGINAL bytes are simply never put in the reply -- only a signed
    # link to the real file -- rather than risking a reply that silently
    # fails to send.
    out_b64 = video.data_b64
    is_preview = len(video.data_b64) > PROVEN_GOOD_CHARS
    if is_preview:
        out_b64 = ""  # no video preview codec exists yet; the link is the only way to see it

    full_video_url = await mint_media_link(ctx, generation_id, storage_path) if is_preview else ""

    record = GeneratedVideoRecord(
        generation_id=generation_id,
        prompt=params.prompt,
        model=MODEL_VIDEO,
        mime_type=mime_type,
        video_base64=out_b64,
        is_preview=is_preview,
        full_video_url=full_video_url,
        url=url,
        text=result.text,
    )
    if is_preview:
        summary = (
            f"Generated a video for: \"{params.prompt}\". It's too large to "
            "send inline in chat -- tell the user the full video is "
            "available at full_video_url, and that it's also saved in the "
            "Gemini Studio panel history."
        )
    else:
        summary = (
            f"Generated a video for: \"{params.prompt}\". "
            "Show it inline in chat using the returned video_base64/mime_type "
            "(don't just paste the raw url as text -- render it as a video), "
            "and mention it's also saved in the Gemini Studio panel history."
        )
    return ActionResult.success(data=record, summary=summary)
