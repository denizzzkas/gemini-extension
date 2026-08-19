"""One image-generation tool per model, so each can carry its own price.

Imperal prices a TOOL, not a parameter value, and the four Nano Banana models
differ several-fold in cost per image. A single tool with ``model=`` could
therefore only ever carry one price -- overcharging for Lite or undercharging
for Pro. These four tools exist so the price can match the real cost, and so
"generate with Nano Banana 2" maps to a distinctly-priced call.

Each tool is a thin wrapper: the model id is fixed and the shared work lives
in :mod:`handlers.image_core`, so the split adds no duplicated logic.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import chat
from gemini_config import (
    DEFAULT_IMAGE_ASPECT_RATIO, DEFAULT_IMAGE_SIZE, IMAGE_ASPECT_RATIO_CHOICES,
    IMAGE_MODEL_CHOICES, IMAGE_SIZE_CHOICES, MAX_IMAGE_COUNT, MAX_PROMPT_LEN,
    MODEL_IMAGE, MODEL_IMAGE_FLASH, MODEL_IMAGE_FLASH_LITE, MODEL_IMAGE_LEGACY,
)
from handlers.image_core import run_image_generation
from handlers.media import MAX_REFERENCE_IMAGES
from prompt_guide import image_prompt_guidance_text
from return_models import GeneratedImageRecord

log = logging.getLogger("gemini.image_tools")

_SIZE_TEXT = "; ".join(f"{k} ({v})" for k, v in IMAGE_SIZE_CHOICES.items())
_ASPECT_TEXT = "; ".join(f"{k} ({v})" for k, v in IMAGE_ASPECT_RATIO_CHOICES.items())


class ModelImageParams(BaseModel):
    """Params for a per-model tool: no ``model`` field, the tool IS the model."""

    prompt: str = Field(
        ...,
        description=(
            "Fully-specified description of the image to generate or edit -- "
            "expand short/vague user requests into a Google-recommended "
            "structured prompt (subject, setting, light, camera/lens for "
            "photorealistic shots; style+medium for illustrations; explicit "
            "on-image text + font/style for text-in-image) before passing it "
            "here. See the tool description for the full template set."
        ),
        min_length=1, max_length=MAX_PROMPT_LEN,
    )
    image_size: str = Field(
        DEFAULT_IMAGE_SIZE,
        description=(
            "Output resolution, default 1K. Higher sizes cost more and are "
            "shown in the panel as a preview rather than at full size. "
            "Options: " + _SIZE_TEXT
        ),
    )
    aspect_ratio: str = Field(
        DEFAULT_IMAGE_ASPECT_RATIO,
        description=(
            "Output aspect ratio (shape), default 1:1 (square). Pick one "
            "that matches what the user describes -- e.g. a phone wallpaper "
            "or story graphic is 9:16, a wide banner or thumbnail is 16:9. "
            "Options: " + _ASPECT_TEXT
        ),
    )
    reference_generation_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_REFERENCE_IMAGES,
        description=(
            "Optional: IDs of this user's OWN past generations (from "
            "a prior generation result) to reuse the same character or setting. "
            "Only this extension's saved generations work as references."
        ),
    )
    count: int = Field(
        1, ge=1, le=MAX_IMAGE_COUNT,
        description=(
            f"How many images to generate from this SAME prompt in one call, "
            f"1-{MAX_IMAGE_COUNT} (default 1). COST WARNING: each extra image "
            "is a FULL EXTRA Gemini generation billed to the user's own "
            "Gemini API key -- count=4 costs roughly 4x a single generation. "
            "Only raise this above 1 when the user explicitly asks for "
            "multiple images/variations, and tell them up front it multiplies "
            "the cost."
        ),
    )


def _describe(model_id: str, extra: str) -> str:
    info = IMAGE_MODEL_CHOICES[model_id]
    return (
        f"Generate or edit an image with {info['label']} ({model_id}). "
        f"{info['description']} {extra} "
        "Pass reference_generation_ids to reuse a character/setting from this "
        "user's own past generations. " + image_prompt_guidance_text()
    )


@chat.function(
    "generate_image_nano_banana_pro",
    action_type="write",
    chain_callable=True,
    effects=["create:media"],
    event="gemini.image_generated",
    data_model=GeneratedImageRecord,
    description=_describe(
        MODEL_IMAGE,
        "Use this when quality matters most -- complex scenes, accurate text "
        "in the image, brand consistency. It is the most expensive of the "
        "image tools, so prefer a cheaper one for drafts and iterations.",
    ),
)
async def fn_generate_image_pro(ctx, params: ModelImageParams) -> ActionResult:
    """Nano Banana Pro — premium quality, priced separately."""
    return await run_image_generation(
        ctx, prompt=params.prompt, model=MODEL_IMAGE,
        image_size=params.image_size, aspect_ratio=params.aspect_ratio,
        reference_generation_ids=params.reference_generation_ids,
        count=params.count,
    )


@chat.function(
    "generate_image_nano_banana_2",
    action_type="write",
    chain_callable=True,
    effects=["create:media"],
    event="gemini.image_generated",
    data_model=GeneratedImageRecord,
    description=_describe(
        MODEL_IMAGE_FLASH,
        "The balanced default for most requests: clearly cheaper than Pro "
        "while still strong on quality and reference consistency.",
    ),
)
async def fn_generate_image_flash(ctx, params: ModelImageParams) -> ActionResult:
    """Nano Banana 2 — balanced quality/cost, priced separately."""
    return await run_image_generation(
        ctx, prompt=params.prompt, model=MODEL_IMAGE_FLASH,
        image_size=params.image_size, aspect_ratio=params.aspect_ratio,
        reference_generation_ids=params.reference_generation_ids,
        count=params.count,
    )


@chat.function(
    "generate_image_nano_banana_2_lite",
    action_type="write",
    chain_callable=True,
    effects=["create:media"],
    event="gemini.image_generated",
    data_model=GeneratedImageRecord,
    description=_describe(
        MODEL_IMAGE_FLASH_LITE,
        "The cheapest and fastest option -- use it for quick drafts, bulk "
        "generation, or when the user says the result need not be perfect. "
        "Not suited to multiple reference images or step-by-step editing.",
    ),
)
async def fn_generate_image_flash_lite(ctx, params: ModelImageParams) -> ActionResult:
    """Nano Banana 2 Lite — cheapest/fastest, priced separately."""
    return await run_image_generation(
        ctx, prompt=params.prompt, model=MODEL_IMAGE_FLASH_LITE,
        image_size=params.image_size, aspect_ratio=params.aspect_ratio,
        reference_generation_ids=params.reference_generation_ids,
        count=params.count,
    )


@chat.function(
    "generate_image_nano_banana_legacy",
    action_type="write",
    chain_callable=True,
    effects=["create:media"],
    event="gemini.image_generated",
    data_model=GeneratedImageRecord,
    description=_describe(
        MODEL_IMAGE_LEGACY,
        "Legacy 1024px model, kept for compatibility. Google recommends Nano "
        "Banana 2 Lite instead for new work -- prefer that unless the user "
        "explicitly asks for this one.",
    ),
)
async def fn_generate_image_legacy(ctx, params: ModelImageParams) -> ActionResult:
    """Nano Banana (legacy) — kept for compatibility, priced separately."""
    return await run_image_generation(
        ctx, prompt=params.prompt, model=MODEL_IMAGE_LEGACY,
        image_size=params.image_size, aspect_ratio=params.aspect_ratio,
        reference_generation_ids=params.reference_generation_ids,
        count=params.count,
    )
