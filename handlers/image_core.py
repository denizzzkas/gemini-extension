"""Shared image-generation core behind the per-model tools.

Why per-model tools instead of one tool with a ``model=`` parameter
------------------------------------------------------------------
Imperal prices a TOOL, not a parameter value (per-tool prices are set via
``save_pricing``). The four Nano Banana models differ several-fold in cost
per image, so a single ``generate_image`` tool could only ever carry one
price -- either overcharging for Lite or undercharging for Pro.

Splitting into one tool per model makes the price match what the call
actually costs, and lets the user say "generate with Nano Banana 2" and hit
a distinctly-priced tool.

All four share this module so the split costs nothing in duplicated logic:
they differ only in which model id they pass.
"""
from __future__ import annotations

import logging

from imperal_sdk import ActionResult

from clients.gemini_client import GeminiAPIError, create_interaction
from gemini_config import (
    IMAGE_MODEL_CHOICES, IMAGE_SIZE_CHOICES, REQUEST_TIMEOUT_IMAGE,
)
from handlers.media import (
    _attach_preview, _get_api_key, _log_generation, _resolve_reference_images,
    _save_media,
)
from return_models import GeneratedImageRecord

log = logging.getLogger("gemini.image_core")


async def run_image_generation(
    ctx,
    *,
    prompt: str,
    model: str,
    image_size: str,
    reference_generation_ids: list[str] | None = None,
) -> ActionResult:
    """Generate one image and persist it, returning a ready ActionResult.

    Kept free of any decorator so each per-model tool can wrap it, and so it
    stays directly unit-testable.
    """
    if model not in IMAGE_MODEL_CHOICES:
        return ActionResult.error(
            f"Unknown image model {model!r}. Valid options: "
            f"{', '.join(IMAGE_MODEL_CHOICES)}.",
            retryable=False,
        )

    if image_size not in IMAGE_SIZE_CHOICES:
        return ActionResult.error(
            f"Unknown image_size {image_size!r}. Valid options: "
            f"{', '.join(IMAGE_SIZE_CHOICES)}.",
            retryable=False,
        )

    api_key = await _get_api_key(ctx)
    if not api_key:
        return ActionResult.error(
            "No Gemini API key configured. Add your key from Google AI Studio "
            "(aistudio.google.com/apikey) in the extension's Secrets panel."
        )

    reference_images = []
    if reference_generation_ids:
        reference_images = await _resolve_reference_images(
            ctx, reference_generation_ids,
        )
        if not reference_images:
            return ActionResult.error(
                "None of the given reference_generation_ids could be resolved "
                "(not found, not owned by you, not an image, or predates this "
                "feature). Use an ID returned by a prior generation or select "
                "a saved image in Gemini Studio.",
                retryable=False,
            )

    try:
        result = await create_interaction(
            ctx, api_key, model, prompt,
            reference_images=reference_images or None,
            # ONLY documented response_format keys. Per
            # ai.google.dev/gemini-api/docs/image-generation the object is
            # {"type": "image", "aspect_ratio": ..., "image_size": ...} --
            # there is no mime_type field, and sending one made the API
            # reject every request (that outage is why this comment exists).
            # The output format is the model's choice; it returns JPEG.
            response_format={"type": "image", "image_size": image_size},
            timeout=REQUEST_TIMEOUT_IMAGE,
        )
    except GeminiAPIError as e:
        log.error("image generation failed on %s: %s", model, e)
        return ActionResult.error(
            f"Image generation failed: {e.message}",
            retryable=e.status_code in (429, 500, 502, 503, 504),
        )

    image = next((m for m in result.media if m.kind == "image"), None)
    if image is None:
        return ActionResult.error(
            "Gemini did not return an image for this prompt. Try rephrasing it.",
            retryable=True,
        )

    # Trust what actually arrived rather than a default: the models return
    # JPEG, and the preview path picks its decoder from these bytes.
    mime_type = image.mime_type or "image/jpeg"
    storage_path, url = await _save_media(ctx, "image", mime_type, image.data_b64)
    generation_id = await _log_generation(
        ctx, "image", prompt, model,
        url=url, storage_path=storage_path, mime_type=mime_type,
        reference_ids=list(reference_generation_ids or []),
    )

    # Build the panel preview NOW, while the bytes are already in memory:
    # otherwise whoever opens it first pays a storage download plus ~0.4-1.3s
    # of pure-Python shrinking (no Pillow in production). See core/preview.py.
    await _attach_preview(ctx, generation_id, image.data_b64, mime_type)

    label = IMAGE_MODEL_CHOICES[model]["label"]
    record = GeneratedImageRecord(
        generation_id=generation_id,
        prompt=prompt,
        model=model,
        mime_type=mime_type,
        image_base64=image.data_b64,
        url=url,
        text=result.text,
    )
    return ActionResult.success(
        data=record,
        summary=(
            f"Generated an image with {label} for: \"{prompt}\". "
            "Show it inline in chat using the returned image_base64/mime_type "
            "(don't just paste the raw url as text -- render it as an image), "
            "and mention it's also saved in the Gemini Studio panel history."
        ),
    )
