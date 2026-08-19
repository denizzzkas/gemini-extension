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

``count`` (1-4): why parallel, not sequential, and why the cost warning
lives in the FORM, not just the tool description
------------------------------------------------------------------------
Imperal bills this TOOL once per call (see the module docstring above) --
but each call still burns the user's OWN Gemini API key once per image
generated (see ``_get_api_key`` / handlers/media.py), so ``count=4`` is 4x
the real Google-side cost even though it is one Imperal-priced call. That
is a real cost the user pays outside Imperal's own metering, which is
exactly why the panel form must show a warning BEFORE submit, not just
after the bill already happened -- see handlers/panel_forms.py.

Generations run with ``asyncio.gather`` (concurrently), not a for-loop:
each call already costs up to ``REQUEST_TIMEOUT_IMAGE`` (60s) round-tripped
to Gemini, and 4 of those run sequentially would risk the ~180s federal
per-call timeout budget (see REQUEST_TIMEOUT_VIDEO's own comment in
gemini_config.py for that same ceiling). Concurrently, the whole batch costs
about as long as the SLOWEST single generation, not the sum of all of them.

Partial success, deliberately: if 2 of 4 succeed and 2 fail (e.g. a
transient 503), the 2 successes are still returned rather than the whole
batch being discarded -- the user already paid Google for those, discarding
them would be losing real generations for no reason. Only when EVERY
generation fails does this return ActionResult.error.
"""
from __future__ import annotations

import asyncio
import logging

from imperal_sdk import ActionResult

from clients.gemini_client import GeminiAPIError, create_interaction
from core.preview import PROVEN_GOOD_CHARS
from gemini_config import (
    IMAGE_ASPECT_RATIO_CHOICES, IMAGE_MODEL_CHOICES, IMAGE_SIZE_CHOICES,
    MAX_IMAGE_COUNT, REQUEST_TIMEOUT_IMAGE,
)
from handlers.media import (
    _attach_preview, _get_api_key, _log_generation, _resolve_reference_images,
    _save_media,
)
from handlers.media_link import mint_media_link
from return_models import GeneratedImageRecord

log = logging.getLogger("gemini.image_core")


async def _generate_single(
    ctx, *, prompt: str, model: str, image_size: str, aspect_ratio: str,
    api_key: str, reference_images: list[dict],
    reference_generation_ids: list[str] | None,
) -> tuple[GeneratedImageRecord | None, str | None]:
    """Run exactly one Gemini call and persist it. Returns (record, error).

    Exactly one of the two return slots is populated -- never both, never
    neither -- so a caller running several of these concurrently can tell
    success from failure per-item without exceptions crossing the gather.
    """
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
            response_format={
                "type": "image", "image_size": image_size,
                "aspect_ratio": aspect_ratio,
            },
            timeout=REQUEST_TIMEOUT_IMAGE,
        )
    except GeminiAPIError as e:
        log.error("image generation failed on %s: %s", model, e)
        return None, f"Image generation failed: {e.message}"

    image = next((m for m in result.media if m.kind == "image"), None)
    if image is None:
        return None, "Gemini did not return an image for this prompt."

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
    preview = await _attach_preview(ctx, generation_id, image.data_b64, mime_type)

    # Same honest constraint as the panel (see core/preview's own docstring):
    # a reply carrying the image as base64 has a measured, undocumented
    # ceiling. Below PROVEN_GOOD_CHARS the untouched original is safe to send
    # straight into the chat reply. Above it, the ORIGINAL bytes are never
    # put in the reply at all -- only the already-built preview (or nothing,
    # if even the preview ladder couldn't fit) -- plus a signed link to the
    # real file, so the user is never left with neither a usable inline image
    # nor a way to get the full one.
    out_b64 = image.data_b64
    out_mime = mime_type
    is_preview = False
    if len(image.data_b64) > PROVEN_GOOD_CHARS:
        is_preview = True
        if preview is not None:
            out_b64, out_mime = preview
        else:
            out_b64 = ""  # nothing fits inline; the link below is the only way to see it

    full_image_url = await mint_media_link(ctx, generation_id, storage_path) if is_preview else ""

    record = GeneratedImageRecord(
        generation_id=generation_id,
        prompt=prompt,
        model=model,
        mime_type=out_mime,
        image_base64=out_b64,
        is_preview=is_preview,
        full_image_url=full_image_url,
        url=url,
        text=result.text,
    )
    return record, None


def _summary_for(record: GeneratedImageRecord, label: str, prompt: str) -> str:
    if record.is_preview and record.image_base64:
        return (
            f"Generated an image with {label} for: \"{prompt}\". The original "
            "was too large to send in full here, so this is a COMPRESSED "
            "PREVIEW (image_base64/mime_type) -- show it inline in chat "
            "(don't just paste the raw url as text -- render it as an "
            "image), and clearly tell the user it's a shrunk preview and "
            "that the full, original-quality image is available at "
            "full_image_url (also saved in the Gemini Studio panel history)."
        )
    if record.is_preview:
        return (
            f"Generated an image with {label} for: \"{prompt}\". It could not "
            "be shown inline here at all (too large even shrunk) -- tell the "
            "user the full image is available at full_image_url, and that "
            "it's also saved in the Gemini Studio panel history."
        )
    return (
        f"Generated an image with {label} for: \"{prompt}\". "
        "Show it inline in chat using the returned image_base64/mime_type "
        "(don't just paste the raw url as text -- render it as an image), "
        "and mention it's also saved in the Gemini Studio panel history."
    )


async def run_image_generation(
    ctx,
    *,
    prompt: str,
    model: str,
    image_size: str,
    aspect_ratio: str = "1:1",
    reference_generation_ids: list[str] | None = None,
    count: int = 1,
) -> ActionResult:
    """Generate 1-4 images from the SAME prompt and persist each.

    Kept free of any decorator so each per-model tool can wrap it, and so it
    stays directly unit-testable. ``count`` defaults to 1 and, at that
    default, behaves EXACTLY as a single generation always did (one record,
    no ``images`` field, same summary wording) -- see the module docstring
    for why count>1 fans out concurrently instead of looping.
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
    if aspect_ratio not in IMAGE_ASPECT_RATIO_CHOICES:
        return ActionResult.error(
            f"Unknown aspect_ratio {aspect_ratio!r}. Valid options: "
            f"{', '.join(IMAGE_ASPECT_RATIO_CHOICES)}.",
            retryable=False,
        )
    if not (1 <= count <= MAX_IMAGE_COUNT):
        return ActionResult.error(
            f"count must be between 1 and {MAX_IMAGE_COUNT} (got {count}).",
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

    outcomes = await asyncio.gather(*(
        _generate_single(
            ctx, prompt=prompt, model=model, image_size=image_size,
            aspect_ratio=aspect_ratio, api_key=api_key,
            reference_images=reference_images,
            reference_generation_ids=reference_generation_ids,
        )
        for _ in range(count)
    ))

    records = [r for r, _ in outcomes if r is not None]
    errors = [e for _, e in outcomes if e is not None]

    if not records:
        # Every single generation in the batch failed -- nothing to show,
        # so this is a real error, not a partial-success summary.
        first_error = errors[0] if errors else "Image generation failed."
        return ActionResult.error(
            first_error if count == 1 else
            f"All {count} generations failed. First error: {first_error}",
            retryable=True,
        )

    label = IMAGE_MODEL_CHOICES[model]["label"]
    first, rest = records[0], records[1:]
    first.images = rest

    summary = _summary_for(first, label, prompt)
    if count > 1:
        ok, failed = len(records), len(errors)
        batch_note = (
            f" Requested {count} images with {label}: {ok} succeeded"
            + (f", {failed} failed" if failed else "")
            + f". Show ALL {ok} images inline (this result plus every entry "
              "in its images[] list), not just the first one."
        )
        summary = summary + batch_note

    return ActionResult.success(data=first, summary=summary)
