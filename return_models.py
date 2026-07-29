"""Typed Pydantic return models for @chat.function data_model=..."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedImageRecord(BaseModel):
    """Result of a single image generation call."""

    generation_id: str = Field("", description="This generation's log ID -- pass it as a reference_generation_ids entry in a follow-up generate_image call to reuse this exact image (e.g. same character/scene)")
    prompt: str = Field(..., description="The prompt used to generate the image")
    model: str = Field(..., description="Gemini model id used")
    mime_type: str = Field("", description="MIME type of the generated image, e.g. image/png")
    image_base64: str = Field("", description="Base64-encoded image bytes")
    url: str = Field("", description="INTERNAL storage reference for the saved image. NOT a publicly viewable link -- extension storage is only readable via the authenticated gateway, so opening this in a browser returns 404. Never present it to the user as a clickable link to view the image; show the image via image_base64 or the Gemini Studio panel instead")
    text: str = Field("", description="Any accompanying text the model returned")


class GeneratedVideoRecord(BaseModel):
    """Result of a single video generation call."""

    generation_id: str = Field("", description="This generation's log ID")
    prompt: str = Field(..., description="The prompt used to generate the video")
    model: str = Field(..., description="Gemini model id used")
    mime_type: str = Field("", description="MIME type of the generated video, e.g. video/mp4")
    video_base64: str = Field("", description="Base64-encoded video bytes")
    url: str = Field("", description="INTERNAL storage reference for the saved video. NOT a publicly viewable link -- extension storage is only readable via the authenticated gateway, so opening this in a browser returns 404. Never present it to the user as a clickable link")
    text: str = Field("", description="Any accompanying text the model returned")


class OriginalMediaRecord(BaseModel):
    """The untouched original bytes for one past generation, for chat display.

    Distinct from GeneratedImageRecord/GeneratedVideoRecord: those are the
    result of just now MAKING something, this is fetching something already
    made, by id, when the panel could only show a shrunk preview (or nothing,
    for a payload too big to inline).

    Field names deliberately MIRROR GeneratedImageRecord/GeneratedVideoRecord
    (``image_base64``/``video_base64``, not a generic ``media_base64``): the
    LLM renders inline media by being told, in a tool's own success summary,
    to use a specific field name -- there is no platform-level convention that
    renders ANY base64 field automatically. Every other tool in this app that
    reliably renders an image in chat uses ``image_base64``; a differently
    named field here is exactly why "View full resolution in chat" used to
    print the raw generation id instead of showing the picture.
    """

    generation_id: str = Field(..., description="The generation ID this original belongs to")
    kind: str = Field(..., description="'image' or 'video'")
    prompt: str = Field("", description="The prompt this generation was made with")
    model: str = Field("", description="Gemini model id used")
    mime_type: str = Field("", description="MIME type of the original bytes")
    image_base64: str = Field("", description="Base64-encoded ORIGINAL image bytes -- untouched, not a shrunk preview. Populated when kind=='image'; render it inline exactly like generate_image's image_base64")
    video_base64: str = Field("", description="Base64-encoded ORIGINAL video bytes -- untouched. Populated when kind=='video'; render it exactly like generate_video's video_base64")


class GeminiConnectionRecord(BaseModel):
    """Whether the Gemini API key is configured and reachable."""

    configured: bool = Field(..., description="Whether an API key is set")
    api_reachable: bool = Field(..., description="Whether the Gemini API responded to a bounded probe")


class GenerationHistoryItem(BaseModel):
    id: str = Field("", description="Generation log ID -- pass as reference_generation_ids in generate_image to reuse this image as a reference")
    kind: str = Field(..., description="'image' or 'video'")
    prompt: str = Field("", description="The FULL prompt this generation was made with")
    model: str
    # A plain '# comment' here was invisible to the model consuming this schema,
    # so the dead link kept being offered to the user as something clickable.
    # The warning has to live in the Field description to actually be seen.
    url: str = Field("", description=(
        "INTERNAL storage reference only -- NOT a publicly viewable link. "
        "Extension storage is served solely from the authenticated gateway "
        "endpoint, so this path returns HTTP 404 in a browser (verified). "
        "NEVER present it to the user as a link to view the result: tell them "
        "to open the Gemini Studio panel and click View image on the entry."
    ))
    created_at: str = ""


class GenerationHistoryRecord(BaseModel):
    items: list[GenerationHistoryItem] = Field(default_factory=list)
    count: int = 0
