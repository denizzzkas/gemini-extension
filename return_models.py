"""Typed Pydantic return models for @chat.function data_model=..."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedImageRecord(BaseModel):
    """Result of a single image generation call."""

    generation_id: str = Field("", description="This generation's log ID -- pass it in reference_generation_ids to a follow-up dedicated image-model tool to reuse this exact image (e.g. same character/scene)")
    prompt: str = Field(..., description="The prompt used to generate the image")
    model: str = Field(..., description="Gemini model id used")
    mime_type: str = Field("", description="MIME type of the generated image, e.g. image/png")
    image_base64: str = Field("", description="Base64-encoded image bytes. THIS IS A COMPRESSED PREVIEW whenever is_preview=true (shrunk so it reliably fits in the chat reply) -- never the untouched original in that case. Show it inline in chat as the image.")
    is_preview: bool = Field(False, description="True when image_base64 is a shrunk preview, not the untouched original (the original was too large to safely inline in a chat reply). When true, ALWAYS tell the user this is a compressed preview and that the full original is available at full_image_url.")
    full_image_url: str = Field("", description="A signed, time-limited link (expires in a few minutes) that opens the full, original-quality image in a browser tab. Only present when is_preview=true and a link could be minted. Share this link with the user as 'the full image' whenever image_base64 is a preview.")
    url: str = Field("", description="INTERNAL storage reference for the saved image. NOT a publicly viewable link -- extension storage is only readable via the authenticated gateway, so opening this in a browser returns 404. Never present it to the user as a clickable link to view the image; show the image via image_base64 or the Gemini Studio panel instead")
    text: str = Field("", description="Any accompanying text the model returned")


class GeneratedVideoRecord(BaseModel):
    """Result of a single video generation call."""

    generation_id: str = Field("", description="This generation's log ID")
    prompt: str = Field(..., description="The prompt used to generate the video")
    model: str = Field(..., description="Gemini model id used")
    mime_type: str = Field("", description="MIME type of the generated video, e.g. video/mp4")
    video_base64: str = Field("", description="Base64-encoded video bytes. THIS IS A COMPRESSED/SMALLER COPY whenever is_preview=true (shrunk so it reliably fits in the chat reply) -- never the untouched original in that case. Show it inline in chat as the video.")
    is_preview: bool = Field(False, description="True when video_base64 is not the untouched original (the original was too large to safely inline in a chat reply). When true, ALWAYS tell the user this is a compressed preview and that the full original is available at full_video_url.")
    full_video_url: str = Field("", description="A signed, time-limited link (expires in a few minutes) that opens the full, original-quality video in a browser tab. Only present when is_preview=true and a link could be minted. Share this link with the user as 'the full video' whenever video_base64 is a preview.")
    url: str = Field("", description="INTERNAL storage reference for the saved video. NOT a publicly viewable link -- extension storage is only readable via the authenticated gateway, so opening this in a browser returns 404. Never present it to the user as a clickable link")
    text: str = Field("", description="Any accompanying text the model returned")


class GeminiConnectionRecord(BaseModel):
    """Whether the Gemini API key is configured and reachable."""

    configured: bool = Field(..., description="Whether an API key is set")
    api_reachable: bool = Field(..., description="Whether the Gemini API responded to a bounded probe")


class UploadedReferenceRecord(BaseModel):
    """One image accepted into the user's reusable Gemini reference library."""

    generation_id: str = Field(..., description="Reference ID to pass in reference_generation_ids")
    filename: str = Field("", description="Original upload filename when supplied")
    mime_type: str = Field(..., description="Detected PNG or JPEG MIME type")
    bytes: int = Field(..., description="Original file size in bytes")


class UploadReferenceResult(BaseModel):
    """Outcome of storing one or more uploaded reference images."""

    stored: list[UploadedReferenceRecord] = Field(default_factory=list)
    generation_ids: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class SavedSecretResult(BaseModel):
    """Outcome of saving the user's own Gemini API key from the left panel.

    Never carries the secret value itself (I-SECRETS-NEVER-LOGGED) -- only
    whether it is now configured.
    """

    configured: bool = Field(..., description="Whether a key is now set for this user")
