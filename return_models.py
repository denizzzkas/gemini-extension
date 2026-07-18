"""Typed Pydantic return models for @chat.function data_model=..."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedImageRecord(BaseModel):
    """Result of a single image generation call."""

    prompt: str = Field(..., description="The prompt used to generate the image")
    model: str = Field(..., description="Gemini model id used")
    mime_type: str = Field("", description="MIME type of the generated image, e.g. image/png")
    image_base64: str = Field("", description="Base64-encoded image bytes")
    url: str = Field("", description="Storage URL of the saved image, if it could be persisted")
    text: str = Field("", description="Any accompanying text the model returned")


class GeneratedVideoRecord(BaseModel):
    """Result of a single video generation call."""

    prompt: str = Field(..., description="The prompt used to generate the video")
    model: str = Field(..., description="Gemini model id used")
    mime_type: str = Field("", description="MIME type of the generated video, e.g. video/mp4")
    video_base64: str = Field("", description="Base64-encoded video bytes")
    url: str = Field("", description="Storage URL of the saved video, if it could be persisted")
    text: str = Field("", description="Any accompanying text the model returned")


class GeminiConnectionRecord(BaseModel):
    """Whether the Gemini API key is configured and reachable."""

    configured: bool = Field(..., description="Whether an API key is set")
    api_reachable: bool = Field(..., description="Whether the Gemini API responded to a bounded probe")


class GenerationHistoryItem(BaseModel):
    kind: str = Field(..., description="'image' or 'video'")
    prompt: str
    model: str
    url: str = ""
    created_at: str = ""


class GenerationHistoryRecord(BaseModel):
    items: list[GenerationHistoryItem] = Field(default_factory=list)
    count: int = 0
