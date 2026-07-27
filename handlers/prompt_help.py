"""Serve the Google prompting guide as DATA, not as tool-description prose.

Why this exists
---------------
:mod:`prompt_guide` already holds Google's official templates, transcribed
from their docs. But it was injected in exactly one place: the ``description=``
of the generation tools. That has a blind spot the user hit directly.

When the user asks "give me a prompt" WITHOUT asking to generate, no
generation tool is invoked, and a tool description is not reference material
the assistant can consult on demand -- long descriptions are also abbreviated
before they reach the model (the image tool's own description arrives ending in
"See tool description for the full template set", with the template set
itself elided). So the assistant was writing prompts from memory while the
authoritative templates sat unread in the repo.

A read-only function fixes that: the templates come back as structured DATA
that can be fetched at will, cost nothing when unused, and cannot be truncated
away the way a decorator string can. It is deliberately ``action_type="read"``
with no side effects -- consulting the docs must never look like generating.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import chat
from prompt_guide import (
    IMAGE_BEST_PRACTICES,
    IMAGE_EDITING_TEMPLATES,
    IMAGE_GENERATION_TEMPLATES,
    VIDEO_AUDIO_TIPS,
    VIDEO_EXTRA_TIPS,
    VIDEO_PROMPT_ELEMENTS,
)

log = logging.getLogger("gemini.prompt_help")

# The docs these templates were transcribed from, kept next to the data so a
# future reader can re-verify rather than trust a commit message.
IMAGE_DOC_URL = "https://ai.google.dev/gemini-api/docs/image-generation"
VIDEO_DOC_URL = "https://ai.google.dev/gemini-api/docs/veo"


class GetPromptGuideParams(BaseModel):
    kind: str = Field(
        "image",
        description=(
            "Which guide to return: 'image' for still-image generation and "
            "editing, 'video' for Veo video, or 'all' for both."
        ),
    )


class PromptGuideRecord(BaseModel):
    kind: str = ""
    source_urls: list[str] = Field(default_factory=list)
    generation_templates: dict[str, str] = Field(default_factory=dict)
    editing_templates: dict[str, str] = Field(default_factory=dict)
    best_practices: list[str] = Field(default_factory=list)
    video_elements: dict[str, str] = Field(default_factory=dict)
    video_audio_tips: list[str] = Field(default_factory=list)
    video_extra_tips: list[str] = Field(default_factory=list)


@chat.function(
    "get_prompt_guide",
    action_type="read",
    chain_callable=True,
    data_model=PromptGuideRecord,
    description=(
        "Fetch Google's official prompting guide for Gemini image/video "
        "generation -- the structured templates, editing patterns and best "
        "practices transcribed from Google's own developer documentation. "
        "ALWAYS call this FIRST whenever the user asks you to WRITE, DRAFT, "
        "IMPROVE or REVIEW a generation prompt, even when they do not want "
        "anything generated yet: it is the authoritative source, so a prompt "
        "written from memory instead is a guess. Read-only and free of side "
        "effects -- it never generates an image, so calling it to check the "
        "docs is always safe. Use kind='image' for stills (default), "
        "kind='video' for Veo, kind='all' for both."
    ),
)
async def fn_get_prompt_guide(ctx, params: GetPromptGuideParams) -> ActionResult:
    """Return the official prompt templates as data the caller can apply."""
    kind = (params.kind or "image").strip().lower()
    if kind not in ("image", "video", "all"):
        return ActionResult.error(
            f"Unknown kind {params.kind!r}. Valid options: image, video, all.",
            retryable=False,
        )

    record = PromptGuideRecord(kind=kind)

    if kind in ("image", "all"):
        record.source_urls.append(IMAGE_DOC_URL)
        record.generation_templates = dict(IMAGE_GENERATION_TEMPLATES)
        record.editing_templates = dict(IMAGE_EDITING_TEMPLATES)
        record.best_practices = list(IMAGE_BEST_PRACTICES)

    if kind in ("video", "all"):
        record.source_urls.append(VIDEO_DOC_URL)
        record.video_elements = dict(VIDEO_PROMPT_ELEMENTS)
        record.video_audio_tips = list(VIDEO_AUDIO_TIPS)
        record.video_extra_tips = list(VIDEO_EXTRA_TIPS)

    # The summary is where the behavioural instruction has to live: returning
    # templates is useless if the caller then answers from memory anyway.
    if kind == "video":
        pick = (
            f"Pick the relevant elements from video_elements ({len(record.video_elements)} "
            "available) and add audio cues where they matter."
        )
    else:
        pick = (
            f"Choose the ONE generation template that fits "
            f"({', '.join(record.generation_templates)}) -- or an editing "
            f"template ({', '.join(record.editing_templates)}) when the user "
            "is modifying an existing image -- then fill every [bracketed] "
            "slot with concrete detail."
        )

    return ActionResult.success(
        data=record,
        summary=(
            "Google's official prompting guide. Now write the user's prompt by "
            f"APPLYING it, not from memory. {pick} Then apply the best "
            "practices: be hyper-specific over vague, state intent, use "
            "positive phrasing rather than negation, and control the camera "
            "with photographic language. Give the user the finished prompt; "
            "do NOT generate anything unless they explicitly asked for it."
        ),
    )
