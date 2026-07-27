"""Tests for get_prompt_guide -- the docs served as fetchable data.

The bug this guards: Google's templates lived ONLY inside the generation
tools' ``description=``. Asking "give me a prompt" invokes no generation tool,
and long descriptions are abbreviated before reaching the model, so the
authoritative templates were unreachable exactly when they were needed and
prompts got written from memory instead.

So the properties worth pinning are not "it returns a dict" but:
  * the real templates come back as DATA (not prose that can be truncated),
  * consulting the docs can never generate anything or cost money,
  * the guidance actually instructs the caller to APPLY the templates.
"""
from __future__ import annotations

import pytest

from handlers.prompt_help import (
    IMAGE_DOC_URL,
    VIDEO_DOC_URL,
    GetPromptGuideParams,
    fn_get_prompt_guide,
)
from prompt_guide import (
    IMAGE_BEST_PRACTICES,
    IMAGE_EDITING_TEMPLATES,
    IMAGE_GENERATION_TEMPLATES,
    VIDEO_PROMPT_ELEMENTS,
)
from tests.fixtures import make_ctx

pytestmark = pytest.mark.asyncio


async def test_image_guide_returns_every_official_template():
    """All of Google's templates, verbatim -- not a subset or a paraphrase."""
    result = await fn_get_prompt_guide(make_ctx(), GetPromptGuideParams(kind="image"))

    assert result.status == "success"
    data = result.data
    assert data.generation_templates == IMAGE_GENERATION_TEMPLATES
    assert data.editing_templates == IMAGE_EDITING_TEMPLATES
    assert data.best_practices == list(IMAGE_BEST_PRACTICES)
    assert IMAGE_DOC_URL in data.source_urls


async def test_video_guide_returns_veo_elements():
    result = await fn_get_prompt_guide(make_ctx(), GetPromptGuideParams(kind="video"))

    assert result.status == "success"
    assert result.data.video_elements == VIDEO_PROMPT_ELEMENTS
    assert VIDEO_DOC_URL in result.data.source_urls
    # Asking for video must not drag the image templates along.
    assert not result.data.generation_templates


async def test_all_returns_both_guides_and_both_sources():
    result = await fn_get_prompt_guide(make_ctx(), GetPromptGuideParams(kind="all"))

    assert result.status == "success"
    assert result.data.generation_templates
    assert result.data.video_elements
    assert sorted(result.data.source_urls) == sorted([IMAGE_DOC_URL, VIDEO_DOC_URL])


async def test_default_kind_is_image():
    """The common case -- "write me a prompt" -- must need no argument."""
    result = await fn_get_prompt_guide(make_ctx(), GetPromptGuideParams())

    assert result.status == "success"
    assert result.data.kind == "image"
    assert result.data.generation_templates


async def test_unknown_kind_is_rejected_and_not_retryable():
    result = await fn_get_prompt_guide(make_ctx(), GetPromptGuideParams(kind="audio"))

    assert result.status == "error"
    assert "audio" in result.error
    assert result.retryable is False


async def test_reading_the_guide_never_calls_the_api_or_costs_money():
    """Consulting the docs must be free and side-effect-free.

    A ctx with no API key and no mocked HTTP would raise or fail on any
    outbound call, so completing successfully proves nothing was generated.
    """
    ctx = make_ctx(with_key=False)

    result = await fn_get_prompt_guide(ctx, GetPromptGuideParams(kind="all"))

    assert result.status == "success"
    # Nothing persisted: no generation row, no stored file.
    assert not ctx.store._data
    assert await ctx.store.count("gemini_generations") == 0


async def test_summary_tells_the_caller_to_apply_the_guide_not_recall_it():
    """Returning templates is useless if the answer is still from memory.

    This is the actual fix, so it is asserted rather than left to prose: the
    summary must direct the caller to apply the templates, and must not
    invite an unrequested generation.
    """
    result = await fn_get_prompt_guide(make_ctx(), GetPromptGuideParams(kind="image"))

    summary = result.summary.lower()
    assert "not from memory" in summary
    assert "unless they explicitly asked" in summary
    # It should name real templates so the choice is concrete.
    assert "photorealistic_scene" in result.summary
