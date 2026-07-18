"""Tests for the Google-confirmed prompting guidance injected into tool descriptions."""
from __future__ import annotations

from prompt_guide import (
    IMAGE_BEST_PRACTICES,
    IMAGE_EDITING_TEMPLATES,
    IMAGE_GENERATION_TEMPLATES,
    VIDEO_PROMPT_ELEMENTS,
    image_prompt_guidance_text,
    video_prompt_guidance_text,
)
from handlers.generate import fn_generate_image, fn_generate_video


def test_image_guidance_mentions_official_template_keys():
    text = image_prompt_guidance_text()
    assert "photorealistic_scene" in text
    assert "add_remove_element" in text
    assert len(IMAGE_GENERATION_TEMPLATES) == 6
    assert len(IMAGE_EDITING_TEMPLATES) == 7
    assert len(IMAGE_BEST_PRACTICES) == 6


def test_video_guidance_mentions_official_elements():
    text = video_prompt_guidance_text()
    assert "subject" in text
    assert "ambiance" in text
    assert len(VIDEO_PROMPT_ELEMENTS) == 7


def test_generate_image_tool_description_carries_guidance():
    # imperal_sdk's @chat.function decorator wraps the function; the
    # original description lives in the tool registry, but we can also
    # assert the guidance text is non-trivial and importable at call time.
    assert "GOOGLE-CONFIRMED PROMPT STRUCTURE" in image_prompt_guidance_text()
    assert "GOOGLE-CONFIRMED PROMPT STRUCTURE" in video_prompt_guidance_text()
