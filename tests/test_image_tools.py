"""Tests for the per-model image tools.

Why these tools exist: Imperal prices a TOOL, not a parameter value, and the
four Nano Banana models differ several-fold in cost per image. One
``generate_image`` tool with ``model=`` could therefore only ever carry a
single price -- overcharging for Lite or undercharging for Pro.

The property that MUST hold is that each tool actually calls the model it
advertises. A copy-paste slip (two tools pointing at the same model) would be
invisible in normal use but would bill the wrong rate for real money, so it is
pinned here per-tool rather than in one loop.
"""
from __future__ import annotations

import pytest

from gemini_config import (
    MODEL_IMAGE, MODEL_IMAGE_FLASH, MODEL_IMAGE_FLASH_LITE, MODEL_IMAGE_LEGACY,
)
from handlers.image_tools import (
    ModelImageParams,
    fn_generate_image_flash,
    fn_generate_image_flash_lite,
    fn_generate_image_legacy,
    fn_generate_image_pro,
)
from tests.fixtures import INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, make_ctx

pytestmark = pytest.mark.asyncio


async def _capture_model(fn) -> tuple[str, object]:
    """Run a per-model tool and report which model id reached the API."""
    ctx = make_ctx(with_key=True)
    captured = {}
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn(ctx, ModelImageParams(prompt="a test subject"))
    return captured.get("json", {}).get("model", ""), result


async def test_pro_tool_calls_pro():
    model, result = await _capture_model(fn_generate_image_pro)
    assert result.status == "success"
    assert model == MODEL_IMAGE


async def test_nano_banana_2_tool_calls_flash():
    model, result = await _capture_model(fn_generate_image_flash)
    assert result.status == "success"
    assert model == MODEL_IMAGE_FLASH


async def test_lite_tool_calls_flash_lite():
    model, result = await _capture_model(fn_generate_image_flash_lite)
    assert result.status == "success"
    assert model == MODEL_IMAGE_FLASH_LITE


async def test_legacy_tool_calls_legacy():
    model, result = await _capture_model(fn_generate_image_legacy)
    assert result.status == "success"
    assert model == MODEL_IMAGE_LEGACY


async def test_every_tool_targets_a_distinct_model():
    """No two tools may share a model, or one of them is mispriced.

    This is the copy-paste guard: four near-identical wrappers are exactly
    where a wrong model id survives review, and the consequence is billing a
    user Pro rates for a Lite render.
    """
    models = []
    for fn in (
        fn_generate_image_pro,
        fn_generate_image_flash,
        fn_generate_image_flash_lite,
        fn_generate_image_legacy,
    ):
        model, _ = await _capture_model(fn)
        models.append(model)

    assert len(set(models)) == 4, f"tools share a model: {models}"
    assert "" not in models, "a tool sent no model at all"


async def test_per_model_tools_are_registered_as_separate_functions():
    """Each must be its own chat function -- that is what carries a price.

    If they collapsed into one registration, the whole point of the split
    (independent per-tool pricing) would be silently lost.
    """
    import main  # noqa: F401  -- registers every handler module
    from app import chat

    expected = {
        "generate_image_nano_banana_pro",
        "generate_image_nano_banana_2",
        "generate_image_nano_banana_2_lite",
        "generate_image_nano_banana_legacy",
    }
    missing = expected - set(chat.functions)
    assert not missing, f"not registered, so not priceable: {missing}"


async def test_legacy_generate_image_still_works():
    """The original tool must keep working: automations may reference it."""
    from handlers.generate import GenerateImageParams, fn_generate_image

    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="still works"))
    assert result.status == "success"


async def test_per_model_tool_has_no_model_parameter():
    """The tool IS the model, so exposing model= would let the caller pick a
    different one than the price was set for."""
    assert "model" not in ModelImageParams.model_fields


async def test_bad_size_is_rejected_before_any_api_call():
    ctx = make_ctx(with_key=True)
    called = False
    real_post = ctx.http.post

    async def _tracking_post(url, **kwargs):
        nonlocal called
        called = True
        return await real_post(url, **kwargs)

    ctx.http.post = _tracking_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="x", image_size="9K"),
    )
    assert result.status == "error"
    assert not called, "an invalid size must not reach the API (it would bill)"
