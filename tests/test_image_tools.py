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


async def test_legacy_generic_generate_image_tool_is_gone():
    """The generic tool was removed -- these four per-model tools replace it.

    Imperal prices a TOOL, not a parameter value, so keeping a generic
    ``generate_image`` alongside these four meant the SAME generation could
    be billed at two different rates depending on which one was called. See
    handlers/generate.py's module docstring for the removal rationale.
    """
    from handlers.generate import GenerateVideoParams, fn_generate_video  # noqa: F401
    import handlers.generate as generate_module

    assert not hasattr(generate_module, "fn_generate_image")
    assert not hasattr(generate_module, "GenerateImageParams")


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


async def test_rejects_a_bogus_size_case():
    """Uppercase K is required by the API; a silent pass-through would 400."""
    ctx = make_ctx(with_key=True)

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="a cat astronaut", image_size="1k"),
    )

    assert result.status == "error"
    assert "image_size" in result.error
    assert result.retryable is False


# ─── output size / response_format (the payload fix at the source) ─────────── #

async def test_sends_only_documented_response_format_keys():
    """The request must carry ONLY keys the Interactions API documents.

    This is the regression guard for a self-inflicted outage: a ``mime_type``
    key was added to ``response_format`` on the theory that the output format
    could be requested. Per ai.google.dev/gemini-api/docs/image-generation the
    object is {"type", "aspect_ratio", "image_size"} -- there is no such
    field, and sending it made the API reject EVERY generation.

    Output format is the model's choice (PNG), not a request parameter, so a
    test asserting a requested mime_type was encoding the bug as a spec.
    """
    from gemini_config import DEFAULT_IMAGE_SIZE

    model, result = await _capture_model(fn_generate_image_pro)
    ctx = make_ctx(with_key=True)
    captured = {}
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(ctx, ModelImageParams(prompt="a cat astronaut"))

    assert result.status == "success"
    fmt = captured["json"]["response_format"]
    assert fmt["type"] == "image"
    assert fmt["image_size"] == DEFAULT_IMAGE_SIZE == "1K"

    documented = {"type", "aspect_ratio", "image_size"}
    undocumented = set(fmt) - documented
    assert not undocumented, (
        f"undocumented response_format keys would 400 the whole call: {undocumented}"
    )


async def test_honours_an_explicit_size():
    ctx = make_ctx(with_key=True)
    captured = {}
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="a detailed poster", image_size="4K"),
    )

    assert result.status == "success"
    assert captured["json"]["response_format"]["image_size"] == "4K"


# ─── aspect_ratio ────────────────────────────────────────────────────────── #

async def test_defaults_to_square_aspect_ratio():
    ctx = make_ctx(with_key=True)
    captured = {}
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(ctx, ModelImageParams(prompt="a cat astronaut"))

    assert result.status == "success"
    assert captured["json"]["response_format"]["aspect_ratio"] == "1:1"


async def test_honours_an_explicit_aspect_ratio():
    ctx = make_ctx(with_key=True)
    captured = {}
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="a phone wallpaper", aspect_ratio="9:16"),
    )

    assert result.status == "success"
    assert captured["json"]["response_format"]["aspect_ratio"] == "9:16"


async def test_rejects_a_bogus_aspect_ratio_before_billing():
    ctx = make_ctx(with_key=True)
    called = False
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        nonlocal called
        called = True
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="a cat astronaut", aspect_ratio="1:2"),
    )

    assert result.status == "error"
    assert "aspect_ratio" in result.error
    assert not called, "an invalid aspect_ratio must not reach the API (it would bill)"
