"""Tests for Gemini generation handlers (generate_image / generate_video)."""
from __future__ import annotations

import base64

import pytest

from handlers.generate import (
    fn_generate_image, GenerateImageParams,
    fn_generate_video, GenerateVideoParams,
)
from app import health_check, on_install
from tests.fixtures import (
    make_ctx, INTERACTIONS_URL,
    SAMPLE_IMAGE_RESPONSE, SAMPLE_VIDEO_RESPONSE,
    FAKE_IMAGE_B64, FAKE_VIDEO_B64,
)


# ─── generate_image ───────────────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_generate_image_success():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="a cat astronaut"))

    assert result.status == "success"
    assert result.data.image_base64 == FAKE_IMAGE_B64
    assert result.data.mime_type == "image/png"
    assert result.data.model == "gemini-3-pro-image"
    assert "cat astronaut" in result.summary


@pytest.mark.asyncio
async def test_generate_image_no_api_key():
    ctx = make_ctx(with_key=False)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="a cat astronaut"))

    assert result.status == "error"
    assert "API key" in result.error


@pytest.mark.asyncio
async def test_generate_image_api_error():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, {"error": {"message": "quota exceeded"}}, status=429)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="a cat astronaut"))

    assert result.status == "error"
    assert "quota exceeded" in result.error
    assert result.retryable is True


@pytest.mark.asyncio
async def test_generate_image_no_media_in_response():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, {
        "id": "x", "status": "completed", "model": "gemini-3-pro-image",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "I can't do that."}]}],
    }, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="something refused"))

    assert result.status == "error"
    assert result.retryable is True


# ─── generate_video ───────────────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_generate_video_success():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_VIDEO_RESPONSE, status=200)

    result = await fn_generate_video(ctx, GenerateVideoParams(prompt="a marble rolling down a track"))

    assert result.status == "success"
    assert result.data.video_base64 == FAKE_VIDEO_B64
    assert result.data.mime_type == "video/mp4"
    assert result.data.model == "gemini-omni-flash-preview"


@pytest.mark.asyncio
async def test_generate_video_no_api_key():
    ctx = make_ctx(with_key=False)

    result = await fn_generate_video(ctx, GenerateVideoParams(prompt="a marble rolling"))

    assert result.status == "error"
    assert "API key" in result.error


@pytest.mark.asyncio
async def test_generate_video_server_error_retryable():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, {"error": {"message": "internal error"}}, status=503)

    result = await fn_generate_video(ctx, GenerateVideoParams(prompt="a marble rolling"))

    assert result.status == "error"
    assert result.retryable is True


# ─── health_check (app-level) ─────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_health_check_reachable():
    # health_check is app-level (no user/per-user store) and, since
    # gemini_api_key is scope="user" (each user brings their own key),
    # it reports only the one genuinely app-level fact: API reachability.
    ctx = make_ctx(with_key=False)
    ctx.http.mock_get("generativelanguage.googleapis.com/v1beta/models", {"models": []}, status=200)

    status = await health_check(ctx)

    assert status.details["api_reachable"] is True
    assert "configured" not in status.details


@pytest.mark.asyncio
async def test_health_check_unreachable():
    ctx = make_ctx(with_key=False)
    ctx.http.mock_get(
        "generativelanguage.googleapis.com/v1beta/models",
        {"error": "unavailable"}, status=503,
    )

    status = await health_check(ctx)

    assert status.details["api_reachable"] is False



# ─── on_install lifecycle hook ──────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_on_install_runs_without_error():
    ctx = make_ctx(with_key=False)
    # Should just log -- no exception, no return value, no side effects on ctx.
    result = await on_install(ctx)
    assert result is None


# ─── model selection (multiple image models) ────────────────────────────────── #

@pytest.mark.asyncio
async def test_generate_image_default_model_is_nano_banana_pro():
    from gemini_config import MODEL_IMAGE
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="a cat astronaut"))

    assert result.status == "success"
    assert result.data.model == MODEL_IMAGE


@pytest.mark.asyncio
async def test_generate_image_explicit_model_is_used_and_sent():
    from gemini_config import MODEL_IMAGE_FLASH_LITE
    ctx = make_ctx(with_key=True)

    captured = {}
    real_post = ctx.http.post
    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)
    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(
        prompt="a quick draft sketch", model=MODEL_IMAGE_FLASH_LITE,
    ))

    assert result.status == "success"
    assert result.data.model == MODEL_IMAGE_FLASH_LITE
    assert captured["json"]["model"] == MODEL_IMAGE_FLASH_LITE


@pytest.mark.asyncio
async def test_generate_image_rejects_unknown_model():
    ctx = make_ctx(with_key=True)

    result = await fn_generate_image(ctx, GenerateImageParams(
        prompt="a cat astronaut", model="not-a-real-model",
    ))

    assert result.status == "error"
    assert "Unknown image model" in result.error
    assert result.retryable is False



# ─── output size / response_format (the payload fix at the source) ─────────── #

@pytest.mark.asyncio
async def test_generate_image_sends_only_documented_response_format_keys():
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

    ctx = make_ctx(with_key=True)
    captured = {}
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="a cat astronaut"))

    assert result.status == "success"
    fmt = captured["json"]["response_format"]
    assert fmt["type"] == "image"
    assert fmt["image_size"] == DEFAULT_IMAGE_SIZE == "1K"

    documented = {"type", "aspect_ratio", "image_size"}
    undocumented = set(fmt) - documented
    assert not undocumented, (
        f"undocumented response_format keys would 400 the whole call: {undocumented}"
    )


@pytest.mark.asyncio
async def test_generate_image_honours_an_explicit_size():
    ctx = make_ctx(with_key=True)
    captured = {}
    real_post = ctx.http.post

    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)

    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(
        ctx, GenerateImageParams(prompt="a detailed poster", image_size="4K"),
    )

    assert result.status == "success"
    assert captured["json"]["response_format"]["image_size"] == "4K"


@pytest.mark.asyncio
async def test_generate_image_rejects_a_bogus_size():
    """Uppercase K is required by the API; a silent pass-through would 400."""
    ctx = make_ctx(with_key=True)

    result = await fn_generate_image(
        ctx, GenerateImageParams(prompt="a cat astronaut", image_size="1k"),
    )

    assert result.status == "error"
    assert "image_size" in result.error
    assert result.retryable is False
