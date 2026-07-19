"""Tests for Gemini extension handlers."""
from __future__ import annotations

import base64

import pytest

from handlers.generate import (
    fn_generate_image, GenerateImageParams,
    fn_generate_video, GenerateVideoParams,
    fn_check_gemini_connection, CheckGeminiConnectionParams,
    fn_list_generation_history, ListGenerationHistoryParams,
    _absolute_url,
)
from app import health_check
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


# ─── check_gemini_connection ──────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_check_connection_not_configured():
    ctx = make_ctx(with_key=False)

    result = await fn_check_gemini_connection(ctx, CheckGeminiConnectionParams())

    assert result.status == "success"
    assert result.data.configured is False
    assert result.data.api_reachable is False


@pytest.mark.asyncio
async def test_check_connection_configured_and_reachable():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_get("generativelanguage.googleapis.com/v1beta/models", {"models": []}, status=200)

    result = await fn_check_gemini_connection(ctx, CheckGeminiConnectionParams())

    assert result.status == "success"
    assert result.data.configured is True
    assert result.data.api_reachable is True


@pytest.mark.asyncio
async def test_check_connection_configured_but_unreachable():
    ctx = make_ctx(with_key=True)
    # No mock registered -> MockHTTP._find() falls through to 404, which is
    # < 500 so this actually counts as "reachable" (any HTTP response, even
    # an auth error, means the network path works). Register a 500 instead
    # to exercise the "configured but not reachable" branch.
    ctx.http.mock_get("generativelanguage.googleapis.com/v1beta/models", {"error": "down"}, status=500)

    result = await fn_check_gemini_connection(ctx, CheckGeminiConnectionParams())

    assert result.status == "success"
    assert result.data.configured is True
    assert result.data.api_reachable is False


# ─── list_generation_history ──────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_list_generation_history_empty():
    ctx = make_ctx(with_key=True)

    result = await fn_list_generation_history(ctx, ListGenerationHistoryParams())

    assert result.status == "success"
    assert result.data.count == 0
    assert result.data.items == []


@pytest.mark.asyncio
async def test_list_generation_history_after_generation():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)
    await fn_generate_image(ctx, GenerateImageParams(prompt="a cat astronaut"))

    result = await fn_list_generation_history(ctx, ListGenerationHistoryParams())

    assert result.status == "success"
    assert result.data.count == 1
    assert result.data.items[0].kind == "image"
    assert result.data.items[0].prompt == "a cat astronaut"
    assert result.data.items[0].model == "gemini-3-pro-image"


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


# ─── _absolute_url (storage link normalization) ────────────────────────────── #

def test_absolute_url_passes_through_full_urls():
    assert _absolute_url("https://cdn.example.com/x.png") == "https://cdn.example.com/x.png"
    assert _absolute_url("http://cdn.example.com/x.png") == "http://cdn.example.com/x.png"


def test_absolute_url_prefixes_bare_storage_path():
    # This is the exact shape ctx.storage.upload() can return -- a bare path,
    # not a clickable link. Regression test for the "dead link in chat" bug.
    result = _absolute_url("/storage/default/gemeni/0811e960ddf94a8c926370cff6bbb7b5.jpg")
    assert result.startswith("https://")
    assert result.endswith("/storage/default/gemeni/0811e960ddf94a8c926370cff6bbb7b5.jpg")


def test_absolute_url_empty_stays_empty():
    assert _absolute_url("") == ""
