"""Tests for the Gemini video generation handler, plus app-level lifecycle.

Image generation tests used to live here too (against a generic
``generate_image`` tool with a ``model=`` parameter). That tool was removed
-- see handlers/generate.py's module docstring -- in favour of the four
distinctly-priced per-model tools in handlers/image_tools.py, which are
covered by tests/test_image_tools.py instead.
"""
from __future__ import annotations

import pytest

from handlers.generate import fn_generate_video, GenerateVideoParams
from app import health_check, on_install
from tests.fixtures import (
    make_ctx, INTERACTIONS_URL,
    SAMPLE_VIDEO_RESPONSE, FAKE_VIDEO_B64,
)


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


@pytest.mark.asyncio
async def test_generate_video_oversized_returns_link_not_bytes():
    """Same size ceiling as images -- there is no video shrink codec, so an

    oversized video must never put its bytes in the reply (risking a silent
    failure above the proven ceiling); it must instead flag ``is_preview``
    and offer a signed link to the real file.
    """
    import base64 as _b64

    from core.preview import PROVEN_GOOD_CHARS

    ctx = make_ctx(with_key=True)
    big_video = _b64.b64encode(b"x" * (PROVEN_GOOD_CHARS + 1000)).decode()
    response = {
        "id": "interaction_vid_big",
        "status": "completed",
        "model": "gemini-omni-flash-preview",
        "steps": [{
            "type": "model_output",
            "content": [
                {"type": "video", "data": big_video, "mime_type": "video/mp4"},
            ],
        }],
    }
    ctx.http.mock_post(INTERACTIONS_URL, response, status=200)

    result = await fn_generate_video(ctx, GenerateVideoParams(prompt="a long scene"))

    assert result.status == "success"
    assert result.data.is_preview is True
    assert result.data.video_base64 == ""
    assert result.data.full_video_url.startswith("https://")


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
