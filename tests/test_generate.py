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


# ─── reference images (character/scene consistency) ────────────────────────── #

@pytest.mark.asyncio
async def test_generate_image_with_valid_reference_builds_multimodal_payload():
    from clients import gemini_client as gc

    ctx = make_ctx(with_key=True)

    # Seed one prior "image" generation this same user owns, with real bytes
    # stashed in mock storage -- this is what reference_generation_ids points at.
    ref_bytes = b"\x89PNG\r\n\x1a\nfake-ref-bytes"
    await ctx.storage.upload("gemini/image/ref123.png", ref_bytes, content_type="image/png")
    doc = await ctx.store.create("gm_generations", {
        "user_id": ctx.user.imperal_id, "kind": "image",
        "prompt": "the antagonist on the rooftop", "model": "gemini-3-pro-image",
        "url": "https://panel.imperal.io/storage/default/gemeni/ref123.png",
        "storage_path": "gemini/image/ref123.png", "mime_type": "image/png",
        "created_at": "2026-07-19T00:00:00Z",
    })

    captured = {}
    real_post = ctx.http.post
    async def _capturing_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return await real_post(url, **kwargs)
    ctx.http.post = _capturing_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(
        prompt="same antagonist, new pose", reference_generation_ids=[doc.id],
    ))

    assert result.status == "success"
    sent = captured["json"]
    assert isinstance(sent["input"], list)  # multimodal shape, not a bare string
    assert sent["input"][0] == {"type": "text", "text": "same antagonist, new pose"}
    assert sent["input"][1]["type"] == "image"
    assert sent["input"][1]["mime_type"] == "image/png"
    decoded = base64.b64decode(sent["input"][1]["data"])
    assert decoded == ref_bytes  # bytes survived the round trip uncorrupted


@pytest.mark.asyncio
async def test_generate_image_with_unresolvable_reference_errors_cleanly():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(
        prompt="same antagonist, new pose", reference_generation_ids=["does-not-exist"],
    ))

    assert result.status == "error"
    assert "reference_generation_ids" in result.error or "list_generation_history" in result.error


@pytest.mark.asyncio
async def test_generate_image_reference_owned_by_another_user_is_ignored():
    ctx = make_ctx(with_key=True)
    other_doc = await ctx.store.create("gm_generations", {
        "user_id": "someone_else", "kind": "image",
        "prompt": "not yours", "model": "gemini-3-pro-image",
        "storage_path": "gemini/image/other.png", "mime_type": "image/png",
    })
    await ctx.storage.upload("gemini/image/other.png", b"other-users-bytes", content_type="image/png")
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(
        prompt="steal their reference", reference_generation_ids=[other_doc.id],
    ))

    assert result.status == "error"  # not resolvable -> no silent cross-user leak


@pytest.mark.asyncio
async def test_generate_image_success_returns_generation_id():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image(ctx, GenerateImageParams(prompt="a cat astronaut"))

    assert result.status == "success"
    assert result.data.generation_id  # non-empty -- usable as a future reference


# ─── on_install lifecycle hook ──────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_on_install_runs_without_error():
    ctx = make_ctx(with_key=False)
    # Should just log -- no exception, no return value, no side effects on ctx.
    result = await on_install(ctx)
    assert result is None
