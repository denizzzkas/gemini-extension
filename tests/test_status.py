"""Tests for Gemini status/history handlers (check_gemini_connection, list_generation_history)."""
from __future__ import annotations

import pytest

from handlers.generate import fn_generate_image, GenerateImageParams
from handlers.status import (
    fn_check_gemini_connection, CheckGeminiConnectionParams,
    fn_list_generation_history, ListGenerationHistoryParams,
)
from tests.fixtures import make_ctx, INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE
from gemini_config import GENERATION_LOG_COLLECTION


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


# ─── retroactive URL normalization (legacy records with a bare path) ────────── #

@pytest.mark.asyncio
async def test_list_generation_history_normalizes_legacy_relative_url():
    ctx = make_ctx(with_key=True)
    # Simulate a record saved before the URL-normalization fix existed --
    # a bare storage path with no host, exactly what ctx.storage.upload()
    # used to hand back verbatim.
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": "an old pre-fix generation",
        "model": "gemini-3-pro-image",
        "url": "/storage/default/gemeni/legacy123.jpg",
        "storage_path": "gemini/image/legacy123.jpg",
        "mime_type": "image/jpeg",
        "created_at": "2026-07-19T00:00:00+00:00",
    })

    result = await fn_list_generation_history(ctx, ListGenerationHistoryParams())

    assert result.status == "success"
    assert result.data.items[0].url.startswith("https://")
    assert result.data.items[0].url.endswith("/storage/default/gemeni/legacy123.jpg")


@pytest.mark.asyncio
async def test_history_is_returned_newest_first():
    """The backend promises no row order, so we must impose one.

    Live symptom this guards: with a capped window and no ordering, recent
    generations could fall outside the page and simply never be shown -- the
    user reported not being able to see all of their images.
    """
    from handlers.status import fn_list_generation_history, ListGenerationHistoryParams

    ctx = make_ctx(with_key=True)
    # Inserted deliberately out of chronological order.
    for created, prompt in [
        ("2026-07-02T10:00:00Z", "middle"),
        ("2026-07-01T10:00:00Z", "oldest"),
        ("2026-07-03T10:00:00Z", "newest"),
    ]:
        await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": "image",
            "prompt": prompt,
            "model": "gemini-3-pro-image",
            "created_at": created,
        })

    result = await fn_list_generation_history(ctx, ListGenerationHistoryParams(limit=10))
    prompts = [i.prompt for i in result.data.items]
    assert prompts == ["newest", "middle", "oldest"], prompts


@pytest.mark.asyncio
async def test_panel_shows_the_full_prompt_not_a_truncated_title():
    """The prompt that produced an image must be readable in full.

    Card titles are clipped to 80 chars; real prompts run many hundreds, so
    the title alone never showed what actually generated the image.
    """
    import json

    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    long_prompt = (
        "A cinematic wide shot of a lone lighthouse on a basalt cliff at dusk, "
        "storm clouds breaking, volumetric light, shot on 35mm film, "
        "high dynamic range, extremely detailed foam on the waves below"
    )
    assert len(long_prompt) > 80  # guard the premise

    png = b"fake-png-bytes"
    await ctx.storage.upload("gemini/image/p.png", png, content_type="image/png")
    doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": long_prompt,
        "model": "gemini-3-pro-image",
        "storage_path": "gemini/image/p.png",
        "mime_type": "image/png",
        "created_at": "2026-07-03T10:00:00Z",
    })

    opened = json.dumps((await gemini_quick_panel(ctx, generation_id=doc.id)).to_dict())
    assert long_prompt in opened, "the full prompt is not rendered anywhere"
