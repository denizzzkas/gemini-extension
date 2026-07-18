"""Tests for the Gemini skeleton refresh handler."""
from __future__ import annotations

import pytest

from handlers.skeleton import refresh_gemini_stats
from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx


@pytest.mark.asyncio
async def test_skeleton_no_key_no_generations():
    ctx = make_ctx(with_key=False)

    result = await refresh_gemini_stats(ctx)

    data = result["response"]
    assert data["configured"] is False
    assert data["image_count"] == 0
    assert data["video_count"] == 0
    assert data["total_count"] == 0
    assert data["last_prompt"] == ""


@pytest.mark.asyncio
async def test_skeleton_configured_with_generations():
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image",
        "prompt": "a cat astronaut", "model": "gemini-3-pro-image",
        "url": "", "created_at": "",
    })
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "video",
        "prompt": "a paper airplane", "model": "gemini-omni-flash-preview",
        "url": "", "created_at": "",
    })

    result = await refresh_gemini_stats(ctx)

    data = result["response"]
    assert data["configured"] is True
    assert data["image_count"] == 1
    assert data["video_count"] == 1
    assert data["total_count"] == 2


@pytest.mark.asyncio
async def test_skeleton_ignores_other_users_generations():
    ctx = make_ctx(with_key=True)
    await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": "someone_else", "kind": "image",
        "prompt": "not mine", "model": "gemini-3-pro-image",
        "url": "", "created_at": "",
    })

    result = await refresh_gemini_stats(ctx)

    assert result["response"]["total_count"] == 0
