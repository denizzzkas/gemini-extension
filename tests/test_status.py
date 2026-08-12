"""Tests for Gemini connection/status and Studio detail behaviour."""
from __future__ import annotations

import pytest

from handlers.status import (
    fn_check_gemini_connection, CheckGeminiConnectionParams,
    fn_save_gemini_api_key, SaveGeminiAPIKeyParams,
)
from tests.fixtures import make_ctx
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
    ctx.http.mock_get("generativelanguage.googleapis.com/v1beta/models", {"error": "down"}, status=500)

    result = await fn_check_gemini_connection(ctx, CheckGeminiConnectionParams())

    assert result.status == "success"
    assert result.data.configured is True
    assert result.data.api_reachable is False


@pytest.mark.asyncio
async def test_panel_shows_the_full_prompt_not_a_truncated_title():
    """Opening an entry in Studio must retain the prompt in full."""
    import json

    from handlers.panel import gemini_studio_panel

    ctx = make_ctx(with_key=True)
    long_prompt = (
        "A cinematic wide shot of a lone lighthouse on a basalt cliff at dusk, "
        "storm clouds breaking, volumetric light, shot on 35mm film, "
        "high dynamic range, extremely detailed foam on the waves below"
    )
    assert len(long_prompt) > 80

    await ctx.storage.upload("gemini/image/p.png", b"fake-png-bytes", content_type="image/png")
    doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": long_prompt,
        "model": "gemini-3-pro-image",
        "storage_path": "gemini/image/p.png",
        "mime_type": "image/png",
        "created_at": "2026-07-03T10:00:00Z",
    })

    opened = json.dumps((await gemini_studio_panel(ctx, generation_id=doc.id)).to_dict())
    assert long_prompt in opened


# ─── save_gemini_api_key ────────────────────────────────────────────────── #

@pytest.mark.asyncio
async def test_save_gemini_api_key_stores_value_with_write_mode_both():
    """The left panel's inline key field submits here -- requires
    write_mode="both" in app.py's ext.secret() declaration; write_mode="user"
    would make ctx.secrets.set() raise unconditionally."""
    ctx = make_ctx(with_key=False)

    result = await fn_save_gemini_api_key(ctx, SaveGeminiAPIKeyParams(api_key="not-a-real-key-fake-test-value"))

    assert result.status == "success"
    assert result.data.configured is True
    assert await ctx.secrets.get("gemini_api_key") == "not-a-real-key-fake-test-value"


@pytest.mark.asyncio
async def test_save_gemini_api_key_rejects_blank():
    ctx = make_ctx(with_key=False)

    result = await fn_save_gemini_api_key(ctx, SaveGeminiAPIKeyParams(api_key="   "))

    assert result.status == "error"
    assert result.retryable is True


@pytest.mark.asyncio
async def test_save_gemini_api_key_overwrites_existing():
    ctx = make_ctx(with_key=True)

    result = await fn_save_gemini_api_key(ctx, SaveGeminiAPIKeyParams(api_key="new-key-456"))

    assert result.status == "success"
    assert await ctx.secrets.get("gemini_api_key") == "new-key-456"


def test_broken_original_media_chat_tool_is_not_registered():
    """Never expose a chat action that cannot deliver viewable/downloadable media."""
    import main  # noqa: F401 -- registers all handler modules
    from app import chat

    assert "get_original_media" not in chat.functions
    assert "list_generation_history" not in chat.functions
