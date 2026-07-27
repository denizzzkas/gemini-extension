"""Reference-image tests: character/scene consistency across generations.

Split out of test_generate.py to keep each file under the 300-line limit the
deploy validator warns about.

These cover the path that makes a reference actually WORK: the bytes must be
re-downloaded from storage and sent as a multimodal block, references owned by
another user must be ignored (an authorization boundary, not a nicety), and an
unresolvable reference must fail loudly instead of silently generating
something unrelated to what was asked for.
"""
from __future__ import annotations

import base64

import pytest

from handlers.generate import fn_generate_image, GenerateImageParams
from tests.fixtures import (
    make_ctx, INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, FAKE_IMAGE_B64,
)


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

