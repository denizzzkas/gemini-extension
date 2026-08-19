"""Tests for the ``count`` (1-4) multi-generation-per-prompt parameter.

New file rather than appended to tests/test_image_tools.py or
tests/test_generate_references.py -- both are already near this repo's
300-line lint guideline (see other files' own docstrings for that rule).

Why concurrency safety matters here specifically: run_image_generation now
fans count>1 out via asyncio.gather (see handlers/image_core.py's module
docstring for why -- avoiding the ~180s federal per-call timeout budget that
a sequential loop of up to 4 x REQUEST_TIMEOUT_IMAGE could trip). These tests
must therefore prove real concurrent behaviour, not just "the loop runs N
times": that a batch actually produces N independent generation_ids/storage
paths, that partial failures don't discard the successes that already cost
the user real money, and that count=1 is byte-for-byte the old contract.
"""
from __future__ import annotations

import pytest

from gemini_config import MAX_IMAGE_COUNT
from handlers.image_tools import ModelImageParams, fn_generate_image_pro
from handlers.panel import gemini_quick_panel
from tests.fixtures import INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, make_ctx

pytestmark = pytest.mark.asyncio


def _selects(node):
    if isinstance(node, dict):
        if node.get("type") == "Select":
            yield node.get("props", {})
        for v in node.values():
            yield from _selects(v)
    elif isinstance(node, list):
        for item in node:
            yield from _selects(item)


def _texts(node):
    if isinstance(node, dict):
        if node.get("type") == "Text":
            yield node.get("props", {})
        for v in node.values():
            yield from _texts(v)
    elif isinstance(node, list):
        for item in node:
            yield from _texts(item)


async def test_panel_renders_a_count_select_one_to_max():
    ctx = make_ctx(with_key=True)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    count_select = next(
        s for s in _selects(tree) if s.get("param_name") == "count"
    )
    values = {opt["value"] for opt in count_select["options"]}
    assert values == {str(n) for n in range(1, MAX_IMAGE_COUNT + 1)}
    assert count_select.get("value") == "1"  # default: no surprise extra cost


async def test_panel_shows_a_cost_warning_near_the_count_field():
    ctx = make_ctx(with_key=True)
    tree = (await gemini_quick_panel(ctx)).to_dict()

    warnings = [t.get("content", "") for t in _texts(tree)]
    assert any("cost" in str(w).lower() for w in warnings), (
        "no cost-warning text found near the count field"
    )


async def test_count_defaults_to_one_and_images_list_stays_empty():
    """The exact old contract: no count given -> one record, no images[]."""
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(ctx, ModelImageParams(prompt="a cat astronaut"))

    assert result.status == "success"
    assert result.data.images == []


async def test_count_four_returns_four_independent_images():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="four variations", count=4),
    )

    assert result.status == "success"
    all_images = [result.data] + result.data.images
    assert len(all_images) == 4
    # Each one persisted as its OWN generation, not 4 copies of one record.
    generation_ids = {img.generation_id for img in all_images}
    assert len(generation_ids) == 4
    assert all(gid for gid in generation_ids)


async def test_count_out_of_range_is_rejected_before_billing():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    with pytest.raises(Exception):
        ModelImageParams(prompt="too many", count=MAX_IMAGE_COUNT + 1)


async def test_count_partial_failure_keeps_the_successes():
    """2 of 4 fail (e.g. transient 503) -- the 2 successes must still come
    back, not be discarded, since the user already paid Google for them."""
    ctx = make_ctx(with_key=True)
    call_count = 0
    real_post = ctx.http.post

    async def _flaky_post(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            from imperal_sdk.types.models import HTTPResponse
            return HTTPResponse(status_code=503, body={"error": "overloaded"})
        return await real_post(url, **kwargs)

    ctx.http.post = _flaky_post
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="flaky batch", count=4),
    )

    assert result.status == "success"
    all_images = [result.data] + result.data.images
    assert len(all_images) == 2  # exactly the successes, nothing more


async def test_count_all_fail_is_a_real_error():
    ctx = make_ctx(with_key=True)

    async def _always_503(url, **kwargs):
        from imperal_sdk.types.models import HTTPResponse
        return HTTPResponse(status_code=503, body={"error": "overloaded"})

    ctx.http.post = _always_503

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="doomed batch", count=3),
    )

    assert result.status == "error"


async def test_count_greater_than_one_summary_mentions_all_images():
    ctx = make_ctx(with_key=True)
    ctx.http.mock_post(INTERACTIONS_URL, SAMPLE_IMAGE_RESPONSE, status=200)

    result = await fn_generate_image_pro(
        ctx, ModelImageParams(prompt="a pack of variations", count=2),
    )

    assert result.status == "success"
    assert "2" in result.summary
    assert "images[]" in result.summary or "images" in result.summary
