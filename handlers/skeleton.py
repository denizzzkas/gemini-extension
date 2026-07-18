"""Skeleton refresh — periodic app-facts snapshot for the LLM context.

Registered via ``@ext.skeleton("gemini_stats", ...)`` which the SDK maps to
the naming convention ``skeleton_refresh_gemini_stats``. The platform polls
this on a TTL and surfaces the returned scalar fields directly in Webbee's
context, so she can answer things like "is my Gemini key connected?" or
"how many images have I generated?" without an extra tool round-trip.
"""
from __future__ import annotations

import logging

from app import ext

log = logging.getLogger("gemini.skeleton")


@ext.skeleton("gemini_stats", ttl=300, description="Gemini key status and generation counts")
async def refresh_gemini_stats(ctx) -> dict:
    """Return a compact snapshot: key configured?, totals, last generation.

    Idempotent, read-only, safe to run on every tick. Deliberately avoids
    any network call (no reachability probe here) — that stays in
    ``check_gemini_connection`` / ``health_check``, which are bounded and
    user/app scoped respectively. Skeleton refreshes must stay cheap.
    """
    configured = False
    try:
        key = await ctx.secrets.get("gemini_api_key")
        configured = bool(key)
    except Exception as e:  # noqa: BLE001
        log.error("skeleton: get gemini_api_key failed: %s", e)

    image_count = 0
    video_count = 0
    last_prompt = ""
    last_kind = ""

    try:
        from gemini_config import GENERATION_LOG_COLLECTION

        image_count = await ctx.store.count(GENERATION_LOG_COLLECTION, where={
            "user_id": ctx.user.imperal_id, "kind": "image",
        })
        video_count = await ctx.store.count(GENERATION_LOG_COLLECTION, where={
            "user_id": ctx.user.imperal_id, "kind": "video",
        })
        recent = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id},
            limit=1,
        )
        if recent.data:
            last_prompt = recent.data[0].data.get("prompt", "")
            last_kind = recent.data[0].data.get("kind", "")
    except Exception as e:  # noqa: BLE001
        log.error("skeleton: generation counts failed: %s", e)

    return {"response": {
        "configured": configured,
        "image_count": image_count,
        "video_count": video_count,
        "total_count": image_count + video_count,
        "last_kind": last_kind,
        "last_prompt": last_prompt[:120],
    }}
