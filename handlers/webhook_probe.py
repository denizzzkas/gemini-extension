"""TEMPORARY probe — can a webhook route serve real image bytes?

The whole "show the original" question reduces to one unknown: when a webhook
handler returns something, does the gateway pass bytes through with a real
Content-Type (=> the URL is usable as ``ui.Image(src=...)`` and as a download
link), or does it always wrap the return value in JSON?

The SDK does not document the response contract, and guessing it wrong is what
burned the last two attempts, so this asks production directly. Each shape is
returned under a different ``?shape=`` so one request tells us which — if any —
survives the trip.

Delete once the answer is known.
"""
from __future__ import annotations

import base64
import logging

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION
from handlers.media import newest_first

log = logging.getLogger("gemini.whprobe")

# A 1x1 red PNG — small enough that the response shape, not the size, is what
# we are reading.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@ext.webhook("imgprobe", method="GET")
async def wh_imgprobe(ctx, headers: dict, body: str, query_params: dict):
    """Return image bytes in whichever shape ``?shape=`` asks for."""
    shape = (query_params or {}).get("shape", "dict")

    if shape == "bytes":
        return _TINY_PNG
    if shape == "b64":
        return base64.b64encode(_TINY_PNG).decode()
    if shape == "dict_ct":
        return {
            "body": base64.b64encode(_TINY_PNG).decode(),
            "content_type": "image/png",
            "is_base64": True,
        }
    if shape == "fastapi":
        # If the runtime is FastAPI-based it may pass a Response straight out.
        try:
            from fastapi import Response  # noqa: PLC0415

            return Response(content=_TINY_PNG, media_type="image/png")
        except Exception as e:  # noqa: BLE001
            return {"fastapi_unavailable": str(e)}
    if shape == "statuscode":
        return {
            "status_code": 200,
            "headers": {"Content-Type": "image/png"},
            "body": base64.b64encode(_TINY_PNG).decode(),
        }
    if shape == "real":
        # The real question: can we stream an ACTUAL generation's bytes?
        gen_id = (query_params or {}).get("generation_id", "")
        docs = await ctx.store.query(GENERATION_LOG_COLLECTION, limit=50)
        rows = newest_first(list(docs))
        target = None
        for d in rows:
            if not gen_id or d.data.get("generation_id") == gen_id:
                if d.data.get("storage_path"):
                    target = d
                    break
        if target is None:
            return {"error": "no generation with a storage_path found"}
        raw = await ctx.storage.download(target.data["storage_path"])
        return {
            "body": base64.b64encode(raw).decode(),
            "content_type": target.data.get("mime_type") or "image/jpeg",
            "is_base64": True,
            "bytes": len(raw),
        }

    return {
        "shape": "dict",
        "note": "default JSON shape",
        "user_id": getattr(ctx, "user_id", "?"),
        "query_params": query_params,
    }
