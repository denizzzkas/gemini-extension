"""Serve a generation's ORIGINAL file as a real, standalone HTML page.

Confirmed working live: a real second route to the ORIGINAL file, served
from ``@ext.webhook`` -- a genuine HTTP endpoint OUTSIDE the panel RPC
envelope entirely -- while the panel itself only ever carries a small
preview. The existing in-panel download (handlers/panel_html.download_block)
is UNCHANGED and stays the primary mechanism; this is a second, opt-in path.

Why HTML+base64, not a raw byte passthrough (settled by a real deploy, not
a guess)
------------------------------------------------------------------------------
``imperal_sdk.types.events.WebhookResponse.body`` is typed ``dict | str`` --
there is no raw-bytes option in the SDK. A live diagnostic (a webhook that
returned all 256 byte values 0x00-0xFF via ``raw_bytes.decode("latin-1")`` as
the ``str`` body) proved the gateway re-encodes that string as UTF-8 before
it reaches the browser: every byte >= 0x80 came back as two bytes instead of
one (0x80 -> ``C2 80``, 0xC0 -> ``C3 80``, etc. -- textbook UTF-8
re-encoding). Real PNG/JPEG bytes are full of values >= 0x80, so a raw
passthrough would corrupt every image. Base64 sidesteps this because it only
ever emits ASCII 0-127, which UTF-8 passes through unchanged -- so the
ORIGINAL is base64-encoded into an HTML page here, exactly as before.

What IS confirmed (by real deploys, not assumption)
------------------------------------------------------
- ``imperal_sdk.extension.Extension.webhook`` registers a real route,
  ``/v1/ext/{app_id}/webhook/{path}``, dispatched by the gateway -- a
  different code path from the panel/tool RPC envelope that
  ``imperal_sdk.rpc.codec`` caps at 256 KB.
- The gateway forwards ``WebhookResponse.headers`` (e.g. ``Content-Type``)
  to the browser -- confirmed by the byte-passthrough diagnostic above,
  whose response was inspected with its declared content type intact.
- ``Context.webhook_url()`` builds a fully public URL, reachable by a plain,
  unauthenticated browser -- this endpoint is live in production and opens
  the original in a new tab from the panel's "Open original in a new tab"
  button.

Security model, since a webhook has none of the panel's per-user auth
--------------------------------------------------------------------------
``ctx`` inside a webhook handler carries the synthetic ``user_id ==
"__webhook__"``, not the caller's real identity, so ownership can't be
checked the normal way. Every link is instead HMAC-signed with an expiry
(handlers/media_link.py) -- unguessable and self-expiring, standing in for
per-user auth that literally cannot exist on this path.
"""
from __future__ import annotations

import base64
import logging

from imperal_sdk import WebhookResponse

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION
from handlers.media_link import verify_media_link

log = logging.getLogger("gemini.media_webhook")

_MAX_INLINE_BYTES = 60 * 1024 * 1024  # sanity ceiling, not the point being tested


def _html_page(mime_type: str, b64_data: str, title: str) -> str:
    """A tiny standalone page: no panel framing, no RPC envelope at all."""
    is_video = mime_type.startswith("video/")
    tag = (
        f'<video src="data:{mime_type};base64,{b64_data}" controls autoplay '
        'style="max-width:100%;max-height:100vh;display:block;margin:auto;"></video>'
        if is_video else
        f'<img src="data:{mime_type};base64,{b64_data}" '
        'style="max-width:100%;max-height:100vh;display:block;margin:auto;" />'
    )
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{safe_title}</title></head>"
        "<body style='margin:0;background:#111;display:flex;min-height:100vh;"
        f"align-items:center;justify-content:center;'>{tag}</body></html>"
    )


def _error_page(title: str, message: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head>"
        "<body style='font-family:system-ui,sans-serif;background:#111;"
        "color:#eee;display:flex;min-height:100vh;align-items:center;"
        f"justify-content:center;text-align:center;'><div><h2>{title}</h2>"
        f"<p>{message}</p></div></body></html>"
    )


@ext.webhook("/media", method="GET")
async def serve_media(ctx, headers, body, query_params):
    """GET /v1/ext/gemini/webhook/media?id=<generation_id>&exp=<ts>&sig=<hmac>

    Verifies the signature+expiry, downloads the ORIGINAL bytes straight
    from storage (no preview, no shrink), and returns them embedded in a
    real HTML page with its own ``Content-Type`` -- outside the panel RPC
    envelope entirely. This is the part of the experiment that needs a real
    deploy to confirm (see this module's own docstring).
    """
    generation_id = query_params.get("id", "")
    exp_raw = query_params.get("exp", "")
    sig = query_params.get("sig", "")

    if not generation_id or not exp_raw or not sig:
        return WebhookResponse(
            status_code=400,
            body=_error_page("Bad link", "This media link is missing required parameters."),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    try:
        exp = int(exp_raw)
    except ValueError:
        return WebhookResponse(
            status_code=400,
            body=_error_page("Bad link", "This media link is malformed."),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    from handlers.media_link import get_or_create_signing_key

    secret = await get_or_create_signing_key(ctx)
    if not verify_media_link(secret, generation_id, exp, sig):
        return WebhookResponse(
            status_code=403,
            body=_error_page(
                "Link expired or invalid",
                "This media link is no longer valid. Open the generation in "
                "the panel again for a fresh link.",
            ),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    try:
        doc = await ctx.store.get(GENERATION_LOG_COLLECTION, generation_id)
    except Exception as e:  # noqa: BLE001
        log.error("media_webhook: store.get failed for %r: %s", generation_id, e)
        doc = None

    if doc is None:
        return WebhookResponse(
            status_code=404,
            body=_error_page("Not found", "This generation no longer exists."),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    storage_path = doc.data.get("storage_path")
    mime_type = doc.data.get("mime_type") or "application/octet-stream"
    if not storage_path:
        return WebhookResponse(
            status_code=404,
            body=_error_page("No file", "This generation has no stored file."),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    try:
        raw = await ctx.storage.download(storage_path)
    except Exception as e:  # noqa: BLE001
        log.error("media_webhook: download failed for %r: %s", storage_path, e)
        return WebhookResponse(
            status_code=502,
            body=_error_page("Read failed", "The stored file could not be read just now."),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    if len(raw) > _MAX_INLINE_BYTES:
        return WebhookResponse(
            status_code=413,
            body=_error_page("Too large", "This file is too large for this test page."),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    b64_data = base64.b64encode(raw).decode()
    page = _html_page(mime_type, b64_data, doc.data.get("prompt") or "Gemini generation")
    log.info(
        "media_webhook: served %r (%d bytes, %s) via webhook route",
        generation_id, len(raw), mime_type,
    )
    return WebhookResponse(
        status_code=200,
        body=page,
        headers={"Content-Type": "text/html; charset=utf-8"},
    )
