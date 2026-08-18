"""TEST: serve a generation's ORIGINAL file as a real, standalone HTML page.

THIS IS AN EXPERIMENT, not the default download path
------------------------------------------------------
The existing in-panel download (handlers/panel_html.download_block) is
UNCHANGED and still the primary mechanism. This module is a second, opt-in
route being tested per the user's own proposal: since a panel reply is
capped at ~256 KB (measured, see handlers/panel_html.py's own comment),
what if the ORIGINAL is instead served from ``@ext.webhook`` -- a genuine
HTTP endpoint OUTSIDE the panel RPC envelope entirely -- while the panel
itself only ever carries a small preview?

Why this might actually bypass the cap (evidence, not a guess)
------------------------------------------------------------------
- ``imperal_sdk.extension.Extension.webhook`` registers a real route,
  ``/v1/ext/{app_id}/webhook/{path}``, dispatched by the gateway -- a
  different code path from the panel/tool RPC envelope that
  ``imperal_sdk.rpc.codec`` caps at 256 KB (that cap is documented AT the
  codec, i.e. specific to the RPC envelope, not to HTTP responses in
  general).
- ``Context.webhook_url()`` builds a fully public URL
  (``https://panel.imperal.io/v1/ext/{app_id}/webhook/{path}``), and
  ``spotify-extension/handlers/auth.py`` already proves a GET webhook is
  reachable by a plain, unauthenticated browser redirect (Spotify's own
  servers send the user's browser straight to it after OAuth consent, with
  no Imperal-issued token attached) -- i.e. this is not a theoretical path.

What is NOT yet confirmed (why this stays a TEST)
----------------------------------------------------
- Whether the gateway forwards ``WebhookResponse.headers`` (e.g.
  ``Content-Type: text/html``) to the browser unchanged -- the SDK dataclass
  accepts them, but nothing in this package proves the gateway honours them.
- Whether the gateway/reverse-proxy imposes its OWN size ceiling on a
  webhook response body, separate from the RPC envelope's 256 KB.
Both can only be confirmed by really deploying this and opening the link.

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


@ext.webhook("/media_bytes_test", method="GET")
async def serve_media_bytes_test(ctx, headers, body, query_params):
    """DIAGNOSTIC ONLY -- to be deleted once the real question is answered.

    Returns all 256 possible byte values (0x00-0xFF) once each, decoded via
    latin-1 into a ``str`` (the only lossless str<->bytes mapping in Python,
    since latin-1 assigns one code point per byte 0-255) and handed to
    ``WebhookResponse(body=...)``. ``WebhookResponse.body`` is typed
    ``dict | str`` in the SDK -- there is no raw-bytes option -- so this
    checks, with a real deploy, whether the gateway's re-encoding of that
    ``str`` back to bytes on the wire is byte-for-byte lossless or whether it
    silently mangles anything (e.g. re-encoding as UTF-8, which would turn
    every byte >= 0x80 into 2+ bytes). This settles it empirically instead of
    guessing from the dataclass alone.
    """
    raw = bytes(range(256))
    return WebhookResponse(
        status_code=200,
        body=raw.decode("latin-1"),
        headers={"Content-Type": "application/octet-stream"},
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
