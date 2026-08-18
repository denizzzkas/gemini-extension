"""TEST: signed links to a generation's ORIGINAL file.

Part of the media-webhook experiment (see handlers/media_webhook.py for the
full architecture and its still-open questions). This module only does the
signing/verification -- kept separate so it can be unit-tested without any
webhook/HTTP machinery involved at all.

Why HMAC, not just "guess a UUID"
----------------------------------
The webhook has none of the panel's per-user auth (``ctx.user_id`` there is
the synthetic ``"__webhook__"``, not the caller's real identity -- see
``imperal_sdk.extension.Extension.webhook``'s own docstring). Without a
signature, the link would be "public if you know the generation id", and
generation ids are short/enumerable. Signing binds an expiry into the link
itself, so a leaked/shared link stops working on its own instead of forever.

Why the signing key is a scope="app" secret this extension owns
------------------------------------------------------------------
This key must be stable across a request that MINTS a link (inside the
panel handler) and a later request that VERIFIES it (inside the webhook
handler, quite possibly a different process) -- a random per-process value
would invalidate every link the moment the worker recycles. It is also not
a per-user credential (nobody types it in), so ``scope="app"`` (shared,
extension-owned) is the right shape, not ``scope="user"``.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time

from gemini_config import MEDIA_LINK_SIGNING_SECRET_NAME, MEDIA_LINK_TTL_SECONDS

log = logging.getLogger("gemini.media_link")


def _sig(secret: str, generation_id: str, exp: int) -> str:
    msg = f"{generation_id}.{exp}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]


def sign_media_link(secret: str, generation_id: str, ttl_seconds: int) -> tuple[str, int]:
    """Return (signature, exp_unix_ts) for a generation_id, valid ttl_seconds from now."""
    exp = int(time.time()) + max(1, ttl_seconds)
    return _sig(secret, generation_id, exp), exp


def verify_media_link(secret: str, generation_id: str, exp: int, sig: str) -> bool:
    """Constant-time signature check, AND the link must not have expired yet."""
    if int(time.time()) > exp:
        return False
    expected = _sig(secret, generation_id, exp)
    return hmac.compare_digest(expected, sig)


async def get_or_create_signing_key(ctx) -> str:
    """Read this extension's own signing key, generating it once if missing.

    Best-effort persistence: if ``ctx.secrets.set`` fails (vault hiccup), the
    freshly generated key is still returned and used for THIS request, so
    the current mint/verify pair stays internally consistent even though a
    later request might mint a different key -- degrades to "this specific
    link may not verify after a vault outage", never to a hard failure here.
    """
    try:
        key = await ctx.secrets.get(MEDIA_LINK_SIGNING_SECRET_NAME)
    except Exception as e:  # noqa: BLE001
        log.error("media_link: signing key read failed: %s", e)
        key = None

    if key:
        return key

    new_key = secrets.token_hex(32)
    try:
        await ctx.secrets.set(MEDIA_LINK_SIGNING_SECRET_NAME, new_key)
    except Exception as e:  # noqa: BLE001
        log.warning("media_link: could not persist a new signing key: %s", e)
    return new_key


async def mint_media_link(ctx, generation_id: str, storage_path: str = "") -> str:
    """Build the full ``ctx.webhook_url(\"/media\")?id=...&exp=...&sig=...`` URL.

    One shared entry point for every caller that needs to hand out a link to
    a generation's ORIGINAL file (panel detail view, and chat generation
    handlers offering a preview + full-size link). ``storage_path`` gates
    this the same way every existing caller already does: no saved file, no
    point minting a link nobody can resolve. Best effort throughout -- any
    failure (no webhook_url on this ctx, secrets vault hiccup) returns ``\"\"``
    rather than raising, so a link that can't be minted degrades to simply
    not being offered, never to a broken response.
    """
    if not storage_path:
        return ""
    try:
        secret = await get_or_create_signing_key(ctx)
        sig, exp = sign_media_link(secret, str(generation_id), MEDIA_LINK_TTL_SECONDS)
        base_url = ctx.webhook_url("/media")
        return f"{base_url}?id={generation_id}&exp={exp}&sig={sig}"
    except Exception as e:  # noqa: BLE001
        log.info("media link mint failed for %r: %s", generation_id, e)
        return ""
