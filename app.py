"""Gemini extension — Extension setup, secrets, lifecycle, health check."""
from __future__ import annotations

import logging

from imperal_sdk import Extension
from imperal_sdk.chat import ChatExtension
from imperal_sdk.types.health import HealthStatus

from gemini_config import GEMINI_API_BASE

log = logging.getLogger("gemini")

# ─── Extension ────────────────────────────────────────────────────────────── #

ext = Extension(
    "gemini",
    version="1.0.6",
    capabilities=["media:generate"],
    config_defaults={},
    display_name="Gemini AI",
    description=(
        "Turn words into pictures and video with your own Gemini API key. "
        "Browse saved generations and reuse images as references."
    ),
    icon="icon.svg",
    actions_explicit=True,
)

# Explicit arguments work on both the older Python 3.11 host SDK and the
# current development SDK. A boot failure here prevents *every* panel
# decorator from running, leaving the app as chat-only.
chat = ChatExtension(
    ext,
    tool_name="tool_gemini_chat",
    description="Gemini AI image and video generation assistant",
)

# Per-user secret: each user brings and stores their own Gemini API key.
# ``Extension.secret`` was added after the 4.1 SDK line.  Some host workers
# still import with that line; calling the missing method aborts app import
# before the panel decorators in bootstrap.py can run, leaving the extension
# chat-only.  Newer workers receive the full declared secret contract; older
# workers continue booting and can still render the Gemini panels.
if hasattr(ext, "secret"):
    # write_mode="both" (not "user"): the value must be settable from BOTH the
    # platform's built-in Secrets tab AND this extension's own left-panel field
    # (handlers/panel.py -- above the generation form, per the user's explicit
    # request once the key became per-user). write_mode="user" forbids
    # ctx.secrets.set() from extension code entirely (SecretClient.set() raises
    # SecretWriteForbidden) -- it would make the in-panel field decorative.
    ext.secret(
        name="gemini_api_key",
        description="Your Gemini API key from Google AI Studio (aistudio.google.com/apikey)",
        required=True,
        write_mode="both",
        scope="user",
        max_bytes=256,
    )(lambda: None)
else:
    log.warning(
        "SDK has no Extension.secret; skipping secret declaration to keep Gemini UI available"
    )


@ext.on_install
async def on_install(ctx) -> None:
    """Log first-time install. No app-level state to initialize -- the only
    per-extension config is the user's own gemini_api_key, set separately
    via the Panel Secrets UI (scope="user"), not here."""
    user_id = ctx.user.imperal_id if hasattr(ctx, "user") and ctx.user else "unknown"
    log.info("Gemini extension installed for user %s", user_id)


@ext.health_check
async def health_check(ctx) -> HealthStatus:
    """App-level health: is the Gemini API reachable?

    Health checks run app-level (no user, no per-user store) — and since
    ``gemini_api_key`` is now ``scope="user"`` (I-KEY-PER-USER, each user
    brings their own key), there is no single app-wide "is a key configured"
    fact left to report honestly here (I-HEALTH-CTX-HONEST). That per-user
    question belongs in ``check_gemini_connection`` instead. This probe
    reports only the one fact that genuinely is app-level: whether the
    Gemini API endpoint itself is reachable from our network at all.
    """
    api_reachable = False
    try:
        resp = await ctx.http.get(f"{GEMINI_API_BASE}/models", timeout=5)
        # Any HTTP response (even 401/403, since we probe with no key) means
        # the Gemini API endpoint itself is reachable from our network.
        api_reachable = resp.status_code < 500
    except Exception as e:  # noqa: BLE001
        log.error("health_check: reachability probe failed: %s", e)

    return HealthStatus.ok({"api_reachable": api_reachable})


# Some hosts load ``app.py`` directly while the CLI loads ``main.py``. Keep
# handler registration here so both supported entrypoints publish the same UI
# contributions; without it a direct app import exposes chat but no panels.
import bootstrap  # noqa: E402,F401
