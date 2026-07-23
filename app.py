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
    display_name="Gemini AI",
    description=(
        "Turn words into pictures and video, right inside your chat. Gemini AI "
        "brings Google's Nano Banana Pro (studio-grade image generation and "
        "editing) and Gemini Omni Flash (fast text-to-video) straight into "
        "Imperal — just describe what you want and watch it appear. Every "
        "generation is saved to a searchable history with instant previews, "
        "and the Gemini Studio panel gives you a dedicated space to create, "
        "browse and iterate without leaving the platform. Bring your own "
        "Gemini API key — each user connects their own privately in Secrets, "
        "so every generation runs on your own Google account, your own quota."
    ),
    icon="icon.svg",
    version="1.0.0",
    capabilities=["media:generate"],
    actions_explicit=True,
    system=False,
    config_defaults={},
)

chat = ChatExtension(ext)

# Per-user secret: each user brings and stores their own Gemini API key.
# write_mode="user" — set via the Panel Secrets UI, never written by the
# extension itself. scope="user" — every user's key is private to them
# (I-KEY-PER-USER); no shared/app-wide key, no cross-user billing surprises.
ext.secret(
    name="gemini_api_key",
    description="Your Gemini API key from Google AI Studio (aistudio.google.com/apikey)",
    required=True,
    write_mode="user",
    scope="user",
    max_bytes=256,
)(lambda: None)


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
