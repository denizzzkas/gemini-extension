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
        "browse and iterate without leaving the platform. One API key, set "
        "once by the extension owner, powers it for everyone — no per-user "
        "setup, no juggling separate accounts."
    ),
    icon="icon.svg",
    version="1.0.0",
    capabilities=["media:generate"],
    actions_explicit=True,
    system=False,
    config_defaults={},
)

chat = ChatExtension(ext)

# One app-level secret: the user's own Gemini API key (from Google AI Studio).
# write_mode="user" — set via the Panel Secrets UI, never written by the
# extension itself. scope="app" — a single key shared by this extension's
# owner across all of their own generations (bring-your-own-key model).
ext.secret(
    name="gemini_api_key",
    description="Your Gemini API key from Google AI Studio (aistudio.google.com/apikey)",
    required=True,
    write_mode="user",
    scope="app",
    max_bytes=256,
)(lambda: None)


@ext.health_check
async def health_check(ctx) -> HealthStatus:
    """App-level health: is a key configured, and is the API reachable?

    Health checks run app-level (no user, no per-user store) — this reports
    only app-scope facts: whether the API key secret is set, plus a bounded
    reachability probe of the Gemini API. Per-user generation status belongs
    in user-facing tools, not here.
    """
    configured = False
    try:
        api_key = await ctx.secrets.get("gemini_api_key")
        configured = bool(api_key)
    except Exception as e:  # noqa: BLE001
        log.error("health_check: could not read gemini_api_key: %s", e)

    api_reachable = False
    try:
        resp = await ctx.http.get(f"{GEMINI_API_BASE}/models", timeout=5)
        # Any HTTP response (even 401/403 for a bad/missing key) means the
        # Gemini API endpoint itself is reachable from our network.
        api_reachable = resp.status_code < 500
    except Exception as e:  # noqa: BLE001
        log.error("health_check: reachability probe failed: %s", e)

    return HealthStatus.ok({"configured": configured, "api_reachable": api_reachable})
