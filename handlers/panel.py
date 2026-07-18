"""Declarative UI panel — Gemini Studio.

A center-slot panel with two generation forms (image / video) and a
history list underneath, pulling from the same store collection the
``list_generation_history`` chat function reads. Panel data is refreshed
on every open (``refresh="manual"`` — user re-opens or clicks to refetch;
generation itself triggers a fresh render via the Form's own action).
"""
from __future__ import annotations

import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION, DEFAULT_HISTORY_LIMIT

log = logging.getLogger("gemini.panel")


async def _connection_alert(ctx) -> ui.UINode:
    try:
        key = await ctx.secrets.get("gemini_api_key")
    except Exception:  # noqa: BLE001
        key = None

    if key:
        return ui.Alert(
            title="Connected",
            message="Gemini API key is configured. Generate away!",
            type="success",
        )
    return ui.Alert(
        title="No API key yet",
        message=(
            "Add your Gemini API key from Google AI Studio "
            "(aistudio.google.com/apikey) in the Secrets tab to start generating."
        ),
        type="warn",
    )


async def _history_section(ctx) -> ui.UINode:
    try:
        page = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id},
            limit=DEFAULT_HISTORY_LIMIT,
        )
        docs = page.data
    except Exception as e:  # noqa: BLE001
        log.error("panel: history query failed: %s", e)
        docs = []

    if not docs:
        return ui.Empty(message="No generations yet — try the forms above.")

    items = []
    for doc in docs:
        d = doc.data
        kind = d.get("kind", "")
        prompt = d.get("prompt", "")
        url = d.get("url", "")
        created_at = d.get("created_at", "")
        if url and kind == "image":
            preview = ui.Image(src=url, alt=prompt, width="100%", caption=prompt)
        elif url and kind == "video":
            preview = ui.Video(src=url, caption=prompt)
        else:
            preview = ui.Text(prompt, variant="caption")
        items.append(
            ui.Card(
                title=prompt[:80] or "(no prompt)",
                subtitle=f"{kind} · {d.get('model', '')} · {created_at}",
                content=preview,
            )
        )
    return ui.Stack(children=items, direction="v", gap=3)


@ext.panel("gemini_studio", slot="center", title="Gemini Studio", icon="Sparkles", refresh="manual")
async def gemini_studio_panel(ctx, **params) -> ui.UINode:
    """Render the Gemini Studio panel: connection status, generation forms, history."""
    alert = await _connection_alert(ctx)
    history = await _history_section(ctx)

    image_form = ui.Card(
        title="Generate image",
        subtitle="Nano Banana Pro (gemini-3-pro-image)",
        content=ui.Form(
            children=[
                ui.TextArea(placeholder="Describe the image you want...", param_name="prompt", rows=3),
            ],
            action="generate_image",
            submit_label="Generate image",
        ),
    )

    video_form = ui.Card(
        title="Generate video",
        subtitle="Gemini Omni Flash (gemini-omni-flash-preview)",
        content=ui.Form(
            children=[
                ui.TextArea(placeholder="Describe the video you want...", param_name="prompt", rows=3),
            ],
            action="generate_video",
            submit_label="Generate video",
        ),
    )

    return ui.Page(
        title="Gemini Studio",
        subtitle="Generate images and videos with your own Gemini API key",
        children=[
            alert,
            ui.Grid(children=[image_form, video_form], columns=2, gap=3),
            ui.Header("Recent generations", level=3),
            history,
        ],
    )
