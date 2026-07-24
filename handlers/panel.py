"""Declarative UI panel — Gemini Studio.

A center-slot panel with two generation forms (image / video) and a
history list underneath, pulling from the same store collection the
``list_generation_history`` chat function reads. Panel data is refreshed
on every open (``refresh="manual"`` — user re-opens or clicks to refetch;
generation itself triggers a fresh render via the Form's own action).
"""
from __future__ import annotations

import asyncio
import base64
import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION, DEFAULT_HISTORY_LIMIT, MODEL_IMAGE, IMAGE_MODEL_CHOICES

log = logging.getLogger("gemini.panel")

# Why previews are inlined as data: URIs instead of just using the stored url
# -------------------------------------------------------------------------
# Extension storage is served ONLY from the gateway's authenticated internal
# endpoint (/v1/internal/storage/download + Bearer token -- see the SDK's
# StorageClient). There is NO public host that serves /storage/<tenant>/<ext>/...
# to a browser: fetching that path against panel.imperal.io (or imperal.io)
# returns HTTP 404 with the panel's HTML shell, which is exactly why every
# generated image showed as "image unavailable" and every link was dead.
# So the only way to actually display a generation is to ship its bytes.
#
# Inlining bytes is what previously made the panel spin forever, so it is
# strictly bounded here: at most _PREVIEW_MAX_ITEMS images are fetched (in
# parallel, each with its own timeout) and inlining stops once
# _PREVIEW_BYTE_BUDGET of base64 is used. Older entries render as text-only
# cards rather than growing the /call payload without limit.
_PREVIEW_MAX_ITEMS = 4
_PREVIEW_BYTE_BUDGET = 3_000_000
_PREVIEW_DOWNLOAD_TIMEOUT_S = 6.0


async def _preview_data_uri(ctx, doc_data: dict) -> str:
    """Re-download one generation's bytes and return a ``data:`` URI.

    Returns ``""`` on any failure/timeout, and the caller then renders a
    text-only card. Deliberately does NOT fall back to ``doc_data["url"]``:
    that URL is not publicly served (404), so using it would render a
    broken-image icon instead of an honest "no preview" card.
    """
    storage_path = doc_data.get("storage_path")
    if not storage_path:
        return ""
    try:
        raw = await asyncio.wait_for(
            ctx.storage.download(storage_path), timeout=_PREVIEW_DOWNLOAD_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001  (includes asyncio.TimeoutError)
        log.warning("panel: preview download failed for %r: %s", storage_path, e)
        return ""
    mime_type = doc_data.get("mime_type") or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"


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

    all_data = [doc.data for doc in docs]

    # Fetch bytes for at most the newest _PREVIEW_MAX_ITEMS images, all in
    # parallel (never one-await-per-item in the loop, which is what made the
    # panel hang). Videos are never inlined -- far too large for a UI payload.
    candidates = [
        i for i, d in enumerate(all_data)
        if d.get("kind") == "image" and d.get("storage_path")
    ][:_PREVIEW_MAX_ITEMS]
    fetched = await asyncio.gather(
        *(_preview_data_uri(ctx, all_data[i]) for i in candidates)
    ) if candidates else []

    # Apply the byte budget in newest-first order.
    srcs: dict[int, str] = {}
    used = 0
    for idx, src in zip(candidates, fetched):
        if not src:
            continue
        if used + len(src) > _PREVIEW_BYTE_BUDGET:
            log.info("panel: preview byte budget reached, skipping remaining previews")
            break
        srcs[idx] = src
        used += len(src)

    items = []
    for i, d in enumerate(all_data):
        kind = d.get("kind", "")
        prompt = d.get("prompt", "")
        created_at = d.get("created_at", "")
        src = srcs.get(i, "")
        if src:
            preview = ui.Image(src=src, alt=prompt, width="100%", caption=prompt)
        elif kind == "video":
            # No public URL exists and inlining video is not viable, so state
            # that plainly instead of rendering a guaranteed-broken player.
            preview = ui.Text("Video generated — preview not available here.", variant="caption")
        else:
            preview = ui.Text(prompt or "(no prompt)", variant="caption")
        items.append(
            ui.Card(
                title=prompt[:80] or "(no prompt)",
                subtitle=f"{kind} · {d.get('model', '')} · {created_at}",
                content=preview,
            )
        )
    return ui.Stack(children=items, direction="v", gap=3)


async def _quick_stats_panel(ctx) -> dict:
    """Compact left-sidebar summary: connection status + counts + shortcut.

    Registered separately from the main ``gemini_studio`` panel so the
    extension has a permanent left-slot presence (validator recommends at
    least one ``slot="left"`` panel for sidebar navigation) without
    cramming the full generation forms into the narrow sidebar column.
    """
    try:
        key = await ctx.secrets.get("gemini_api_key")
    except Exception:  # noqa: BLE001
        key = None

    image_count = 0
    video_count = 0
    try:
        image_count = await ctx.store.count(GENERATION_LOG_COLLECTION, where={
            "user_id": ctx.user.imperal_id, "kind": "image",
        })
        video_count = await ctx.store.count(GENERATION_LOG_COLLECTION, where={
            "user_id": ctx.user.imperal_id, "kind": "video",
        })
    except Exception as e:  # noqa: BLE001
        log.error("quick panel: count query failed: %s", e)

    status = ui.Badge(
        label="Connected" if key else "No API key",
        color="green" if key else "amber",
    )
    stats = ui.Stats(children=[
        ui.Stat(label="Images", value=image_count, icon="Image"),
        ui.Stat(label="Videos", value=video_count, icon="Video"),
    ])
    open_button = ui.Button(
        label="Open Gemini Studio",
        variant="primary",
        full_width=True,
        icon="Sparkles",
        # Panels are fetched via the /call endpoint as __panel__{panel_id}
        # (see ext.panel()'s docstring in the SDK) -- there is no frontend
        # route for a raw /ext/<app>/<panel_id> URL path, so ui.Navigate(path=...)
        # 404s. ui.Call("__panel__gemini_studio") is the same pattern the
        # working Spotify extension uses for its own center-overlay panel
        # (ui.Call("__panel__spotify_detail", ...)).
        on_click=ui.Call("__panel__gemini_studio"),
    )

    tree = ui.Stack(gap=3, children=[status, stats, open_button])
    return {"ui": tree.to_dict(), "panel_id": "gemini_quick"}


ext.panel(
    "gemini_quick", slot="left", title="Gemini", icon="Sparkles", refresh="manual",
)(_quick_stats_panel)


@ext.panel(
    "gemini_studio", slot="center", title="Gemini Studio", icon="Sparkles",
    refresh="manual", center_overlay=True,
)
async def gemini_studio_panel(ctx, **params) -> ui.UINode:
    """Render the Gemini Studio panel: connection status, generation forms, history."""
    alert = await _connection_alert(ctx)
    history = await _history_section(ctx)

    image_form = ui.Card(
        title="Generate image",
        subtitle="Nano Banana (pick a model below)",
        content=ui.Form(
            children=[
                ui.TextArea(placeholder="Describe the image you want...", param_name="prompt", rows=3),
                ui.Select(
                    options=[
                        {"value": mid, "label": info["label"]}
                        for mid, info in IMAGE_MODEL_CHOICES.items()
                    ],
                    value=MODEL_IMAGE,
                    param_name="model",
                ),
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
