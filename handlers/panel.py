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
from handlers.media import _absolute_url

log = logging.getLogger("gemini.panel")

# Hard per-image cap on ctx.storage.download() while rendering the panel.
# Without this a single slow/hanging download (the storage client itself
# uses a 60s timeout) can make the WHOLE panel look like it's loading
# forever -- this bounds the worst case for any one image and we still
# fall back to the (possibly stale) signed url on timeout, never blocking
# panel render on it.
_PREVIEW_DOWNLOAD_TIMEOUT_S = 6.0


async def _image_preview_src(ctx, doc_data: dict) -> str:
    """Prefer re-downloading the saved bytes over the stored ``url``.

    ``ctx.storage.upload()`` returns a *signed* URL (see the SDK's own
    FileInfo schema docstring) -- it can and does expire, which is the
    root cause of "image unavailable" reports for generations that were
    working right after creation. ``storage_path`` is a stable internal
    path (not signed/time-limited), so re-downloading via
    ``ctx.storage.download()`` and embedding as a data: URI sidesteps the
    expiry entirely. Falls back to the (possibly stale) url if the
    download fails OR is slow (see _PREVIEW_DOWNLOAD_TIMEOUT_S) -- a
    broken/stale preview is better than a hung panel.
    """
    storage_path = doc_data.get("storage_path")
    if storage_path:
        try:
            raw = await asyncio.wait_for(
                ctx.storage.download(storage_path), timeout=_PREVIEW_DOWNLOAD_TIMEOUT_S,
            )
            mime_type = doc_data.get("mime_type") or "image/png"
            b64 = base64.b64encode(raw).decode()
            return f"data:{mime_type};base64,{b64}"
        except asyncio.TimeoutError:
            log.warning("panel: download of %r timed out after %ss, falling back to url", storage_path, _PREVIEW_DOWNLOAD_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            log.warning("panel: could not re-download %r for preview, falling back to url: %s", storage_path, e)
    return _absolute_url(doc_data.get("url", ""))


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

    # Resolve all image previews CONCURRENTLY, not one ctx.storage.download()
    # await at a time in the loop -- sequential awaits here is what made the
    # panel look like it hangs forever with a full history (up to
    # DEFAULT_HISTORY_LIMIT downloads back-to-back). _image_preview_src also
    # has its own per-item timeout, so one slow file can't block the rest.
    all_data = [doc.data for doc in docs]
    preview_tasks = [
        _image_preview_src(ctx, d) if d.get("kind") == "image" else None
        for d in all_data
    ]
    resolved_previews = await asyncio.gather(*(t for t in preview_tasks if t is not None))
    previews_by_index = {}
    _it = iter(resolved_previews)
    for i, t in enumerate(preview_tasks):
        if t is not None:
            previews_by_index[i] = next(_it)

    items = []
    for i, d in enumerate(all_data):
        kind = d.get("kind", "")
        prompt = d.get("prompt", "")
        created_at = d.get("created_at", "")
        if kind == "image":
            # Embed as a data: URI re-downloaded from storage_path instead of
            # the stored (signed, expiring) url -- see _image_preview_src.
            src = previews_by_index.get(i, "")
            preview = ui.Image(src=src, alt=prompt, width="100%", caption=prompt) if src else ui.Text(prompt, variant="caption")
        elif kind == "video":
            # Videos aren't re-embedded as data: URIs (would bloat the panel
            # payload) -- still subject to the same signed-url expiry as a
            # known limitation, tracked separately from the image fix.
            url = _absolute_url(d.get("url", ""))
            preview = ui.Video(src=url, caption=prompt) if url else ui.Text(prompt, variant="caption")
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
