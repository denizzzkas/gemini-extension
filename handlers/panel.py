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

# How generated media is displayed, and why it works this way
# ----------------------------------------------------------
# Extension storage is readable ONLY via the gateway's authenticated internal
# endpoint (Bearer token -- see the SDK StorageClient). NO public host serves
# /storage/<tenant>/<ext>/...: that path 404s (panel's HTML shell) on both
# panel.imperal.io and imperal.io, verified seconds after upload, so this is
# not link expiry. A url-based <Image> can therefore only render broken, and
# the stored url must never be shown as a viewable link. Sending the bytes as
# a data: URI is the only option (panel CSP allows img-src data:).
#
# BUT bytes must never be fetched while rendering the LIST -- that is what
# made the panel load forever, twice; even 4 downloads under a 3 MB budget
# reproduced it. The list is now strictly zero-I/O (store query only), and one
# image is fetched on demand by the viewer panel below, so a slow storage read
# costs one image instead of the whole panel.
_IMAGE_DOWNLOAD_TIMEOUT_S = 8.0

# How many of the user's recent generations the viewer scans to resolve one id.
# The history list itself only shows DEFAULT_HISTORY_LIMIT, so anything the user
# can actually click is inside this window; the ceiling keeps the lookup bounded.
MAX_LOOKUP_SCAN = 200


async def _image_data_uri(ctx, doc_data: dict) -> str:
    """Download ONE generation's bytes and return a ``data:`` URI.

    Only ever called by the single-image viewer panel (never while rendering
    the history list). Returns ``""`` on failure/timeout so the caller can
    show an honest message. Deliberately does NOT fall back to
    ``doc_data["url"]``: that URL is not publicly served (404), so using it
    would render a broken-image icon instead.
    """
    storage_path = doc_data.get("storage_path")
    if not storage_path:
        return ""
    try:
        raw = await asyncio.wait_for(
            ctx.storage.download(storage_path), timeout=_IMAGE_DOWNLOAD_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001  (includes asyncio.TimeoutError)
        log.warning("panel: image download failed for %r: %s", storage_path, e)
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
    """Render the history list with ZERO media I/O.

    Downloading bytes here (even a few, even with timeouts and a byte budget)
    is what made the panel load forever -- so this function only ever touches
    the store. Each image gets a "View" button that fetches that one image on
    demand via the gemini_image panel.
    """
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
        has_bytes = bool(d.get("storage_path"))

        if kind == "image" and has_bytes:
            body = ui.Button(
                label="View image",
                variant="secondary",
                icon="Image",
                # Fetches THIS image only, in its own /call -- keeps the
                # history payload tiny and a slow read costs one image.
                on_click=ui.Call("__panel__gemini_image", generation_id=doc.id),
            )
        elif kind == "video" and has_bytes:
            # Video bytes are far too large to inline into a UI payload and
            # there is no public URL to link to, so say so plainly rather
            # than render a guaranteed-broken player.
            body = ui.Text("Video saved — not viewable in the panel yet.", variant="caption")
        else:
            body = ui.Text("No stored file for this entry.", variant="caption")

        items.append(
            ui.Card(
                title=prompt[:80] or "(no prompt)",
                subtitle=f"{kind} · {d.get('model', '')} · {d.get('created_at', '')}",
                content=body,
            )
        )
    return ui.Stack(children=items, direction="v", gap=3)


def _param(params: dict, name: str) -> str:
    """Read a panel param, tolerating a nested ``params`` envelope.

    ``ui.Call(fn, **kw)`` serializes to ``{"action","function","params":{...}}``,
    so depending on how the host forwards a panel action the handler may be
    invoked as ``fn(ctx, generation_id=...)`` OR as ``fn(ctx, params={...})``.
    Reading only the flat key silently yields None in the second shape, which
    is indistinguishable from a genuinely missing record downstream.
    """
    value = params.get(name)
    if value:
        return str(value)
    nested = params.get("params")
    if isinstance(nested, dict) and nested.get(name):
        return str(nested[name])
    return ""


async def _find_generation(ctx, generation_id: str):
    """Resolve one generation for THIS user. Returns (doc_or_None, lookup_failed).

    Deliberately resolves via ``ctx.store.query`` rather than
    ``ctx.store.get(collection, id)``: ``get`` does not send ``user_id`` to the
    gateway (only extension/tenant), whereas ``query`` does -- and ``query`` is
    the call the history list already uses successfully. Using the same scoped
    path here removes a whole class of "the list shows it but the viewer cannot
    find it" mismatch. Ownership is still re-checked below, so this never
    widens access.

    ``lookup_failed`` distinguishes "storage errored" from "no such row" so the
    UI can tell the user which one actually happened.
    """
    try:
        page = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id},
            limit=MAX_LOOKUP_SCAN,
        )
    except Exception as e:  # noqa: BLE001
        log.error("image panel: lookup failed for %r: %s", generation_id, e)
        return None, True

    for doc in page.data:
        if doc.id == generation_id:
            # Re-assert ownership: query is user-scoped, but never rely on a
            # single layer for an authorization decision.
            if doc.data.get("user_id") != ctx.user.imperal_id:
                log.warning("image panel: ownership mismatch for %r", generation_id)
                return None, False
            return doc, False

    log.info("image panel: %r not in this user's history", generation_id)
    return None, False


async def _image_viewer_panel(ctx, **params) -> dict:
    """Show ONE generation's image, fetched on demand.

    Separate panel so the history list never pays for media I/O: opening this
    downloads exactly one file, and a failure/timeout degrades to a message
    about that single image instead of hanging the whole studio panel.
    """
    generation_id = _param(params, "generation_id")
    node: ui.UINode

    if not generation_id:
        # The click never carried an id -- a UI/transport problem, NOT a
        # missing record. Saying "not found" here sent me hunting the store
        # for a document that was never asked for.
        log.error("image panel: no generation_id in params (keys=%s)", sorted(params))
        node = ui.Alert(
            title="Nothing to show",
            message=(
                "This viewer opened without a generation id, so there is no "
                "image to load. Re-open Gemini Studio and click View image "
                "on a specific entry."
            ),
            type="warn",
        )
        tree = ui.Page(title="Generated image", children=[node])
        return {"ui": tree.to_dict(), "panel_id": "gemini_image"}

    doc, lookup_failed = await _find_generation(ctx, generation_id)

    if doc is None:
        node = ui.Alert(
            title="Could not load that generation",
            message=(
                "Reading it from storage failed just now — please try again."
                if lookup_failed else
                "That entry is no longer in your generation history."
            ),
            type="warn",
        )
    else:
        src = await _image_data_uri(ctx, doc.data)
        prompt = doc.data.get("prompt", "")
        if src:
            node = ui.Stack(gap=2, children=[
                ui.Image(src=src, alt=prompt, width="100%", caption=prompt),
            ])
        else:
            node = ui.Alert(
                title="Image unavailable",
                message=(
                    "The stored file for this generation could not be read. "
                    "Older entries created before files were kept may no longer exist."
                ),
                type="warn",
            )

    tree = ui.Page(title="Generated image", children=[node])
    return {"ui": tree.to_dict(), "panel_id": "gemini_image"}


ext.panel(
    "gemini_image", slot="center", title="Generated image", icon="Image",
    refresh="manual", center_overlay=True,
)(_image_viewer_panel)


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
