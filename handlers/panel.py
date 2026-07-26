"""Declarative UI panels — Gemini.

WHY THE UI IS SHAPED LIKE THIS (read before moving anything)
-----------------------------------------------------------
Per I-PANEL-RENDERING-CONTRACT (imperal_sdk/types/contributions.py):

  left / right -> "permanent"      : fetched at session-init discovery and
                                     ALWAYS rendered as a column.
  center       -> "center-overlay" : fetched ON DEMAND via a __panel__<id>
                                     action, historically only when panel_id
                                     sat in the host's hardcoded allowlist
                                     ({compose, email_viewer, editor,
                                     workshop}). ``center_overlay=True`` is
                                     the declarative replacement, but it only
                                     helps if the host reads the flag.
  overlay / bottom / chat-sidebar -> "reserved": no render path at all.

Design rules that came out of real bugs:
  1. EVERY button must work on a "permanent" slot, and no button may depend on
     a SECOND panel opening -- viewing renders INLINE via a self-call to the
     same panel_id. Betting the UI on the center slot is what left dead buttons.
  2. ONE panel per slot. Two center panels meant the host opened the wrong one
     (param-less) and the useful one was unreachable.
"""
from __future__ import annotations

import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION, PANEL_HISTORY_LIMIT
from handlers.media import newest_first
from handlers.probe import probe_section, probe_toggle_button
from handlers.panel_viewer import (
    CLOSED_SENTINEL, FAIL_NONE, _failure_message, _find_generation, _load_image,
    _opened_id,
)

log = logging.getLogger("gemini.panel")

# The history list is strictly ZERO media I/O: downloading bytes while
# rendering it is what made this panel load forever, twice. Render buttons,
# never bytes -- one image is fetched only when the user asks for it.


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


from handlers.panel_forms import _image_form, _video_form  # noqa: E402


def _entry_card(
    doc, panel_id: str, opened_id: str, image_src: str, fail_reason: str = FAIL_NONE,
) -> ui.UINode:
    """One history row. The View/Hide button re-renders THIS panel.

    ``on_click`` targets ``panel_id`` -- the panel the card is already being
    rendered in -- so the click never depends on a different panel being
    granted a render path. That indirection is exactly what silently failed
    before.
    """
    d = doc.data
    kind = d.get("kind", "")
    prompt = d.get("prompt", "")
    has_bytes = bool(d.get("storage_path"))
    is_open = doc.id == opened_id

    children: list[ui.UINode] = []

    if kind == "image" and has_bytes:
        if is_open and image_src:
            children.append(ui.Image(src=image_src, alt=prompt[:120], width="100%"))
            # The FULL prompt, not the 80-char card title. Seeing exactly what
            # produced an image is the whole point of a generation history;
            # a truncated title made real prompts (often 700+ chars) unreadable.
            children.append(ui.Text("Prompt", variant="caption"))
            children.append(ui.Text(prompt or "(no prompt)"))
            children.append(ui.Button(
                label="Hide",
                variant="secondary",
                icon="ChevronUp",
                # Must OVERWRITE generation_id, not omit it: the host merges a
                # re-fetch's params INTO the accumulated ones, so a param-less
                # call leaves the image open. That was the "Hide" bug.
                on_click=ui.Call(
                    f"__panel__{panel_id}", generation_id=CLOSED_SENTINEL,
                ),
            ))
        elif is_open:
            # Asked for, but the bytes could not be fetched -- say WHY here, in
            # place. One generic message hid four different causes and is why
            # "one opens, another does not" stayed a mystery for so long.
            children.append(ui.Text(
                _failure_message(fail_reason), variant="caption",
            ))
            children.append(ui.Button(
                label="Retry",
                variant="secondary",
                icon="RefreshCw",
                on_click=ui.Call(f"__panel__{panel_id}", generation_id=doc.id),
            ))
        else:
            children.append(ui.Button(
                label="View image",
                variant="secondary",
                icon="Image",
                on_click=ui.Call(f"__panel__{panel_id}", generation_id=doc.id),
            ))
    elif kind == "video" and has_bytes:
        # Video bytes are far too large to inline and there is no public URL,
        # so state that plainly rather than render a guaranteed-broken player.
        children.append(ui.Text(
            "Video saved — not viewable in the panel yet.", variant="caption",
        ))
    else:
        children.append(ui.Text("No stored file for this entry.", variant="caption"))

    title = prompt[:80] or "(no prompt)"
    if len(prompt) > 80:
        title += "…"
    return ui.Card(
        title=title,
        subtitle=f"{kind} · {d.get('model', '')} · {d.get('created_at', '')}",
        content=ui.Stack(children=children, direction="v", gap=2),
    )


async def _history_section(ctx, panel_id: str, opened_id: str = "") -> ui.UINode:
    """Render the history list with ZERO media I/O, except one opened image.

    Only the entry the user explicitly clicked costs a storage read, so a slow
    read degrades that single card instead of hanging the whole panel.
    """
    try:
        page = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id},
            limit=PANEL_HISTORY_LIMIT,
        )
        # Explicit ordering: the backend does not promise one, so without this
        # a capped page can silently omit recent generations.
        docs = newest_first(page.data)
    except Exception as e:  # noqa: BLE001
        log.error("panel: history query failed: %s", e)
        return ui.Alert(
            title="Could not load history",
            message="Reading your generations failed just now — try again.",
            type="warn",
        )

    if not docs:
        return ui.Empty(message="No generations yet — try the form above.")

    image_src = ""
    fail_reason = FAIL_NONE
    if opened_id:
        target = next((d for d in docs if d.id == opened_id), None)
        if target is None:
            # Clicked entry is outside the listed window -- resolve it directly.
            target, _ = await _find_generation(ctx, opened_id)
        if target is not None:
            image_src, fail_reason = await _load_image(ctx, target.data)

    return ui.Stack(
        children=[
            _entry_card(d, panel_id, opened_id, image_src, fail_reason)
            for d in docs
        ],
        direction="v",
        gap=3,
    )


@ext.panel(
    "gemini_quick", slot="left", title="Gemini", icon="Sparkles",
    refresh="manual", default_width=380, min_width=300,
)
async def gemini_quick_panel(ctx, **params) -> ui.UINode:
    """PRIMARY surface: everything works here, on a permanent slot.

    The left slot is "permanent" in I-PANEL-RENDERING-CONTRACT -- always
    fetched and rendered -- so this panel does not depend on the host's
    center-overlay allowlist. Generation forms, history and inline image
    viewing all live here, which is why every button in it actually fires.
    """
    opened_id = _opened_id(params)

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

    header = ui.Stack(direction="h", gap=2, children=[
        ui.Badge(
            label="Connected" if key else "No API key",
            color="green" if key else "amber",
        ),
    ])
    stats = ui.Stats(children=[
        ui.Stat(label="Images", value=image_count, icon="Image"),
        ui.Stat(label="Videos", value=video_count, icon="Video"),
    ])

    history = await _history_section(ctx, "gemini_quick", opened_id)

    children: list[ui.UINode] = [header, stats]
    if not key:
        children.append(await _connection_alert(ctx))
    children += [
        _image_form(),
        ui.Header("Recent generations", level=3),
        # Refresh also collapses an open image, so it carries the same reset
        # sentinel -- a bare call would inherit the accumulated generation_id.
        ui.Button(
            label="Refresh",
            variant="ghost",
            icon="RefreshCw",
            on_click=ui.Call(
                "__panel__gemini_quick", generation_id=CLOSED_SENTINEL,
            ),
        ),
        history,
    ]

    # Diagnostic section, off unless explicitly opened -- it never competes
    # with the real content for the single left slot.
    probe = probe_section("gemini_quick", params)
    children.append(probe_toggle_button("gemini_quick", probe is not None))
    if probe is not None:
        children.append(probe)

    return ui.Stack(children=children, direction="v", gap=3)


@ext.panel(
    "gemini_studio", slot="center", title="Gemini Studio", icon="Sparkles",
    refresh="manual", center_overlay=True,
)
async def gemini_studio_panel(ctx, **params) -> ui.UINode:
    """Wide bonus surface for hosts that honour center_overlay.

    Deliberately NOT the only way to reach anything: the left panel above is
    fully self-sufficient, so if this surface never opens the extension is
    still completely usable. Image viewing here is also inline (self-call),
    not a hop to another panel.
    """
    opened_id = _opened_id(params)

    alert = await _connection_alert(ctx)
    history = await _history_section(ctx, "gemini_studio", opened_id)

    return ui.Page(
        title="Gemini Studio",
        subtitle="Generate images and videos with your own Gemini API key",
        children=[
            alert,
            ui.Grid(children=[_image_form(), _video_form()], columns=2, gap=3),
            ui.Header("Recent generations", level=3),
            ui.Button(
                label="Refresh",
                variant="ghost",
                icon="RefreshCw",
                on_click=ui.Call(
                    "__panel__gemini_studio", generation_id=CLOSED_SENTINEL,
                ),
            ),
            history,
        ],
    )
