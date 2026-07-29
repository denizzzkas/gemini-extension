"""Declarative UI panels — Gemini.

WHY THE UI IS SHAPED LIKE THIS (read before moving anything)
-----------------------------------------------------------
Per I-PANEL-RENDERING-CONTRACT (imperal_sdk/types/contributions.py) and
docs.imperal.io/en/concepts/panels:

  left / right -> "permanent"      : fetched at session-init discovery and
                                     ALWAYS rendered as a column.
  center       -> "center-overlay" : fetched ON DEMAND via a __panel__<id>
                                     action. Declaring ``center_overlay=True``
                                     (SDK v4.1.8+, this app runs on 5.9.x) is
                                     what makes the kernel publish
                                     ``center_overlay: true`` into the panel's
                                     manifest entry -- the frontend then reads
                                     that flag declaratively instead of
                                     consulting a hardcoded panel_id list. This
                                     is a REAL, supported render path, not a
                                     historical dead end -- see the SDK's own
                                     ``ext.panel()`` docstring.
  overlay / bottom / chat-sidebar  : "reserved" — no render path at all.

Design rules that came out of real bugs:
  1. A button that opens a SECOND panel must target that panel EXPLICITLY
     (``ui.Call("__panel__<other_id>", ...)``) -- self-calls only make sense
     for actions that stay inside the current panel (Refresh, pick-a-
     reference). Mixing the two up is what made "View image" look dead.
  2. ONE panel per slot. Two center panels meant the host opened the wrong
     one (param-less) and the useful one was unreachable -- so ``gemini_image``
     (handlers/panel_viewer.py) stays deliberately UNREGISTERED; its helpers
     are reused, the panel itself is not declared.
  3. A center panel must never do unbounded media I/O with no timeout: THAT
     (not the slot itself) is what actually produced the "infinite loading"
     symptom before. ``handlers/image_loader.py`` now bounds every storage
     download to ``_IMAGE_DOWNLOAD_TIMEOUT_S`` (25s) via ``asyncio.wait_for``,
     so this panel can safely do the one storage read a detail view needs.

This extension previously removed the "gemini_studio" center panel outright,
on the theory that slot="center" could never render for this host. That
theory does not hold up against the SDK's own docs (center_overlay=True is
exactly the mechanism meant to fix this) -- the earlier failure was far more
likely the missing download timeout above, which is fixed now. So the panel
is restored, and it now carries the FULL generation detail (prompt, copy
button, reference, download, regenerate) that had been crammed into every
opened card in the permanent left panel -- decluttering "gemini_quick" back
to what it is meant to be: a compact list plus the generation forms.
"""
from __future__ import annotations

import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION, MODEL_IMAGE
from handlers.panel_detail import detail_content, load_detail
from handlers.panel_viewer import (
    CLOSED_SENTINEL, _find_generation, _opened_id, _param,
)

log = logging.getLogger("gemini.panel")

# The history list is strictly ZERO storage I/O: downloading bytes while
# rendering it is what made this panel look dead on load before -- render
# cached-preview thumbnails and buttons, never a fresh download.


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


from handlers.panel_forms import generation_tabs  # noqa: E402
from handlers.panel_history import (  # noqa: E402
    _history_section, _selected_references,
)


@ext.panel(
    "gemini_quick", slot="left", title="Gemini", icon="Sparkles",
    refresh="manual", default_width=380, min_width=300,
)
async def gemini_quick_panel(ctx, **params) -> ui.UINode:
    """PRIMARY surface: generate, and browse a compact history list.

    The left slot is "permanent" -- always fetched and rendered -- so
    generation and the history list live here regardless of whether the host
    grants the centre slot a render path. Full detail on any one generation
    now opens in "gemini_studio" (below) instead of expanding inline, so this
    column stays a list, not a wall of expanded cards.
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

    history = await _history_section(ctx, "gemini_quick")

    children: list[ui.UINode] = [header, stats]
    if not key:
        children.append(await _connection_alert(ctx))
    # Image and video generation are switched by a button toggle (Image/Video)
    # rather than two stacked forms or ui.Tabs (reported broken in the real
    # host). Both forms used to be open at once, which made this column a wall
    # of inputs where the video form pushed history off-screen -- and only one
    # of the two is ever being used at a time.
    children += [
        generation_tabs(
            await _selected_references(ctx, _param(params, "refs")),
            active=_param(params, "gen_tab") or "image",
            active_model=_param(params, "model") or MODEL_IMAGE,
        ),
        ui.Header("Recent generations", level=3),
        ui.Button(
            label="Refresh",
            variant="ghost",
            icon="RefreshCw",
            on_click=ui.Call("__panel__gemini_quick"),
        ),
        history,
    ]

    return ui.Stack(children=children, direction="v", gap=3)


@ext.panel(
    "gemini_studio", slot="center", title="Gemini Studio", icon="Sparkles",
    refresh="manual", center_overlay=True,
)
async def gemini_studio_panel(ctx, **params) -> ui.UINode:
    """The centre surface: ONE opened generation, in full detail.

    This is the ONLY place a generation's full detail (prompt, copy button,
    reference, download, regenerate) renders. Bounded to a single storage
    read via :func:`load_detail`, which goes through
    ``handlers/image_loader._load_image`` -- itself bounded by
    ``_IMAGE_DOWNLOAD_TIMEOUT_S`` so a slow/failing download degrades this one
    view instead of spinning forever.
    """
    opened_id = _opened_id(params)

    if not opened_id:
        return ui.Page(
            title="Gemini Studio",
            subtitle="Pick a generation from the Gemini panel to open it here",
            children=[
                ui.Empty(message=(
                    "Nothing open yet — click \"Image info\" or \"Video info\" "
                    "on any entry in the Gemini panel's history to see it in "
                    "full here, with its prompt, reference, download and "
                    "regenerate actions."
                )),
            ],
        )

    doc, lookup_failed = await _find_generation(ctx, opened_id)
    if doc is None:
        return ui.Page(
            title="Gemini Studio",
            children=[ui.Alert(
                title="Could not open that generation",
                message=(
                    "Reading it failed just now — try again."
                    if lookup_failed else
                    "That generation no longer exists, or it belongs to another account."
                ),
                type="warn",
            )],
        )

    detail = await load_detail(ctx, doc)

    return ui.Page(
        title="Gemini Studio",
        subtitle="Generated with your own Gemini API key",
        children=[
            ui.Button(
                label="Close",
                variant="ghost",
                icon="X",
                on_click=ui.Call(
                    "__panel__gemini_studio", generation_id=CLOSED_SENTINEL,
                ),
            ),
            *detail_content(
                doc,
                image_src=detail["image_src"],
                fail_reason=detail["fail_reason"],
                raw_original=detail["raw_original"],
                references=detail["references"],
                is_preview=detail["is_preview"],
            ),
        ],
    )
