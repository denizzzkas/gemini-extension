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
                                     workshop}). ``center_overlay=True`` is the
                                     declarative replacement, but only helps if
                                     the host reads the flag.
  overlay / bottom / chat-sidebar  : "reserved" — no render path at all.

Design rules that came out of real bugs:
  1. EVERY button must work on a "permanent" slot, and no button may depend on
     a SECOND panel opening -- viewing renders INLINE via a self-call to the
     same panel_id. Betting the UI on the center slot is what left dead buttons.
  2. ONE panel per slot. Two center panels meant the host opened the wrong one
     (param-less) and the useful one was unreachable.

This extension declares NO center-slot panel any more. It had one
("gemini_studio") and the user confirmed in real testing that it never opened
at all -- exactly what rule 1 above predicts for a slot this host does not
grant a render path to. Rather than leave a dead entry point in the UI, it was
removed outright; everything it showed (prompt, reference, download,
regenerate) already renders inline in "gemini_quick", the one permanent left
panel below.
"""
from __future__ import annotations

import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION
from handlers.panel_viewer import CLOSED_SENTINEL, _opened_id, _param

log = logging.getLogger("gemini.panel")

# The history list is strictly ZERO media I/O: downloading bytes while rendering
# it is what made this panel load forever, twice -- render buttons, never bytes.


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
    # Image and video generation are switched by a button toggle (Image/Video)
    # rather than two stacked forms or ui.Tabs (reported broken in the real
    # host). Both forms used to be open at once, which made this column a wall
    # of inputs where the video form pushed history off-screen -- and only one
    # of the two is ever being used at a time.
    children += [
        generation_tabs(
            await _selected_references(ctx, _param(params, "refs")),
            active=_param(params, "gen_tab") or "image",
        ),
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

    # The payload-ceiling probe used to be offered here. It was a developer
    # measuring instrument -- it rendered synthetic images of increasing size
    # so somebody could find the exact byte at which a panel stops rendering
    # -- and it had no meaning for someone who just wants to generate images.
    # Its job is done: the ceiling is known well enough to build against, and
    # previews/downloads no longer depend on guessing it. Removed rather than
    # left as a button that does nothing useful.

    return ui.Stack(children=children, direction="v", gap=3)

# There used to be a second panel here, "gemini_studio" on slot="center".
# REMOVED, not just unlinked: per I-PANEL-RENDERING-CONTRACT a center-slot
# panel only renders when the host grants it a render path (historically a
# hardcoded allowlist this app was never in), so it could never be reached --
# confirmed by the user in real testing (“центральная панель вообще не открывается”). Nothing calls "__panel__gemini_studio" any more (the
# generation detail view already renders inline in gemini_quick, the
# permanent left panel, via detail_content). Leaving a declared-but-dead
# panel around is worse than removing it: it is one more thing imperal.json
# has to describe and one more surface a user can click expecting something
# to happen. detail_view(), its legacy wrapper, is removed too -- see
# handlers/panel_detail.py.
