"""Declarative UI panels — Gemini.

WHY THE UI IS SHAPED LIKE THIS (read before moving anything)
-----------------------------------------------------------
Per I-PANEL-RENDERING-CONTRACT (imperal_sdk/types/contributions.py) and
docs.imperal.io/en/concepts/panels:

  left / right -> "permanent"      : fetched at session-init discovery and
                                     ALWAYS rendered as a column.
  center       -> "center-overlay" : fetched on demand through a
                                     ``__panel__gemini_studio`` call. The
                                     declarative ``center_overlay=True`` flag
                                     tells the host to render Studio over the
                                     chat while leaving the permanent Gemini
                                     sidebar visible.
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
  4. ``gemini_quick`` (the left sidebar: generation controls, the API key
     field, and the ONLY button that opens Studio) now lives in its own
     module, ``handlers/panel_quick.py`` -- split out to stay under the
     300-line file-size limit the deploy validator enforces. It is
     re-exported below unchanged so every existing ``from handlers.panel
     import gemini_quick_panel`` keeps working.

This extension previously removed the "gemini_studio" center panel outright,
on the theory that slot="center" could never render for this host. That
theory does not hold up against the SDK's own docs (center_overlay=True is
exactly the mechanism meant to fix this) -- the earlier failure was far more
likely the missing download timeout above, which is fixed now. So the panel
is restored, and it carries the FULL generation detail (prompt, copy button,
reference, download, regenerate) AND all history (3-per-row grid) -- per the
user's explicit request that the left column carry generation controls only.
"""
from __future__ import annotations

import asyncio
import logging

from imperal_sdk import ui

from app import ext
from handlers.panel_detail import detail_content, load_detail
from handlers.panel_history import _history_section
from handlers.panel_quick import gemini_quick_panel  # noqa: F401  (re-export)
from handlers.panel_viewer import (
    CLOSED_SENTINEL, _find_generation, _opened_id,
)

log = logging.getLogger("gemini.panel")

# History now renders ONLY inside this centre panel -- ``gemini_quick`` (the
# left sidebar, handlers/panel_quick.py) holds no history at all any more,
# per the user's explicit request that the left column carry generation
# controls exclusively. This bound keeps a slow/failing store read from
# hanging the studio's default landing view forever.
_STUDIO_HISTORY_TIMEOUT_S = 8.0


async def _bounded_history_section(ctx) -> ui.UINode:
    """Bounded wrapper around :func:`_history_section`."""
    try:
        return await asyncio.wait_for(
            _history_section(ctx), timeout=_STUDIO_HISTORY_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        log.error("history section timed out or failed: %s", e)
        return ui.Alert(
            title="History unavailable",
            message="Loading your recent generations timed out — try Refresh.",
            type="warn",
        )


@ext.panel(
    "gemini_studio", slot="center", title="Gemini Studio", icon="Sparkles",
    refresh="manual", center_overlay=True,
)
async def gemini_studio_panel(ctx, **params) -> ui.UINode:
    """The centre surface: ALL history, plus ONE opened generation in full.

    This is the ONLY place a generation's full detail (prompt, copy button,
    reference, download, regenerate) renders, and the ONLY place history
    renders. Bounded to a single storage read via :func:`load_detail`, which
    goes through ``handlers/image_loader._load_image`` -- itself bounded by
    ``_IMAGE_DOWNLOAD_TIMEOUT_S`` so a slow/failing download degrades this
    one view instead of spinning forever.
    """
    opened_id = _opened_id(params)

    if not opened_id:
        # Studio is also the useful default landing view.  It must not depend
        # on the left sidebar being visible: otherwise a user who opens the
        # extension into the centre slot sees only an instruction that points
        # to controls they cannot reach.  This list reads only generation-log
        # metadata and cached thumbnails; it never downloads originals.
        return ui.Stack(
            direction="v",
            gap=3,
            children=[
                ui.Header("Gemini Studio", level=2, subtitle="Your recent Gemini generations"),
                ui.Button(
                    label="Refresh",
                    variant="ghost",
                    icon="RefreshCw",
                    on_click=ui.Call("__panel__gemini_studio"),
                ),
                await _bounded_history_section(ctx),
            ],
        )

    doc, lookup_failed = await _find_generation(ctx, opened_id)
    if doc is None:
        return ui.Stack(
            direction="v",
            gap=3,
            children=[
                ui.Header("Gemini Studio", level=2),
                ui.Alert(
                    title="Could not open that generation",
                    message=(
                        "Reading it failed just now — try again."
                        if lookup_failed else
                        "That generation no longer exists, or it belongs to another account."
                    ),
                    type="warn",
                ),
            ],
        )

    detail = await load_detail(ctx, doc)

    return ui.Stack(
        direction="v",
        gap=3,
        children=[
            ui.Header("Gemini Studio", level=2, subtitle="Generated with your own Gemini API key"),
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
                media_link_url=detail.get("media_link_url", ""),
            ),
        ],
    )
