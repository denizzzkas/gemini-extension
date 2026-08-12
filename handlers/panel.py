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

import asyncio
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

# ``gemini_quick`` is a PERMANENT left-slot panel: the host fetches it at
# session-init discovery, so if it hangs, the ENTIRE extension looks dead on
# open ("left panel tries to load and never appears" -- the exact symptom
# this bounds). Its three store/secrets reads each carry their OWN
# independent transport timeout (5s for secrets, a hardcoded 30s for every
# ``ctx.store`` call) with no shared ceiling across them -- run sequentially,
# a merely SLOW (not down) gateway can hold this panel open for ~95s before
# it ever returns a UI tree, which reads as "never renders", not as an
# error. Bounding each with ``asyncio.wait_for`` and running them
# CONCURRENTLY (``asyncio.gather``) caps worst case at one timeout window
# instead of the sum of four, and every branch still degrades to a rendered
# panel (zeroed stats / a retry alert) instead of raising.
_QUICK_PANEL_IO_TIMEOUT_S = 8.0


async def _bounded_history_section(ctx) -> ui.UINode:
    """Bounded wrapper around :func:`_history_section`.

    History now renders ONLY inside ``gemini_studio`` (the centre panel) --
    ``gemini_quick`` (the left sidebar) holds no history at all any more, per
    the user's explicit request that the left column carry generation
    controls exclusively.
    """
    try:
        return await asyncio.wait_for(
            _history_section(ctx), timeout=_QUICK_PANEL_IO_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        log.error("history section timed out or failed: %s", e)
        return ui.Alert(
            title="History unavailable",
            message="Loading your recent generations timed out — try Refresh.",
            type="warn",
        )


async def _quick_panel_key(ctx) -> str | None:
    try:
        return await asyncio.wait_for(
            ctx.secrets.get("gemini_api_key"), timeout=_QUICK_PANEL_IO_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        log.error("quick panel: secrets.get timed out or failed: %s", e)
        return None


async def _quick_panel_count(ctx, kind: str) -> int:
    try:
        return await asyncio.wait_for(
            ctx.store.count(GENERATION_LOG_COLLECTION, where={
                "user_id": ctx.user.imperal_id, "kind": kind,
            }),
            timeout=_QUICK_PANEL_IO_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        log.error("quick panel: count query failed (kind=%s): %s", kind, e)
        return 0


async def _connection_alert(key: str | None) -> ui.UINode:
    """Render the API-key status alert.

    Takes the ALREADY-FETCHED key rather than calling ``ctx.secrets.get``
    again: this used to re-fetch unbounded (no ``asyncio.wait_for``) even
    after ``gemini_quick_panel`` had already fetched the same secret through
    the bounded ``_quick_panel_key`` helper -- a second, unprotected network
    call that could hang the whole panel exactly like the ones already
    fixed above.
    """
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
from handlers.panel_secret import api_key_field  # noqa: E402


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
    # Bounded AND concurrent: worst case is now one _QUICK_PANEL_IO_TIMEOUT_S
    # window, not the sum of three sequential transport timeouts. Each helper
    # already swallows its own failure into a safe default, so `gather`
    # cannot raise here -- the panel always returns a UI tree. No history
    # read here any more -- history lives exclusively in gemini_studio now.
    key, image_count, video_count = await asyncio.gather(
        _quick_panel_key(ctx),
        _quick_panel_count(ctx, "image"),
        _quick_panel_count(ctx, "video"),
    )

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

    children: list[ui.UINode] = [header, stats]
    # The API key field renders ABOVE the generation form, always -- per the
    # user's explicit request once the key moved from app-level to per-user
    # (I-KEY-PER-USER): there is no longer a single app-wide "is a key set"
    # fact to hide this behind, and generation cannot work without it, so it
    # belongs where it is unmissable rather than tucked into an alert.
    children.append(api_key_field(configured=bool(key)))
    if not key:
        children.append(await _connection_alert(key))
    # Image and video generation are switched by a button toggle (Image/Video)
    # rather than two stacked forms or ui.Tabs (reported broken in the real
    # host). Both forms used to be open at once, which made this column a wall
    # of inputs -- and only one of the two is ever being used at a time.
    #
    # This panel renders GENERATION CONTROLS ONLY -- no history section here
    # any more (per the user's explicit request): open Gemini Studio (the
    # centre panel) to browse past generations. Keeping history out of this
    # permanent left column is also what keeps its own I/O bound small and
    # its layout a form, not a growing list.
    children.append(
        generation_tabs(
            await _selected_references(ctx, _param(params, "refs")),
            active=_param(params, "gen_tab") or "image",
            active_model=_param(params, "model") or MODEL_IMAGE,
        ),
    )

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
        # Studio is also the useful default landing view.  It must not depend
        # on the left sidebar being visible: otherwise a user who opens the
        # extension into the centre slot sees only an instruction that points
        # to controls they cannot reach.  This list reads only generation-log
        # metadata and cached thumbnails; it never downloads originals.
        return ui.Page(
            title="Gemini Studio",
            subtitle="Your recent Gemini generations",
            children=[
                ui.Header("Recent generations", level=2),
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
