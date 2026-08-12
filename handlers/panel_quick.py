"""The permanent left-slot panel: generation controls only.

Split out of ``handlers/panel.py`` to stay under the 300-line file limit the
deploy validator enforces (same reason ``handlers/panel_forms.py`` exists).

``gemini_quick`` is a PERMANENT left-slot panel: the host fetches it at
session-init discovery, so if it hangs, the ENTIRE extension looks dead on
open ("left panel tries to load and never appears" -- the exact symptom
this bounds). Its reads each carry their OWN independent transport timeout
(5s for secrets, a hardcoded 30s for every ``ctx.store`` call) with no
shared ceiling across them -- run sequentially, a merely SLOW (not down)
gateway can hold this panel open for a long time before it ever returns a
UI tree, which reads as "never renders", not as an error. Bounding each with
``asyncio.wait_for`` and running them CONCURRENTLY (``asyncio.gather``) caps
worst case at one timeout window instead of the sum of several, and every
branch still degrades to a rendered panel (zeroed stats / a retry alert)
instead of raising.

History does NOT render here any more -- it lives exclusively in
``gemini_studio`` (handlers/panel.py), per the user's explicit request that
the left column carry generation controls only. The ONLY way into that
centre panel from here is the "Open Gemini Studio" button below: without
it, gemini_studio is declared in the manifest but nothing ever calls it,
which for the user is indistinguishable from it not existing at all.
"""
from __future__ import annotations

import asyncio
import logging

from imperal_sdk import ui

from app import ext
from gemini_config import GENERATION_LOG_COLLECTION, MODEL_IMAGE
from handlers.panel_forms import generation_tabs
from handlers.panel_history import _selected_references
from handlers.panel_secret import api_key_field

log = logging.getLogger("gemini.panel_quick")

_QUICK_PANEL_IO_TIMEOUT_S = 8.0


def _param(params: dict, name: str) -> str | None:
    value = params.get(name)
    return value if value else None


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


@ext.panel(
    "gemini_quick", slot="left", title="Gemini", icon="Sparkles",
    refresh="manual", default_width=380, min_width=300,
)
async def gemini_quick_panel(ctx, **params) -> ui.UINode:
    """PRIMARY surface: generation controls only.

    The left slot is "permanent" -- always fetched and rendered -- so
    generation controls live here regardless of whether the host grants the
    centre slot a render path. Full detail on any one generation, and ALL
    history, live in "gemini_studio" (handlers/panel.py) instead.
    """
    # Bounded AND concurrent: worst case is now one _QUICK_PANEL_IO_TIMEOUT_S
    # window, not the sum of three sequential transport timeouts. Each helper
    # already swallows its own failure into a safe default, so `gather`
    # cannot raise here -- the panel always returns a UI tree.
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
    # The ONLY entry point into "gemini_studio" (the centre panel, which now
    # holds ALL history) left after history-cards moved out of this column --
    # without this, gemini_studio is declared in the manifest but nothing
    # ever calls it, so it is unreachable and effectively does not exist for
    # the user. A plain param-less self-call opens the studio's own default
    # landing view (its own recent-generations list).
    open_studio = ui.Button(
        label=f"Open Gemini Studio ({image_count + video_count})",
        icon="LayoutGrid",
        variant="secondary",
        on_click=ui.Call("__panel__gemini_studio"),
    )

    children: list[ui.UINode] = [header, stats, open_studio]
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
