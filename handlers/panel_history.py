"""The generation history list: compact cards, and the reference choices they feed.

Split out of ``handlers/panel.py`` so that file declares panels while this one
renders the list of what you have made.

The rule the card layout is built around: the list does ZERO media I/O beyond
a CACHED preview thumbnail (the same ~20 KB base64 already stored at
generation time, see core/preview.py) -- no storage download happens just to
render the list, so a slow or failing read can never hang the whole panel.

Full detail moved back to the centre ("Image info" / "Video info")
-------------------------------------------------------------------
This used to expand INLINE, in this same card, when opened -- which is
exactly what made the left column "overloaded" (the user's own word): one
opened image pushed prompt, copy button, reference thumbnail, metadata,
download and regenerate all into the same narrow sidebar as the generation
forms and the rest of the history. That inline-expansion design was a
workaround for a real but different problem (the centre slot spinning
forever on a slow, timeout-less storage download -- since fixed in
handlers/image_loader.py) that got blamed on the WRONG cause ("the host will
never render slot='center'"), which is contradicted by both the SDK's own
``ext.panel()`` docstring and docs.imperal.io/en/concepts/panels: a panel
declared with ``center_overlay=True`` on ``slot="center"`` renders like any
other panel once the flag is set (SDK v4.1.8+, this app runs 5.9.12+).

So each card here is compact again: a cached thumbnail (if one exists),
one button that opens the full detail in the centre panel
(``handlers/panel.py::gemini_studio_panel``), and, for images, "Use as
reference". No card ever expands inline any more -- there is exactly one
place a generation's full detail renders.

Where this list renders now
----------------------------
Only inside ``gemini_studio`` (the centre panel) -- ``gemini_quick`` (the
left sidebar) no longer renders history at all, so it stays pure generation
controls, per the user's explicit request. Because of that, "Use as
reference" always targets ``__panel__gemini_quick`` EXPLICITLY: the
generation form that actually consumes the ``refs`` param lives only there,
regardless of which panel the card itself is drawn in.
"""
from __future__ import annotations

import logging

from imperal_sdk import ui

from gemini_config import GENERATION_LOG_COLLECTION, PANEL_HISTORY_LIMIT
from handlers.media import newest_first
from handlers.panel_viewer import CLOSED_SENTINEL, _find_generation

log = logging.getLogger("gemini.panel_history")

# Cap on references attached at once. Each one renders as a base64 thumbnail in
# the form, so this is a payload guard as much as a usability one; it also
# matches MAX_REFERENCE_IMAGES, the cap the generation call itself enforces.
MAX_SELECTED_REFERENCES = 6

# The panel's whole reply has a measured ~256 KB hard cap (confirmed live: a
# real account with 28 generations, each carrying a preview up to
# PREVIEW_BUDGET_CHARS=110,000 base64 chars, silently blew this and the panel
# never rendered -- no error, just a permanent spinner, since the client had
# no ui tree to mount). This budgets the CUMULATIVE preview payload across
# every card in one history render, leaving headroom for the surrounding
# JSON (buttons, prompts, the Grid/Stack wrapper itself). Deliberately well
# under 256_000: a handful of cards near the ceiling is safer than one that
# reproduces the exact failure this exists to prevent.
HISTORY_PAYLOAD_BUDGET_CHARS = 150_000


def _entry_card(doc) -> ui.UINode:
    """One compact history row: cached thumbnail + buttons, zero storage I/O.

    This card only ever renders inside ``gemini_studio`` now (the left
    ``gemini_quick`` panel holds no history). "Use as reference" always
    targets ``__panel__gemini_quick`` explicitly -- not a self-call -- because
    the generation form that consumes the ``refs`` param lives in THAT panel,
    not in whichever panel is currently showing this card. "Image info"/
    "Video info" call the centre panel (``gemini_studio``) directly: opening
    full detail is exactly the one thing this card intentionally does NOT do
    inline any more.
    """
    d = doc.data
    kind = d.get("kind", "")
    prompt = d.get("prompt", "")
    has_bytes = bool(d.get("storage_path"))

    children: list[ui.UINode] = []

    cached = d.get("preview_b64")
    cached_mime = d.get("preview_mime") or "image/png"
    if kind == "image" and cached:
        # Fixed height + object_fit="cover", not just width="100%": generations
        # come out at whatever aspect ratio the model/resolution picked
        # (square, portrait, widescreen...), so with only width pinned each
        # thumbnail's HEIGHT varied with its own image, and so did every card
        # around it in the 3-per-row Grid below -- a visibly uneven wall of
        # boxes instead of a tidy grid. Cropping to one fixed box (cover, like
        # any photo-grid UI) makes every card the same size regardless of what
        # resolution or aspect ratio that particular generation used.
        children.append(ui.Image(
            src=f"data:{cached_mime};base64,{cached}",
            alt=prompt[:120], width="100%", height="180px", object_fit="cover",
        ))

    if kind == "image" and has_bytes:
        children.append(ui.Button(
            label="Image info",
            variant="secondary",
            icon="Image",
            on_click=ui.Call("__panel__gemini_studio", generation_id=doc.id),
        ))
        # Choosing a reference HERE is the fix for the picker that listed prompt
        # text: this button sits next to the image itself, so the choice is made
        # by sight instead of by remembering which wall of text made which
        # picture. It re-renders THIS panel with the id carried in ``refs``.
        children.append(ui.Button(
            label="Use as reference",
            variant="ghost",
            icon="Link2",
            on_click=ui.Call("__panel__gemini_quick", refs=doc.id),
        ))
    elif kind == "video" and has_bytes:
        children.append(ui.Button(
            label="Video info",
            variant="secondary",
            icon="Video",
            on_click=ui.Call("__panel__gemini_studio", generation_id=doc.id),
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


async def _selected_references(ctx, refs_param: str) -> list[dict]:
    """Resolve the ``refs`` panel param into thumbnails for the form.

    ``refs_param`` is a comma-separated list of generation ids, set by the
    "Use as reference" button on a history card. It is read from the panel
    params rather than from a dropdown because the choice is made by LOOKING at
    an image, which is the entire point of the redesign.

    Only a CACHED preview is used as the thumbnail -- never a storage download.
    Rendering the form must stay free of media I/O; downloading here would put
    the original bytes of every attached reference into the response and
    reintroduce both the slow load and the payload problem the panel had.
    Records without a cached preview still appear, by label.
    """
    ids = [
        p.strip() for p in (refs_param or "").split(",")
        if p.strip() and p.strip() != CLOSED_SENTINEL
    ]
    if not ids:
        return []

    out: list[dict] = []
    for gen_id in ids[:MAX_SELECTED_REFERENCES]:
        # _find_generation, not ctx.store.get: get() does not send user_id to
        # the gateway, so it is both unscoped and the call that already failed
        # to resolve records elsewhere in this panel. This one is user-scoped.
        doc, _failed = await _find_generation(ctx, gen_id)
        if doc is None:
            continue
        d = doc.data
        cached = d.get("preview_b64")
        mime = d.get("preview_mime") or "image/png"
        label = (d.get("prompt") or "image").strip()
        out.append({
            "id": doc.id,
            "src": f"data:{mime};base64,{cached}" if cached else "",
            "label": (label[:48] + "…") if len(label) > 48 else label,
        })
    return out


async def _history_section(ctx) -> ui.UINode:
    """Render the history list -- ZERO storage I/O, cached-preview thumbnails only.

    Full detail (and the one storage read it needs) now happens exclusively
    in the centre panel when a card's info button is clicked, so this list
    can never be slowed down or hung by a single bad generation. Called only
    from ``gemini_studio`` -- ``gemini_quick`` renders no history at all.

    Measured, not guessed: a panel response has an ~256 KB hard cap on total
    reply size (server-enforced -- confirmed live: 28 generations, each
    carrying a cached preview up to ``PREVIEW_BUDGET_CHARS`` (110,000 base64
    chars, see core/preview.py) truncated the ENTIRE reply, so the panel
    never rendered at all -- no exception, no console error, just a
    perpetual spinner, because the client had no ``ui`` tree to mount.
    ``PANEL_HISTORY_LIMIT`` (60) bounds row COUNT but not cumulative payload
    size, which is what actually blew the cap. This stops adding cards once
    the running base64 total would risk the same overflow, and says so
    honestly instead of silently dropping the whole panel.
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

    cards: list[ui.UINode] = []
    shown = 0
    budget_chars = 0
    for d in docs:
        preview_chars = len(d.data.get("preview_b64") or "")
        if cards and (budget_chars + preview_chars) > HISTORY_PAYLOAD_BUDGET_CHARS:
            # Stop BEFORE adding this card, not after -- at least one card
            # already fit, so the list degrades gracefully instead of
            # blowing the whole reply's size cap the way an unbounded Grid
            # did in production with 28 real generations.
            break
        cards.append(_entry_card(d))
        budget_chars += preview_chars
        shown += 1

    # Three cards per row, per the user's explicit layout request -- a
    # vertical Stack (the old layout) put one generation per row, which made
    # a page of history mostly scrolling rather than browsing.
    grid = ui.Grid(children=cards, columns=3, gap=3)
    if shown >= len(docs):
        return grid

    return ui.Stack(
        direction="v",
        gap=3,
        children=[
            grid,
            ui.Text(
                f"Showing {shown} of {len(docs)} most recent generations — "
                "older ones are hidden to keep this panel loading reliably.",
                variant="caption",
            ),
        ],
    )
