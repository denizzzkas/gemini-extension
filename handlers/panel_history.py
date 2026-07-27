"""The generation history list: cards, and the reference choices they feed.

Split out of ``handlers/panel.py`` so that file declares panels while this one
renders the list of what you have made.

The rule the card layout is built around: the list does ZERO media I/O. Only
the single entry the user explicitly opened costs a storage read, so a slow or
failing read degrades that one card instead of hanging the whole panel -- the
behaviour that once made the panel look dead on load.
"""
from __future__ import annotations

import logging

from imperal_sdk import ui

from gemini_config import GENERATION_LOG_COLLECTION, PANEL_HISTORY_LIMIT
from handlers.media import newest_first
from handlers.panel_forms import REFERENCE_CHOICE_LIMIT
from handlers.panel_viewer import (
    CLOSED_SENTINEL, FAIL_NONE, _failure_message, _find_generation, _load_image,
)

log = logging.getLogger("gemini.panel_history")

# Cap on references attached at once. Each one renders as a base64 thumbnail in
# the form, so this is a payload guard as much as a usability one; it also
# matches MAX_REFERENCE_IMAGES, the cap the generation call itself enforces.
MAX_SELECTED_REFERENCES = 6


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
        # Opening in the centre gives the full detail view (prompt, reference,
        # download, regenerate). It stays SECONDARY to the inline view above:
        # the centre slot renders only if the host honours center_overlay, and
        # betting a button on that is what left dead buttons here before.
        # "Image info" rather than "Open in Studio": the old label named the
        # PLACE it opens, which tells the user nothing about what they get.
        # This button leads to the prompt, the reference image, the file size,
        # the download and regenerate -- i.e. information about this image.
        children.append(ui.Button(
            label="Image info",
            variant="ghost",
            icon="Info",
            on_click=ui.Call("__panel__gemini_studio", generation_id=doc.id),
        ))
        # Choosing a reference HERE is the fix for the picker that listed prompt
        # text: this button sits next to the image itself, so the choice is made
        # by sight instead of by remembering which wall of text made which
        # picture. It re-renders this panel with the id carried in ``refs``.
        children.append(ui.Button(
            label="Use as reference",
            variant="ghost",
            icon="Link2",
            on_click=ui.Call(f"__panel__{panel_id}", refs=doc.id),
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


async def _reference_choices(ctx) -> list[dict]:
    """Recent images of this user, shaped for the form's reference picker.

    Images only: a video cannot be a reference, and offering one would produce
    a choice that silently does nothing. Uploaded references are included --
    they are images the user owns, which is the entire point of uploading one.
    """
    try:
        page = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id, "kind": "image"},
            limit=REFERENCE_CHOICE_LIMIT,
        )
    except Exception as e:  # noqa: BLE001
        log.error("panel: reference choices query failed: %s", e)
        return []

    choices: list[dict] = []
    for doc in newest_first(page.data):
        if not doc.data.get("storage_path"):
            continue  # no bytes -> cannot be sent as a reference
        label = (doc.data.get("prompt") or "untitled").strip()
        if doc.data.get("source") == "upload":
            label = f"⬆ {label}"
        choices.append({
            "value": doc.id,
            "label": (label[:60] + "…") if len(label) > 60 else label,
        })
    return choices


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


