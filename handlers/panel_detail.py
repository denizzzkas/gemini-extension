"""Generation detail: one image, in full, with what to do next.

Layout the user asked for: the left column generates, lists, AND (once an
entry is opened) shows it in full -- prompt, reference, download, regenerate.

Why this no longer targets the center slot
-------------------------------------------
This used to be a separate "Gemini Studio" panel on ``slot="center"``, which
I-PANEL-RENDERING-CONTRACT documents as rendered ON DEMAND and only if the
host honours ``center_overlay`` -- unreliable by design, not by bug. The user
confirmed in real testing it never opened at all, so it was removed outright
(see handlers/panel.py). :func:`detail_content` builds the same blocks as a
plain node list, parameterised by ``panel_id``, and renders reliably inline
inside the permanent left panel (``gemini_quick``) -- the only panel this
extension declares any more.

The honest constraint this view is built around
-------------------------------------------------
A panel receives its image as base64 inside the response, and there is a
measured ceiling on that payload (see core/preview). So what is DISPLAYED is
a preview when the original is too big to inline. The button offered for the
original is NOT an in-panel download any more either: ``ui.Open`` on a
``data:`` URI -- the previous attempt -- cannot ever work, because Chrome has
outright blocked top-frame navigation to ``data:`` URIs since 2017 regardless
of size (see handlers/panel_html.py for the full account). Instead the button
hands off to chat, the one channel already proven to deliver a real,
full-resolution image (every generation reply already renders ``image_base64``
inline). That distinction is stated in the UI rather than hidden, because the
previous version of this extension quietly showed a shrunk image and called
it the result.
"""
from __future__ import annotations

import base64
import logging

from imperal_sdk import ui

from gemini_config import IMAGE_TOOL_FOR_MODEL, MODEL_IMAGE
from handlers.image_loader import _failure_message, _load_image
from handlers.panel_html import copy_prompt_block, view_full_resolution_block

log = logging.getLogger("gemini.panel_detail")


def _reference_block(refs: list[dict]) -> list[ui.UINode]:
    """Show the reference image(s) this generation was made from."""
    if not refs:
        return []
    children: list[ui.UINode] = [ui.Text("Made from reference", variant="caption")]
    for ref in refs:
        src = ref.get("src") or ""
        if src:
            children.append(ui.Image(src=src, alt=ref.get("label", "reference"), width="140px"))
        children.append(ui.Text(ref.get("label") or "(reference image)", variant="caption"))
    return children


def detail_content(
    doc,
    *,
    image_src: str,
    fail_reason: str,
    raw_original: bytes | None,
    references: list[dict],
    is_preview: bool,
    panel_id: str = "gemini_quick",
) -> list[ui.UINode]:
    """Build the content nodes for one opened generation: image, prompt,
    reference, actions -- as a plain node LIST, not wrapped in its own Card,
    so the caller decides whether this sits inside a history card.

    ``panel_id`` is which panel's self-call "Regenerate" targets. It must be
    the panel the caller is ACTUALLY rendering inside -- ``gemini_quick``
    (left, permanent), the only panel this extension declares.
    """
    d = doc.data
    prompt = d.get("prompt") or ""
    model = d.get("model") or ""
    kind = d.get("kind") or "image"
    source = d.get("source") or "generated"

    children: list[ui.UINode] = []

    if image_src:
        children.append(ui.Image(src=image_src, alt=prompt[:120], width="100%"))
        if is_preview:
            # Say plainly that this is not the original. Showing a shrunk image
            # as though it were the result is the exact thing that made the
            # earlier panel misleading.
            children.append(ui.Text(
                "Shown at preview size — use \"View full resolution in chat\" "
                "for the untouched original.",
                variant="caption",
            ))
    else:
        children.append(ui.Alert(
            title="Image unavailable",
            message=_failure_message(fail_reason),
            type="warn",
        ))

    children.append(ui.Divider())
    children.append(ui.Text("Prompt", variant="caption"))
    children.append(ui.Text(prompt or "(no prompt recorded)"))
    # copy_prompt_block returns None when there is nothing worth copying (an
    # empty or whitespace-only prompt), so the result is checked rather than
    # appended blindly -- a None child would break the render.
    copy_button = copy_prompt_block(prompt)
    if copy_button is not None:
        children.append(copy_button)

    children += _reference_block(references)

    children.append(ui.KeyValue(items=[
        {"key": "Model", "value": model or "—"},
        {"key": "Type", "value": kind},
        {"key": "Source", "value": "uploaded reference" if source == "upload" else "generated"},
        {"key": "Created", "value": d.get("created_at") or "—"},
    ], columns=2))

    if d.get("storage_path"):
        children.append(view_full_resolution_block(doc.id, kind))

    # Regenerate must hit the per-model tool, not the generic one: Imperal
    # prices a tool, and these models differ several-fold in cost, so calling
    # the wrong one bills the wrong amount.
    if kind == "image" and source != "upload" and prompt:
        tool = IMAGE_TOOL_FOR_MODEL.get(model) or IMAGE_TOOL_FOR_MODEL[MODEL_IMAGE]
        ref_ids = [r["id"] for r in references if r.get("id")]
        children.append(ui.Button(
            label="Regenerate with the same prompt",
            on_click=ui.Call(
                tool,
                prompt=prompt,
                image_size=d.get("image_size") or "1K",
                reference_generation_ids=ref_ids,
            ),
            variant="secondary",
        ))

    return children


async def load_detail(ctx, doc) -> dict:
    """Gather everything the detail view needs for one generation.

    Returns the preview src, whether it IS a preview (rather than the original
    inlined whole), the original bytes (used only to tell if it's a preview
    and to show its size -- never embedded raw into a panel response any
    more), and the resolved reference images.
    """
    image_src, fail_reason = await _load_image(ctx, doc.data, doc.id)

    raw_original: bytes | None = None
    storage_path = doc.data.get("storage_path")
    if storage_path:
        try:
            raw_original = await ctx.storage.download(storage_path)
        except Exception as e:  # noqa: BLE001 — the view still works without it
            log.info("detail: original download failed for %r: %s", storage_path, e)

    # "Is this a preview?" is answered by comparing what is displayed against
    # the original, instead of assuming: when the original was small enough to
    # inline, _load_image serves it verbatim and no warning should appear.
    is_preview = False
    if image_src and raw_original is not None:
        shown_chars = len(image_src.split(",", 1)[-1])
        is_preview = shown_chars < len(base64.b64encode(raw_original).decode())

    references: list[dict] = []
    for ref_id in doc.data.get("reference_ids") or []:
        ref_doc, _failed = await _find_ref(ctx, str(ref_id))
        if ref_doc is None:
            continue
        ref_src, _reason = await _load_image(ctx, ref_doc.data, ref_doc.id)
        references.append({
            "id": ref_doc.id,
            "src": ref_src,
            "label": ref_doc.data.get("prompt") or "reference",
        })

    return {
        "image_src": image_src,
        "fail_reason": fail_reason,
        "raw_original": raw_original,
        "references": references,
        "is_preview": is_preview,
    }


async def _find_ref(ctx, generation_id: str):
    """Local import wrapper -- avoids a circular import at module load."""
    from handlers.panel_viewer import _find_generation

    return await _find_generation(ctx, generation_id)
