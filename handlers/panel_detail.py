"""Generation detail: one image, in full, with what to do next.

Layout the user asked for: the left column generates and lists compactly;
opening one entry shows it in full -- prompt, reference, download,
regenerate -- in the restored centre panel (``gemini_studio``).

Why this is one shared builder, called from more than one panel
------------------------------------------------------------------
``gemini_studio`` (``slot="center"``, ``center_overlay=True`` -- see
handlers/panel.py) is the one place this now renders; the left panel
(``gemini_quick``) only shows compact history cards that hand off to it (see
handlers/panel_history.py). :func:`detail_content` builds the content as a
plain node LIST of prompt/reference/actions, independent of any one panel --
that is what lets it be reused, and unit-tested, from more than one caller
without hard-coding which panel embeds it.

The honest constraint this view is built around
-------------------------------------------------
A panel receives its image as base64 inside the response, and there is a
measured ceiling on that payload (see core/preview). So what is DISPLAYED is
a preview when the original is too big to inline. A real download IS offered
for the original when it was actually fetched (an ``<a download>`` anchor
under its own size ceiling -- see handlers/panel_html.py for why the earlier
"data: URIs are categorically blocked" conclusion here was an overreach: the
2017 Chrome change blocked page-initiated top-frame *navigation*, not an
anchor's forced-save `download` attribute). Above that ceiling the panel says
plainly that it cannot safely carry the original, rather than offering a chat
control that only confirms a fetch without delivering usable media.
"""
from __future__ import annotations

import base64
import logging

from imperal_sdk import ui

from gemini_config import IMAGE_TOOL_FOR_MODEL, MODEL_IMAGE
from handlers.image_loader import _failure_message, _load_image
from handlers.panel_html import copy_prompt_block, download_block

log = logging.getLogger("gemini.panel_detail")


def _ext_for(mime_type: str) -> str:
    """Best-effort file extension for a download filename."""
    if "png" in mime_type:
        return "png"
    if "jpeg" in mime_type or "jpg" in mime_type:
        return "jpg"
    if "mp4" in mime_type:
        return "mp4"
    return "bin"


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
) -> list[ui.UINode]:
    """Build the content nodes for one opened generation: image, prompt,
    reference, actions -- as a plain node LIST, not wrapped in its own Card,
    so the caller decides whether this sits inside a history card.

    Every action here is either a direct tool call ("Regenerate" invokes the
    per-model image tool itself, not a panel self-call) or a chat/native
    action (download, view-in-chat) -- none of it needs to know which panel
    is rendering it, so there is no panel_id to thread through any more.
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
                "Shown at preview size. Download the original below when it is available.",
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
        # Always offer a download attempt -- never gate this on raw_original
        # being present. download_block itself degrades in steps (original ->
        # cached preview -> honest "nothing available" alert), which is what
        # guarantees every generation has SOME way to download its result,
        # per the user's explicit ask.
        mime_type = d.get("mime_type") or ""
        filename = f"{doc.id}.{_ext_for(mime_type)}"
        children.append(download_block(
            raw_original, mime_type, filename,
            fallback_b64=d.get("preview_b64"),
            fallback_mime=d.get("preview_mime"),
        ))


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
