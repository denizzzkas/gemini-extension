"""Generation detail: one image, in full, with what to do next.

Layout the user asked for: the left column generates, lists, AND (once an
entry is opened) shows it in full -- prompt, reference, download, regenerate.

Why this no longer targets the center slot
-------------------------------------------
This used to be a separate "Gemini Studio" panel on ``slot="center"``, which
I-PANEL-RENDERING-CONTRACT documents as rendered ON DEMAND and only if the
host honours ``center_overlay`` -- unreliable by design, not by bug. Every
button built on it (download, regenerate, even opening it) inherited that
unreliability, which is exactly the "Image info falls over sometimes" and
"download spins forever" reports. :func:`detail_content` now builds the same
blocks as a plain node list, parameterised by ``panel_id``, so the SAME code
renders reliably inline inside the permanent left panel (``gemini_quick``).
The old center panel is kept only as a secondary, best-effort surface.

The honest constraint this view is built around
-------------------------------------------------
A panel receives its image as base64 inside the response, and there is a
measured ceiling on that payload (see core/preview). So what is DISPLAYED is
a preview, while the ORIGINAL bytes are handed over through a download anchor
instead of being rendered -- a browser saves a data: URI to disk perfectly
well at sizes it would refuse to lay out inline. That distinction is stated in
the UI rather than hidden, because the previous version of this extension
quietly showed a shrunk image and called it the result.
"""
from __future__ import annotations

import base64
import logging

from imperal_sdk import ui

from gemini_config import IMAGE_TOOL_FOR_MODEL, MODEL_IMAGE
from handlers.image_loader import _failure_message, _load_image
from handlers.panel_html import copy_prompt_block, download_block

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
    download_armed: bool = False,
    panel_id: str = "gemini_quick",
) -> list[ui.UINode]:
    """Build the content nodes for one opened generation: image, prompt,
    reference, actions -- as a plain node LIST, not wrapped in its own Card,
    so the caller decides whether this sits inside a history card (left
    panel) or a standalone page (the legacy Studio panel).

    ``panel_id`` is which panel's self-call the actions here target. It must
    be the panel the caller is ACTUALLY rendering inside: hardcoding
    "gemini_studio" previously meant every "Download original"/"Back to the
    image" button fired against the center-overlay slot, which only renders
    if the host grants it a render path. Passing the real panel_id is what
    lets these same actions work reliably inside ``gemini_quick`` (left,
    permanent) instead.

    ``download_armed`` is why the original is not embedded on every render.
    Measured in production, an original inlines to 571k-1005k base64 chars
    while ~954k was proven NOT to render -- so embedding it unconditionally
    would risk killing the whole panel just because someone opened an image.
    Merely opening a generation stays cheap; the heavy payload is attached only
    after an explicit click on "Download original".
    """
    d = doc.data
    prompt = d.get("prompt") or ""
    model = d.get("model") or ""
    kind = d.get("kind") or "image"
    mime_type = d.get("mime_type") or "image/jpeg"
    source = d.get("source") or "generated"

    # ARMED = a deliberately MINIMAL response carrying one thing: the original.
    #
    # This is the second half of the "download spins forever" fix. The original
    # measures 571k-1005k base64 chars on real generations, and a panel response
    # has an undocumented size ceiling (measured: ~127k renders, ~954k does not).
    # The old armed view stacked the preview image, the prompt, the metadata AND
    # every reference preview into the SAME response as those bytes, so the
    # response could land well past anything proven to render -- and a response
    # that never renders is exactly a spinner that never stops.
    #
    # So while armed, everything optional is dropped. The full view is one click
    # away via "Back to the image".
    if download_armed and raw_original:
        ext = "jpg" if "jpeg" in mime_type else ("png" if "png" in mime_type else "bin")
        kb = len(raw_original) // 1024
        return [
            ui.Text(f"Download original -- {kb} KB · {mime_type}", variant="caption"),
            ui.Text(
                "This is the untouched file from the model — full "
                "resolution, not the preview.",
                variant="caption",
            ),
            download_block(raw_original, mime_type, f"gemini-{doc.id}.{ext}"),
            ui.Button(
                label="Back to the image",
                variant="ghost",
                icon="ArrowLeft",
                on_click=ui.Call(f"__panel__{panel_id}", generation_id=doc.id),
            ),
        ]

    children: list[ui.UINode] = []

    if image_src:
        children.append(ui.Image(src=image_src, alt=prompt[:120], width="100%"))
        if is_preview:
            # Say plainly that this is not the original. Showing a shrunk image
            # as though it were the result is the exact thing that made the
            # earlier panel misleading.
            children.append(ui.Text(
                "Shown at preview size — use Download original for full quality.",
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

    # Only the ARMING button here -- the armed state returned early above with a
    # minimal response, so this branch is the unarmed one by construction.
    if d.get("storage_path"):
        size_note = (
            f" (~{len(raw_original) // 1024} KB)" if raw_original else ""
        )
        children.append(ui.Button(
            label=f"Download original{size_note}",
            variant="secondary",
            icon="Download",
            on_click=ui.Call(
                f"__panel__{panel_id}",
                generation_id=doc.id,
                download="1",
            ),
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


def detail_view(
    doc,
    *,
    image_src: str,
    fail_reason: str,
    raw_original: bytes | None,
    references: list[dict],
    is_preview: bool,
    download_armed: bool = False,
) -> ui.UINode:
    """Legacy standalone-page wrapper, kept for the ``gemini_studio`` panel.

    Best-effort only: this panel only renders if the host grants the center
    slot a render path. The primary, always-working surface is the inline
    detail block in ``gemini_quick`` built directly from ``detail_content``.
    """
    d = doc.data
    prompt = d.get("prompt") or ""
    kind = d.get("kind") or "image"
    model = d.get("model") or ""
    content = detail_content(
        doc,
        image_src=image_src,
        fail_reason=fail_reason,
        raw_original=raw_original,
        references=references,
        is_preview=is_preview,
        download_armed=download_armed,
        panel_id="gemini_studio",
    )
    return ui.Card(
        title=(prompt[:70] + "…") if len(prompt) > 70 else (prompt or "Generation"),
        subtitle=f"{kind} · {model}",
        content=ui.Stack(children=content, direction="v", gap=3),
    )


async def load_detail(ctx, doc) -> dict:
    """Gather everything the detail view needs for one generation.

    Returns the preview src, whether it IS a preview (rather than the original
    inlined whole), the original bytes for the download anchor, and the
    resolved reference images.
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
