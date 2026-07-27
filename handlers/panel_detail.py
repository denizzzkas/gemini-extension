"""The centre detail view: one generation, in full, with what to do next.

Layout the user asked for: the left column generates and lists, the CENTRE
opens a single chosen image together with the prompt that made it, the
reference image it was made from, and the actions that follow naturally --
download, view at full resolution, regenerate with the same inputs.

The honest constraint this view is built around
-----------------------------------------------
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
import html
import logging

from imperal_sdk import ui

from gemini_config import IMAGE_TOOL_FOR_MODEL, MODEL_IMAGE
from handlers.image_loader import _failure_message, _load_image

log = logging.getLogger("gemini.panel_detail")

# A data: URI in a download anchor is how the ORIGINAL leaves the panel. The
# extension's own storage has no publicly fetchable URL (verified: the
# gateway serves it only from an authenticated internal endpoint, and the
# webhook route wraps every response in JSON), so a link to a URL is not an
# option -- the bytes must travel in the page.
_DOWNLOAD_CEILING_CHARS = 6_000_000


def _download_block(raw: bytes, mime_type: str, filename: str) -> ui.UINode:
    """An anchor that saves the ORIGINAL bytes, not the preview.

    Rendered as raw HTML because a download needs the anchor's ``download``
    attribute, which no declarative component exposes: ``ui.Link`` emits a
    plain href (the browser would navigate to a data: URI instead of saving)
    and ``ui.Open`` only opens a tab.
    """
    encoded = base64.b64encode(raw).decode()
    if len(encoded) > _DOWNLOAD_CEILING_CHARS:
        return ui.Alert(
            title="Original too large to hand over here",
            message=(
                f"The file is {len(raw) // 1024} KB, which exceeds what a panel "
                "response can carry. Ask in chat for the image and it will be "
                "returned at full size there."
            ),
            type="warn",
        )

    safe_name = html.escape(filename, quote=True)
    href = f"data:{mime_type};base64,{encoded}"
    # The anchor is styled inline: the HTML block is sandboxed in an iframe,
    # so the panel's stylesheet does not reach it.
    return ui.Html(
        content=(
            f'<a href="{href}" download="{safe_name}" '
            'style="display:inline-block;padding:8px 14px;border-radius:8px;'
            'background:#5b8def;color:#fff;font:600 13px system-ui,sans-serif;'
            'text-decoration:none">Download original</a>'
        ),
        sandbox=True,
        max_height=60,
    )


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
    """Render the opened generation: image, prompt, reference, actions.

    ``download_armed`` is why the original is not embedded on every render.
    Measured in production, an original inlines to 571k-1005k base64 chars
    while ~954k was proven NOT to render -- so embedding it unconditionally
    would risk killing the whole panel just because someone opened an image.
    Merely opening a generation stays cheap; the heavy payload is attached only
    after an explicit click on "Prepare download".
    """
    d = doc.data
    prompt = d.get("prompt") or ""
    model = d.get("model") or ""
    kind = d.get("kind") or "image"
    mime_type = d.get("mime_type") or "image/jpeg"
    source = d.get("source") or "generated"

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

    children += _reference_block(references)

    children.append(ui.KeyValue(items=[
        {"key": "Model", "value": model or "—"},
        {"key": "Type", "value": kind},
        {"key": "Source", "value": "uploaded reference" if source == "upload" else "generated"},
        {"key": "Created", "value": d.get("created_at") or "—"},
    ], columns=2))

    if download_armed and raw_original:
        ext = "jpg" if "jpeg" in mime_type else ("png" if "png" in mime_type else "bin")
        children.append(_download_block(raw_original, mime_type, f"gemini-{doc.id}.{ext}"))
    elif d.get("storage_path"):
        size_note = (
            f" (~{len(raw_original) // 1024} KB)" if raw_original else ""
        )
        children.append(ui.Button(
            label=f"Prepare download{size_note}",
            variant="secondary",
            icon="Download",
            on_click=ui.Call(
                "__panel__gemini_studio",
                generation_id=doc.id,
                download="1",
            ),
        ))

    # Regenerate must hit the per-model tool, not the generic one: Imperal
    # prices a tool, and these models differ several-fold in cost, so calling
    # the wrong one bills the wrong amount.
    if kind == "image" and source != "upload" and prompt:
        tool = IMAGE_TOOL_FOR_MODEL.get(model) or IMAGE_TOOL_FOR_MODEL[MODEL_IMAGE]
        refs = [r["id"] for r in references if r.get("id")]
        children.append(ui.Button(
            label="Regenerate with the same prompt",
            on_click=ui.Call(
                tool,
                prompt=prompt,
                image_size=d.get("image_size") or "1K",
                reference_generation_ids=refs,
            ),
            variant="secondary",
        ))

    return ui.Card(
        title=(prompt[:70] + "…") if len(prompt) > 70 else (prompt or "Generation"),
        subtitle=f"{kind} · {model}",
        content=ui.Stack(children=children, direction="v", gap=3),
    )


async def load_detail(ctx, doc) -> dict:
    """Gather everything the detail view needs for one generation.

    Returns the preview src, whether it IS a preview (rather than the original
    inlined whole), the original bytes for the download anchor, and the
    resolved reference images.
    """
    image_src, fail_reason = await _load_image(ctx, doc.data)

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
        ref_src, _reason = await _load_image(ctx, ref_doc.data)
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
