"""Accept a user-supplied image as a reference for generation.

Why this exists
---------------
Reference images could previously come from ONE place: a past generation of
your own (``reference_generation_ids``). So there was no way to say "use THIS
photo" -- not from the panel, not from chat. That is the gap this closes.

Design choice that keeps it small: an upload is stored as an ordinary
generation record with ``kind="image"`` and ``source="upload"``. It therefore
flows through the EXISTING reference machinery (``_resolve_reference_images``
looks up a doc, checks ownership, downloads ``storage_path``) with no changes
to the generation path at all. An upload is simply an image you own that
happens not to have been generated.

The wire shape of an uploaded file is not documented in the SDK, and guessing
undocumented contracts is exactly what broke earlier work here, so
:func:`_extract_upload` accepts every plausible shape rather than betting on
one, and says so plainly when it recognises none.
"""
from __future__ import annotations

import base64
import binascii
import logging

from pydantic import BaseModel, Field, field_validator

from imperal_sdk import ActionResult

from app import chat
from core.preview import build_preview, sniff_format
from handlers.image_loader import _cache_preview
from handlers.media import _log_generation, _save_media
from return_models import (
    UploadedReferenceRecord, UploadReferenceResult,
)

log = logging.getLogger("gemini.uploads")

# Generous but bounded: a reference image only needs to be recognisable, and
# an unbounded upload would be stored and re-encoded in memory.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# Keys the frontend might plausibly use for the payload and the filename. The
# SDK documents only "base64 file data", not the field names.
_DATA_KEYS = ("data", "data_base64", "data_b64", "base64", "content", "bytes", "b64")
_NAME_KEYS = ("name", "filename", "file_name", "title")
_MIME_KEYS = ("mime_type", "mime", "content_type", "type")


class UploadReferenceParams(BaseModel):
    files: list = Field(
        default_factory=list,
        description=(
            "Image file(s) from the panel dropzone. Sent automatically by the "
            "upload widget -- not something to construct by hand. A single "
            "image may also be passed on its own, without a list."
        ),
    )

    @field_validator("files", mode="before")
    @classmethod
    def _accept_a_single_file(cls, v):
        """Wrap one item into a list instead of rejecting it.

        Verified against production: calling this tool with one image as a
        plain string failed outright with a pydantic list_type error. The body
        below is deliberately tolerant about the SHAPE of each item, so being
        rigid about the container was an inconsistency that turned a perfectly
        clear request into an error the caller could not act on.
        """
        if v is None:
            return []
        if isinstance(v, (str, bytes, dict)):
            return [v]
        return v
    label: str = Field(
        "",
        max_length=200,
        description="Optional note describing the image, shown in history.",
    )


def _strip_data_uri(text: str) -> str:
    """Return the base64 payload of a ``data:`` URI, or the text unchanged."""
    if text.startswith("data:") and "," in text:
        return text.split(",", 1)[1]
    return text


def _extract_upload(item) -> tuple[bytes, str, str]:
    """Pull ``(raw_bytes, filename, declared_mime)`` out of one uploaded item.

    Tolerant by necessity: the upload payload shape is undocumented, so a
    plain base64 string, a ``data:`` URI and several dict spellings are all
    accepted. Raises ``ValueError`` with a readable reason when the item
    carries no recognisable image data -- an unreadable upload must say so,
    not fail silently the way a missing preview once did.
    """
    name = ""
    mime = ""
    payload = None

    # ``arg_kind="bytes_ref"`` in the chat file-sink contract resolves to
    # upload bytes for the duration of this tool call. The parameter validator
    # already wraps a bare bytes value into ``files=[...]``; accepting it here
    # is therefore the normal chat-upload path, not an unusual fallback.
    if isinstance(item, (bytes, bytearray)):
        return bytes(item), name, mime
    if isinstance(item, str):
        payload = item
    elif isinstance(item, dict):
        for k in _NAME_KEYS:
            if isinstance(item.get(k), str) and item[k]:
                name = item[k]
                break
        for k in _MIME_KEYS:
            v = item.get(k)
            if isinstance(v, str) and "/" in v:
                mime = v
                break
        for k in _DATA_KEYS:
            v = item.get(k)
            if isinstance(v, str) and v:
                payload = v
                break
            if isinstance(v, (bytes, bytearray)):
                return bytes(v), name, mime
    else:
        raise ValueError(f"unsupported upload item of type {type(item).__name__}")

    if not isinstance(payload, str) or not payload:
        raise ValueError("upload carried no file data")

    try:
        raw = base64.b64decode(_strip_data_uri(payload), validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"file data was not valid base64 ({e})") from e

    if not raw:
        raise ValueError("decoded file was empty")
    return raw, name, mime


@chat.function(
    "upload_reference_image",
    action_type="write",
    chain_callable=True,
    effects=["create:media"],
    event="gemini.reference_image_uploaded",
    data_model=UploadReferenceResult,
    description=(
        "Store an image the user supplies (photo, screenshot, artwork) so it "
        "can be used as a reference for image generation. Returns a "
        "generation_id to pass in reference_generation_ids. Use when the user "
        "attaches or uploads a picture and wants it used as the basis, style "
        "or character reference for a new image."
    ),
)
async def fn_upload_reference_image(ctx, params: UploadReferenceParams) -> ActionResult:
    """Persist uploaded image(s) as reference-usable records."""
    if not params.files:
        return ActionResult.error(
            "No file was received. Attach an image and try again.",
            retryable=False,
        )

    stored: list[dict] = []
    problems: list[str] = []

    for item in params.files:
        try:
            raw, name, declared_mime = _extract_upload(item)
        except ValueError as e:
            problems.append(str(e))
            continue

        if len(raw) > MAX_UPLOAD_BYTES:
            problems.append(
                f"{name or 'file'} is {len(raw) // (1024 * 1024)}MB -- the limit "
                f"is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB"
            )
            continue

        # Trust the BYTES, not the declared type: the mime a client sends is
        # frequently wrong or absent, and a mislabelled image is what made the
        # PNG-only preview path a silent no-op on every real render.
        fmt = sniff_format(raw)
        if not fmt:
            problems.append(
                f"{name or 'file'} is not a PNG or JPEG image (Gemini "
                "references must be one of those)"
            )
            continue
        mime = f"image/{'jpeg' if fmt == 'jpeg' else 'png'}"

        data_b64 = base64.b64encode(raw).decode()
        storage_path, url = await _save_media(ctx, "image", mime, data_b64)
        if not storage_path:
            problems.append(f"could not store {name or 'file'}")
            continue

        label = params.label or name or "Uploaded reference image"
        gen_id = await _log_generation(
            ctx, "image", label, "upload", source="upload",
            url=url, storage_path=storage_path, mime_type=mime,
        )
        if not gen_id:
            problems.append(f"could not record {name or 'file'}")
            continue

        # Same treatment as a generated image: build the panel-sized preview
        # NOW, while the bytes are already in memory. It must be the SHRUNK
        # preview -- caching the full-size base64 here would store a payload
        # far above the display ceiling and put the panel back in the exact
        # "image never appears" state this extension already paid for once.
        preview = build_preview(raw, mime)
        if preview:
            encoded, preview_mime = preview
            await _cache_preview(ctx, {"id": gen_id}, encoded, preview_mime)
        else:
            log.info("upload %s: no preview built; panel will build on open", gen_id)
        stored.append({
            "generation_id": gen_id,
            "filename": name,
            "mime_type": mime,
            "bytes": len(raw),
        })

    if not stored:
        return ActionResult.error(
            "Could not use that upload: " + "; ".join(problems or ["unknown reason"]),
            retryable=False,
        )

    ids = [s["generation_id"] for s in stored]
    summary = (
        f"Stored {len(stored)} reference image(s). Pass "
        f"reference_generation_ids={ids} to generate from them."
    )
    if problems:
        summary += " Skipped: " + "; ".join(problems)

    return ActionResult.success(
        data=UploadReferenceResult(
            stored=[UploadedReferenceRecord(**item) for item in stored],
            generation_ids=ids,
            skipped=problems,
        ),
        summary=summary,
    )


# Let Webbee route a file the user drops into CHAT here, not just the panel
# dropzone. Without this the only way in is the panel, which is precisely the
# half of the gap the user reported.
chat.ext.file_sink(
    "upload_reference_image",
    accepts=["image/png", "image/jpeg"],
    arg="files",
    arg_kind="bytes_ref",
    description="Use an attached image as a reference for Gemini generation",
)
