"""Runtime diagnostics — report FACTS from the live environment.

Why this exists: every previous attempt at "images do not open" was diagnosed
on a laptop and guessed about production. The deploy validator proved the two
environments differ (Pillow is absent there), so a local measurement says
nothing about what the user actually hits.

This function measures the real pipeline in the real runtime: whether Pillow
is importable, how big each stored file really is, how long a download takes,
and what the base64 payload would be. Read-only; touches nothing.
"""
from __future__ import annotations

import base64
import io
import logging
import sys
import time

from pydantic import BaseModel, Field

from imperal_sdk import ActionResult

from app import chat
from core.preview import build_preview, sniff_format
from gemini_config import GENERATION_LOG_COLLECTION
from handlers.image_loader import PREVIEW_FIELD
from handlers.media import newest_first

log = logging.getLogger("gemini.diag")


class DiagnoseParams(BaseModel):
    limit: int = Field(5, ge=1, le=20, description="How many recent generations to probe")


class ProbeRow(BaseModel):
    generation_id: str = ""
    prompt_head: str = ""
    model: str = ""
    has_storage_path: bool = False
    downloaded_ok: bool = False
    download_ms: int = 0
    raw_bytes: int = 0
    base64_chars: int = 0
    shrunk_bytes: int = 0
    shrunk_base64_chars: int = 0
    detected_format: str = ""
    preview_cached_chars: int = 0
    preview_built_chars: int = 0
    preview_build_ms: int = 0
    note: str = ""


class DiagnoseRecord(BaseModel):
    python_version: str = ""
    pillow_available: bool = False
    pillow_version: str = ""
    rows: list[ProbeRow] = Field(default_factory=list)


def _pillow():
    try:
        from PIL import Image  # noqa: PLC0415
        import PIL  # noqa: PLC0415
        return Image, getattr(PIL, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        log.warning("diag: Pillow unavailable: %s", e)
        return None, ""


@chat.function(
    "diagnose_image_pipeline",
    action_type="read",
    chain_callable=True,
    data_model=DiagnoseRecord,
    description=(
        "Diagnostic: report why generated images may fail to display. Measures "
        "the real runtime — Pillow availability, stored file sizes, download "
        "timings and base64 payload sizes for recent generations."
    ),
)
async def fn_diagnose_image_pipeline(ctx, params: DiagnoseParams) -> ActionResult:
    """Probe the live image pipeline and report hard numbers."""
    Image, pil_version = _pillow()
    record = DiagnoseRecord(
        python_version=sys.version.split()[0],
        pillow_available=Image is not None,
        pillow_version=pil_version,
    )

    try:
        page = await ctx.store.query(
            GENERATION_LOG_COLLECTION,
            where={"user_id": ctx.user.imperal_id, "kind": "image"},
            limit=50,
        )
        docs = newest_first(page.data)[: params.limit]
    except Exception as e:  # noqa: BLE001
        return ActionResult.error(error=f"history query failed: {e}", retryable=True)

    for doc in docs:
        d = doc.data
        row = ProbeRow(
            generation_id=doc.id,
            prompt_head=(d.get("prompt") or "")[:60],
            model=d.get("model", ""),
            has_storage_path=bool(d.get("storage_path")),
        )
        path = d.get("storage_path")
        if not path:
            row.note = "no storage_path on the record"
            record.rows.append(row)
            continue

        started = time.monotonic()
        try:
            raw = await ctx.storage.download(path)
            row.download_ms = int((time.monotonic() - started) * 1000)
            row.downloaded_ok = True
            row.raw_bytes = len(raw)
            row.base64_chars = len(base64.b64encode(raw))
        except Exception as e:  # noqa: BLE001
            row.download_ms = int((time.monotonic() - started) * 1000)
            row.note = f"download failed: {type(e).__name__}: {e}"[:200]
            record.rows.append(row)
            continue

        # What the bytes ACTUALLY are, from the magic number rather than the
        # stored mime_type: the models return JPEG, and records predating the
        # mime_type field claim PNG by default. Falls back to the leading
        # bytes in hex so an unrecognised format is still diagnosable.
        row.detected_format = sniff_format(raw) or raw[:4].hex()

        # The preview already stored on the record -- what the panel serves.
        row.preview_cached_chars = len(doc.data.get(PREVIEW_FIELD) or "")

        # Measure the REAL shrink path in the real runtime. The previous
        # version of this probe measured a Pillow path that cannot execute
        # here, so it reported 0 and said nothing about the live pipeline.
        t0 = time.monotonic()
        try:
            built = build_preview(raw, doc.data.get("mime_type") or "")
            row.preview_build_ms = int((time.monotonic() - t0) * 1000)
            if built is None:
                row.note = (
                    f"no preview for {row.detected_format!r} -- panel must "
                    "inline the full payload"
                )
            else:
                row.preview_built_chars = len(built[0])
                row.shrunk_base64_chars = len(built[0])
        except Exception as e:  # noqa: BLE001
            row.preview_build_ms = int((time.monotonic() - t0) * 1000)
            row.note = f"preview failed: {type(e).__name__}: {e}"[:200]

        if Image is not None:
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    im.thumbnail((512, 512))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=70, optimize=True)
                row.shrunk_bytes = len(buf.getvalue())
            except Exception:  # noqa: BLE001, S110
                pass  # Pillow is absent in production; not worth reporting.

        record.rows.append(row)

    ok = sum(1 for r in record.rows if r.downloaded_ok)
    summary = (
        f"Probed {len(record.rows)} image generations: {ok} downloaded. "
        f"Pillow available: {record.pillow_available} ({record.pillow_version or 'n/a'})."
    )
    return ActionResult.success(data=record, summary=summary)
