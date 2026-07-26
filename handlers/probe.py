"""Payload-ceiling probe — MEASURE the wall instead of guessing at it.

Why this exists
---------------
Every previous "images do not open" fix tuned a number on OUR side of an
undocumented limit. Production measurement (``diagnose_image_pipeline``)
proved the split empirically:

    ~90k  base64 chars  -> renders
    ~127k base64 chars  -> renders
    ??? .................. nobody has measured this range
    ~954k base64 chars  -> fails
    ~1.25M base64 chars -> fails

The SDK documents NO panel payload limit (searched: manifest_schema_ui,
ui/*, rpc/*, runtime/*), so the thresholds currently baked into
``panel_viewer`` are invented. This panel closes that blind spot: it
renders a synthetic image of a REQUESTED size inside an already-working
panel, so the operator can walk the size up until the image stops
appearing and read the ceiling off the last size that worked.

Design constraints that make the measurement HONEST
---------------------------------------------------
* **Incompressible payload.** The bytes are ``os.urandom``-derived, so any
  transport-level gzip cannot shrink them. A compressible payload (e.g. a
  flat colour) would sail through and report a falsely high ceiling —
  which is exactly how a 68KB white square "passed" while real renders
  failed.
* **A genuinely valid PNG.** If the probe emitted random bytes with an
  image mime type, a blank panel would be ambiguous: payload rejected, or
  payload delivered but undecodable? So the probe builds a real,
  spec-conformant PNG (stdlib ``zlib`` + manual chunks — Pillow is NOT
  available in production, verified: ``pillow_available: false``). If the
  image appears, the payload of that size crossed the wire intact.
* **Zero cost, zero network.** No Gemini call, no storage I/O, no billing.
  Sizes are reproducible, unlike real generations.

Read-only: touches no store collection and no user data.
"""
from __future__ import annotations

import base64
import logging
import os
import struct
import zlib

from imperal_sdk import ui

from handlers.panel_viewer import CLOSED_SENTINEL

log = logging.getLogger("gemini.probe")

# Ladder of base64 payload sizes to walk, in KB of base64 characters. The
# dense steps sit inside the measured blind zone (127k..954k) because that
# is the only region where the answer is genuinely unknown; the outer two
# reproduce a known-good and a known-bad result as controls.
PROBE_LADDER_KB = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

DEFAULT_PROBE_KB = 300


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    """One PNG chunk: length, type, payload, CRC32 over type+payload."""
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _noise_png(target_b64_chars: int) -> bytes:
    """Build a valid RGB PNG whose base64 form is ~``target_b64_chars`` long.

    Random pixel data makes the file incompressible, so the encoded size is
    driven by the pixel count rather than by the content — the property that
    makes this a trustworthy ruler. Base64 inflates by 4/3, and PNG stores
    raw-ish deflate output for noise, so the pixel budget is derived from
    the requested character count and then squared off into a bitmap.
    """
    target_bytes = max(1, int(target_b64_chars * 3 / 4))
    # 3 bytes/pixel + 1 filter byte per row; solve for a square side.
    side = max(1, int((target_bytes / 3) ** 0.5))

    raw = bytearray()
    for _ in range(side):
        raw.append(0)                    # filter type 0 (None) for this row
        raw.extend(os.urandom(side * 3))  # incompressible RGB pixels

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        # level=0 keeps noise from being re-expanded and keeps the size
        # predictable; the deflate stream stays spec-valid either way.
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 0))
        + _png_chunk(b"IEND", b"")
    )


def probe_toggle_button(panel_id: str, active: bool) -> ui.UINode:
    """Entry point rendered by the host panel: opens or closes the probe."""
    return ui.Button(
        label="Close payload probe" if active else "Measure payload ceiling",
        variant="ghost",
        icon="Ruler",
        on_click=ui.Call(
            f"__panel__{panel_id}",
            probe_kb=CLOSED_SENTINEL if active else str(DEFAULT_PROBE_KB),
        ),
    )


def _ladder_row(panel_id: str, current_kb: int) -> ui.UINode:
    """Clickable ladder so the wall can be walked without retyping params.

    Uses ``ui.Call("__panel__<id>", ...)`` — the same self-call pattern the
    working history buttons in ``handlers/panel.py`` already use, rather than
    a navigate-to-path action (which is what silently did nothing before).
    """
    return ui.Row(children=[
        ui.Button(
            label=("> " if kb == current_kb else "") + f"{kb}KB",
            size="sm",
            on_click=ui.Call(f"__panel__{panel_id}", probe_kb=str(kb)),
            variant="primary" if kb == current_kb else "secondary",
        )
        for kb in PROBE_LADDER_KB
    ])


def probe_section(panel_id: str, params: dict) -> ui.UINode | None:
    """Build the probe section, or ``None`` when the probe is not requested.

    NOT its own ``@ext.panel``: the left slot renders exactly ONE panel, so a
    second registration there would have silently hidden the working
    ``gemini_quick`` panel -- caught by
    ``tests/test_manifest.py::test_no_two_panels_claim_the_same_slot``, which
    is exactly the class of bug this whole investigation is about. So the
    probe renders INSIDE a panel already proven to render, driven by the
    ``probe_kb`` param.
    """
    requested = params.get("probe_kb")
    if requested in (None, "", CLOSED_SENTINEL):
        return None
    try:
        kb = int(str(requested).strip())
    except (TypeError, ValueError):
        kb = DEFAULT_PROBE_KB
    kb = max(1, min(kb, 4000))

    target_chars = kb * 1024
    png = _noise_png(target_chars)
    encoded = base64.b64encode(png).decode()

    log.info(
        "probe: requested %dKB -> png %d bytes -> %d base64 chars",
        kb, len(png), len(encoded),
    )

    return ui.Column(children=[
        ui.Text(content="Payload ceiling probe", variant="heading"),
        ui.Text(
            content=(
                "Walk the ladder until the image below stops appearing. The "
                "largest size that still renders IS the panel payload "
                "ceiling. Noise pixels keep the payload incompressible, so "
                "the number is honest."
            ),
            variant="caption",
        ),
        _ladder_row(panel_id, kb),
        ui.Divider(),
        ui.Text(content=f"requested: {kb}KB of base64", variant="body"),
        ui.Text(content=f"actual base64 chars: {len(encoded):,}", variant="body"),
        ui.Text(content=f"png bytes: {len(png):,}", variant="body"),
        ui.Divider(),
        ui.Text(
            content="Image renders below if this payload size survives:",
            variant="caption",
        ),
        ui.Image(
            src=f"data:image/png;base64,{encoded}",
            alt=f"{kb}KB noise probe",
            width="100%",
        ),
    ])
