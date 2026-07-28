"""Download and copy-prompt affordances, built from NATIVE components only.

Why this file no longer uses ``ui.Html``
-----------------------------------------
The first version of this file used raw ``<a download>``/``<button onclick=...>``
markup via ``ui.Html``, reasoning that ``sandbox=True`` (the default, which
wraps the markup in a sandboxed iframe) was what blocked both the download and
the clipboard write. That reasoning was tested for real and DISPROVEN: even
with ``sandbox=False``, the download button appeared only for small images and
still could not be clicked, and the copy button was not clickable either. So
the unreliable part was never the sandbox flag -- it is raw custom HTML/JS
itself, most likely because the panel host's Content-Security-Policy blocks
inline ``onclick=`` handlers regardless of iframe sandboxing. Guessing at a
second sandbox-shaped fix would repeat the same mistake, so both features are
rebuilt from components already PROVEN reliable elsewhere in this same panel
(``ui.Button`` + a native ``UIAction``), instead of custom markup.

Download: native content-type sniffing, not a ``download`` attribute
----------------------------------------------------------------------
``ui.Open`` fires a real, native "open URL" action -- the same action class
every other working button in this panel already uses (view image, refresh,
regenerate). Pointed at a ``data:`` URI whose MIME type is forced to
``application/octet-stream``, the browser cannot display it inline and saves
it instead -- a long-standing, JS-free technique. No custom HTML, no
``onclick``, nothing that depends on this frontend's CSP.

Copy: a selectable text block, not a clipboard button
--------------------------------------------------------
There is no native Copy/Clipboard component in the SDK (checked: the only
clipboard mention anywhere is a secrets panel that deliberately forbids it),
so a genuine one-click copy is not buildable from components alone. Rather
than ship a second custom-JS button after the first one failed in the same
way, the prompt is shown as ``ui.Code`` -- a plain, syntax-block text area the
user selects and copies with the browser's own Ctrl+C, which cannot silently
fail the way a blocked inline handler can.
"""
from __future__ import annotations

import base64
import html

from imperal_sdk import ui

__all__ = ["DOWNLOAD_CEILING_CHARS", "download_block", "copy_prompt_block"]

# A SAFETY VALVE, not a measured limit.
#
# What IS measured, in production: an image rendered through ``ui.Image`` stops
# appearing somewhere between ~127k base64 chars (renders) and ~954k (does not).
# A UIAction param is a different case -- the browser never lays those bytes
# out as a document, it only navigates to them -- so that number must not
# simply be assumed to apply here. This is set high enough to hand over a
# normal render (real originals measure 571k-1005k chars) and low enough to
# refuse something absurd, with an honest message instead of a silent failure.
DOWNLOAD_CEILING_CHARS = 2_000_000


def download_block(raw: bytes, mime_type: str, filename: str) -> ui.UINode:
    """A button that triggers a native browser download of the ORIGINAL bytes.

    ``raw`` must be the untouched bytes as stored, so what lands on disk is
    byte-for-byte what the model returned. Nothing here re-encodes, resizes or
    recompresses; that is the entire contract of this function.

    The MIME is deliberately overridden to ``application/octet-stream``
    regardless of the real ``mime_type``: navigating to a viewable type (e.g.
    ``image/jpeg``) just opens the image in the tab, it does not download it.
    An opaque type is what makes the browser save the file instead -- the
    filename is passed only as a caption since a data: URI has no filename of
    its own; most browsers name the saved file generically (e.g. "download").
    """
    encoded = base64.b64encode(raw).decode()
    if len(encoded) > DOWNLOAD_CEILING_CHARS:
        return ui.Alert(
            title="Original too large to hand over here",
            message=(
                f"The file is {len(raw) // 1024} KB, which exceeds what a panel "
                "response can carry. Ask for this image in chat and you get it "
                "at full size there."
            ),
            type="warn",
        )

    href = f"data:application/octet-stream;base64,{encoded}"
    return ui.Stack(direction="v", gap=1, children=[
        ui.Button(
            label="Download original",
            variant="primary",
            icon="Download",
            on_click=ui.Open(href),
        ),
        ui.Text(
            f"Saves as \"{html.escape(filename)}\" — some browsers name it "
            "\"download\" instead; rename after saving if so.",
            variant="caption",
        ),
    ])


def copy_prompt_block(prompt: str) -> ui.UINode | None:
    """The full prompt as a selectable block, or None when there is nothing to show.

    Not a one-click copy -- the SDK has no clipboard component, and the
    previous custom-HTML button that faked one was not reliably clickable in
    the real panel. ``ui.Code`` puts the WHOLE prompt (not just the visible
    part of a truncated title) somewhere it can be selected and copied with
    the browser's own Ctrl+C, which cannot silently fail the way a blocked
    inline JS handler did.
    """
    if not prompt.strip():
        return None
    return ui.Code(content=prompt)
