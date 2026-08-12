"""The "view full resolution" affordance, the copy-prompt button, and download.

History of the download button, and what actually changed
-----------------------------------------------------------
Three implementations were tried, and real clicks -- not this file's own
comments -- are the only evidence that counts:

1. A raw ``<a download>`` anchor via ``ui.Html`` (``sandbox=True``, the
   default) -- reported broken. ``sandbox=True`` wraps the markup in an
   ``<iframe sandbox="...">`` with no ``allow-downloads`` token, which
   silently swallows the click: no file, no error. This alone explains the
   report without needing any theory about ``data:`` URIs at all.
2. The same markup with ``sandbox=False`` -- also reported broken. This is
   the one genuinely unresolved data point: MDN documents the ``download``
   attribute on an ``<a>`` as the sanctioned way to force a save from a
   ``data:``/``blob:`` URI (a mechanism distinct from what Chrome actually
   blocks -- see below), so on paper this should have worked. It was
   replaced anyway rather than re-tried blindly a third time with no new
   evidence about why it failed for real.
3. ``ui.Button`` + ``ui.Open`` on a ``data:`` URI -- a comment here used to
   claim this "cannot ever work" because Chrome blocks ``data:`` navigation.
   Re-checked against Chrome's own release notes for that change (Chrome 60,
   2017): the blocked case is a page-INITIATED navigation of the TOP FRAME
   to a ``data:`` URL (``<a>`` without ``download``, ``window.open``,
   ``window.location``) -- Google's own text says plainly "The data: scheme
   will still work for resources loaded by a page." An anchor's ``download``
   attribute forces a save instead of a navigation, which is a different
   code path in the browser and is NOT what that deprecation removed. So the
   sweeping "impossible at any size" conclusion this file used to state was
   an overreach from a real fact, not a fact itself -- ``ui.Open`` bypassing
   the DOM anchor mechanism entirely was the real, narrower gap.

What this file does now
------------------------
``download_block`` brings back a real, native browser download: a plain
``<a download>`` anchor over ``ui.Html(sandbox=False)`` -- attempt 2 above,
which the public record (MDN, Chromium's own deprecation notice) supports as
sound, gated by its OWN size ceiling (DOWNLOAD_CEILING_CHARS below -- deliberately
not core/preview.PROVEN_GOOD_CHARS, see the constant's own comment for why)
so a huge original does not risk the whole panel response.

Every generation gets SOME download, not just the lucky ones
---------------------------------------------------------------
The user's explicit ask: every generation must offer a way to download its
result. Before this, "the original is None" (a failed/slow storage read) or
"the original is over the ceiling" both fell through to EITHER nothing at
all, or a dead-end alert -- while a small cached preview (the same PNG
thumbnail the history list already shows, see core/preview.py) often sat
right there in the record, unused. Now ``download_block`` accepts that
cached preview as a FALLBACK: when the original cannot be handed over, the
preview is offered instead, labelled honestly as a smaller stand-in, not a
silent swap. An alert with no download at all is now reserved for the one
case where truly nothing is available -- no original, no cached preview.

Format honesty: still not WEBP
-------------------------------
The user asked for at least a WEBP download option. Real WEBP encoding needs
an encoder this runtime does not have: no Pillow in production
(``pillow_available: false``, confirmed), and core/png.py + core/jpeg.py are
hand-written PNG/JPEG codecs with no WEBP support. Rather than mislabel a
PNG/JPEG file with a ``.webp`` extension (which would just hand the user a
file their OS or the target app may refuse to open), downloads keep their
REAL format and extension. This is a documented, known gap, not a silent
one -- closing it for real needs a WEBP encoder to become available in the
production runtime.

Copy: back to a real one-click button
--------------------------------------
There is no native Copy/Clipboard component in the SDK (checked: the only
clipboard mention anywhere is a secrets panel that deliberately forbids it).
The previous version of this file replaced the one-click copy button with a
plain ``ui.Code`` block (manual Ctrl+C) on the theory that the button was
"not reliably clickable" -- but the actual bug that was ever pinned down and
fixed was an *escaping* bug (an apostrophe in the prompt broke out of the
onclick attribute early, see the ``quote=True`` note below), not the
sandbox=False + onclick mechanism itself. Restoring the working button
rather than permanently downgrading to manual selection.
"""
from __future__ import annotations

import base64
import html
import json

from imperal_sdk import ui

__all__ = [
    "DOWNLOAD_CEILING_CHARS", "download_block", "copy_prompt_block",
]

# A SAFETY VALVE, not a measured limit -- and deliberately NOT the same
# number as core/preview.PROVEN_GOOD_CHARS (~127k).
#
# That 127k ceiling was measured for <Image> nodes, where the browser must
# DECODE and LAY OUT pixels -- a different code path from an <a href="data:">
# anchor, whose bytes the browser never decodes at all, only writes to disk.
# Reusing the Image ceiling here would make the download button unavailable
# for nearly every real generation (originals measure ~571k-1005k base64
# chars), defeating its purpose on an unconfirmed assumption that the two
# cases share a failure mode. This is set high enough to hand over a normal
# render and low enough to refuse something absurd, with an honest message
# instead of a silent failure when it trips.
DOWNLOAD_CEILING_CHARS = 2_000_000


def _download_anchor(encoded: str, mime_type: str, filename: str, label: str) -> ui.UINode:
    safe_name = html.escape(filename, quote=True)
    href = f"data:{mime_type};base64,{encoded}"
    safe_label = html.escape(label)
    return ui.Html(
        content=(
            f'<a href="{href}" download="{safe_name}" '
            'style="display:inline-block;padding:9px 15px;border-radius:8px;'
            'background:#5b8def;color:#fff;font:600 13px system-ui,sans-serif;'
            f'text-decoration:none">{safe_label}</a>'
        ),
        sandbox=False,
        max_height=64,
    )


def download_block(
    raw: bytes | None,
    mime_type: str,
    filename: str,
    *,
    fallback_b64: str | None = None,
    fallback_mime: str | None = None,
) -> ui.UINode:
    """A real download: an anchor with the ``download`` attribute.

    Every generation must offer SOME way to download its result -- so this
    now degrades in steps instead of an all-or-nothing original:

    1. ``raw`` present and under ``DOWNLOAD_CEILING_CHARS`` -> download the
       untouched original, byte for byte (nothing here re-encodes, resizes,
       or recompresses it).
    2. ``raw`` missing or over the ceiling, but a cached ``fallback_b64``
       preview exists (the same small thumbnail the history list already
       shows, see core/preview.py) -> offer THAT instead, labelled honestly
       as a smaller stand-in, not silently swapped in as if it were the
       original.
    3. Neither is available -> an honest alert. This is the one case where
       there really is nothing to hand over.

    Real WEBP is NOT produced here (see this module's own docstring for why:
    no Pillow in production, and the stdlib PNG/JPEG codecs this app ships
    have no WEBP support) -- downloads keep their real format/extension.
    """
    too_large = False
    if raw is not None:
        encoded = base64.b64encode(raw).decode()
        if len(encoded) <= DOWNLOAD_CEILING_CHARS:
            return _download_anchor(
                encoded, mime_type, filename, "Download the original file",
            )
        too_large = True

    if fallback_b64:
        fb_mime = fallback_mime or "image/png"
        fb_ext = "png" if "png" in fb_mime else ("jpg" if "jpeg" in fb_mime or "jpg" in fb_mime else "bin")
        fb_name = f"{filename.rsplit('.', 1)[0]}-preview.{fb_ext}"
        return ui.Stack(direction="v", gap=1, children=[
            _download_anchor(
                fallback_b64, fb_mime, fb_name, "Download a smaller preview",
            ),
            ui.Text(
                "The original is unavailable or too large for this panel -- "
                "this is a smaller stand-in, not the full-resolution file.",
                variant="caption",
            ),
        ])

    if too_large:
        return ui.Alert(
            title="Too large to download here",
            message=(
                f"This file is {len(raw) // 1024} KB, over what a panel "
                "response can reliably carry, and no smaller cached preview "
                "is available either. The original remains stored, but "
                "cannot be delivered through this panel right now."
            ),
            type="warn",
        )

    return ui.Alert(
        title="No download available",
        message=(
            "Neither the original file nor a cached preview could be "
            "retrieved for this generation, so there is nothing to hand "
            "over right now."
        ),
        type="warn",
    )


def copy_prompt_block(prompt: str) -> ui.UINode | None:
    """A one-click copy of the WHOLE prompt, or None when there is nothing to copy.

    The escaping is ``quote=True``, and that is not a detail
    -----------------------------------------------------
    The prompt ends up inside an ``onclick`` attribute delimited by SINGLE
    quotes. ``json.dumps`` produces a valid JS string literal but does not
    escape apostrophes, and ``html.escape(..., quote=False)`` leaves them
    alone too -- so the single most ordinary thing a prompt can contain, an
    English apostrophe ("a lighthouse in Denis's harbour"), terminated the
    attribute early and produced a button that silently did nothing.
    ``quote=True`` encodes ``'`` and ``"`` as character references, which the
    HTML parser decodes back into the JS string AFTER the attribute boundary
    is settled -- correct and injection-safe in one step. This was the real,
    previously-diagnosed bug; the fix is the escaping, not abandoning the
    button.
    """
    if not prompt.strip():
        return None

    literal = html.escape(json.dumps(prompt), quote=True)
    return ui.Html(
        content=(
            "<button "
            'style="padding:8px 13px;border-radius:8px;border:1px solid #4a5568;'
            "background:transparent;color:#cbd5e0;font:600 13px system-ui,"
            'sans-serif;cursor:pointer" '
            f"onclick='navigator.clipboard.writeText({literal})"
            '.then(function(){this.textContent="Copied";}.bind(this))'
            "'>Copy the full prompt</button>"
        ),
        sandbox=False,
        max_height=56,
    )
