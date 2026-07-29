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
so a huge original does not risk the whole panel response. Above that
ceiling, the honest fallback is the one channel
already proven end-to-end: chat, via ``view_full_resolution_block`` below,
which now renders correctly since get_original_media returns
``image_base64``/``video_base64`` (the SAME field names generate_image /
generate_video use), not the differently-named field that used to make the
button print a bare id instead of a picture.

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
    "DOWNLOAD_CEILING_CHARS", "download_block",
    "view_full_resolution_block", "copy_prompt_block",
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


def download_block(raw: bytes, mime_type: str, filename: str) -> ui.UINode:
    """A real download: an anchor with the ``download`` attribute.

    ``raw`` must be the untouched bytes as stored -- nothing here re-encodes,
    resizes or recompresses. Above ``DOWNLOAD_CEILING_CHARS`` this returns an
    honest message pointing at chat instead of a button that may silently
    fail, since that size is unproven for a panel response of any shape.
    """
    encoded = base64.b64encode(raw).decode()
    if len(encoded) > DOWNLOAD_CEILING_CHARS:
        return ui.Alert(
            title="Too large to download here",
            message=(
                f"This file is {len(raw) // 1024} KB, over what a panel "
                "response can reliably carry. Use \"View full resolution in "
                "chat\" below instead -- that channel handles files of any "
                "size."
            ),
            type="warn",
        )

    safe_name = html.escape(filename, quote=True)
    href = f"data:{mime_type};base64,{encoded}"
    return ui.Html(
        content=(
            f'<a href="{href}" download="{safe_name}" '
            'style="display:inline-block;padding:9px 15px;border-radius:8px;'
            'background:#5b8def;color:#fff;font:600 13px system-ui,sans-serif;'
            'text-decoration:none">Download the original file</a>'
        ),
        sandbox=False,
        max_height=64,
    )


def view_full_resolution_block(generation_id: str, kind: str = "image") -> ui.UINode:
    """A button that asks Webbee, in chat, to show this generation at full size.

    Sends a plain chat message rather than calling a tool directly: the LLM
    turn that follows is what renders the returned ``image_base64`` /
    ``video_base64`` inline, exactly like every generation reply already
    does. The generation id is spelled out so the request is unambiguous.
    """
    noun = "video" if kind == "video" else "image"
    return ui.Stack(direction="v", gap=1, children=[
        ui.Button(
            label="View full resolution in chat",
            variant="primary",
            icon="Maximize",
            on_click=ui.Send(
                f"Show me the full-resolution original {noun} for "
                f"generation {generation_id}."
            ),
        ),
        ui.Text(
            "Opens a chat turn that fetches and shows the untouched file — "
            "full resolution, not this preview.",
            variant="caption",
        ),
    ])


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
