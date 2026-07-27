"""The two raw-HTML affordances the declarative components cannot express.

Both live here rather than in ``handlers/panel_detail.py`` so that file stays
under the 300-line limit the deploy validator enforces (exceeding it has cost a
deploy point before).

Why raw HTML at all
-------------------
Two things a panel needs are simply not in the component set, checked against
the SDK rather than assumed:

* **Saving a file.** A download needs an anchor's ``download`` attribute.
  ``ui.Link`` emits a plain href, so a browser NAVIGATES to a ``data:`` URI
  instead of saving it, and ``ui.Open`` only opens a tab.
* **Copying text.** There is no Copy/Clipboard component and no ``copyable=``
  flag on ``Text``/``Code`` (searched the whole SDK: the only clipboard mention
  is a secrets panel that deliberately forbids it).

``sandbox=False`` on both, and why it is the actual bug fix
----------------------------------------------------------
``ui.Html`` defaults to ``sandbox=True``, which wraps the markup in
``<iframe sandbox="...">``. That attribute, with no ``allow-downloads`` token,
makes the browser BLOCK the download outright: the click produces no file and no
error -- it is swallowed. That is precisely the "download spins forever" symptom.
``navigator.clipboard`` is unavailable in the same situation. The SDK exposes no
way to add sandbox tokens, so not sandboxing is the only way either affordance
can work.

The safety this gives up is bounded deliberately: every byte of markup below is
built by these two functions, and the only interpolated values are a ``data:``
URI of the user's own image, an ``html.escape``d filename, and a prompt passed
through ``json.dumps`` (a correct JS string literal) and then ``html.escape``d
so it cannot close the element. No unescaped model output reaches the DOM.
"""
from __future__ import annotations

import base64
import html
import json

from imperal_sdk import ui

__all__ = ["DOWNLOAD_CEILING_CHARS", "download_block", "copy_prompt_block"]

# A SAFETY VALVE, not a measured limit.
#
# What IS measured, in production: an image rendered through ``ui.Image`` stops
# appearing somewhere between ~127k base64 chars (renders) and ~954k (does not).
# An anchor is a different case -- the browser never decodes or lays out those
# bytes, it only writes them to disk -- so that number must not simply be
# assumed to apply here. This is set high enough to hand over a normal render
# (real originals measure 571k-1005k chars) and low enough to refuse something
# absurd, with an honest message instead of a silent failure when it trips.
DOWNLOAD_CEILING_CHARS = 2_000_000


def download_block(raw: bytes, mime_type: str, filename: str) -> ui.UINode:
    """An anchor that saves the ORIGINAL bytes -- never the preview.

    ``raw`` must be the untouched bytes as stored, so what lands on disk is
    byte-for-byte what the model returned. Nothing here re-encodes, resizes or
    recompresses; that is the entire contract of this function.
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


def copy_prompt_block(prompt: str) -> ui.UINode | None:
    """A one-click copy of the WHOLE prompt, or None when there is nothing to copy.

    Selecting a long prompt by hand in a ~380px column is miserable, and the
    point of the button is that it takes the entire text, not the part that
    happens to be visible.

    The escaping is ``quote=True``, and that is not a detail
    -----------------------------------------------------
    The prompt ends up inside an ``onclick`` attribute delimited by SINGLE
    quotes. ``json.dumps`` produces a valid JS string literal but does not
    escape apostrophes, and ``html.escape(..., quote=False)`` leaves them alone
    too -- so the single most ordinary thing a prompt can contain, an English
    apostrophe ("a lighthouse in Denis's harbour"), terminated the attribute
    early and produced a button that silently did nothing. ``quote=True``
    encodes ``'`` and ``"`` as character references, which the HTML parser
    decodes back into the JS string AFTER the attribute boundary is settled --
    correct and injection-safe in one step.
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
            '.then(function(){this.textContent=\"Copied\";}.bind(this))'
            "'>Copy the full prompt</button>"
        ),
        sandbox=False,
        max_height=56,
    )
