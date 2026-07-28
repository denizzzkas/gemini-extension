"""The "view full resolution" affordance, and the copy-prompt block.

Why this is no longer a download button
-----------------------------------------
Three implementations of an in-panel download were tried, in order, and all
three were disproven by real use, not by guesswork:

1. A raw ``<a download>`` anchor via ``ui.Html`` -- reported broken.
2. The same markup with ``sandbox=False`` -- still reported broken, which
   disproved the sandbox theory.
3. ``ui.Button`` + ``ui.Open`` pointed at a ``data:`` URI with an opaque MIME
   type -- this looked sound (native action, no custom JS) but Chrome has
   OUTRIGHT BLOCKED top-frame navigation to ``data:`` URIs since Chrome 60
   (2017, "Intent to Deprecate and Remove: Top-frame navigations to data
   URLs") -- confirmed against Chromium's own bug tracker and release notes,
   not assumed. ``ui.Open`` has no other way to trigger a save. This is why
   raising ``DOWNLOAD_CEILING_CHARS`` never helped: the button was never a
   size problem, it could not have worked at ANY size.

So this stops trying to save a file from inside the panel at all. Instead the
button hands off to the one channel already PROVEN to deliver a real,
full-resolution image end to end: chat. ``generate_image`` and
``list_generation_history`` already return ``image_base64``, and Webbee
renders that inline in her own reply -- this happens on every generation
today. ``ui.Send`` posts a real chat message (unlike ``ui.Call``, which
bypasses chat and calls the tool directly with no LLM turn to render
anything), so clicking the button asks Webbee, in a normal turn, to fetch and
show this exact generation's original -- the same rendering path already in
daily use, not a new and unverified one.

Copy: a selectable text block, not a clipboard button
--------------------------------------------------------
There is no native Copy/Clipboard component in the SDK (checked: the only
clipboard mention anywhere is a secrets panel that deliberately forbids it),
so a genuine one-click copy is not buildable from components alone. The
prompt is shown as ``ui.Code`` -- a plain, syntax-block text area the user
selects and copies with the browser's own Ctrl+C, which cannot silently fail
the way a blocked inline handler can.
"""
from __future__ import annotations

from imperal_sdk import ui

__all__ = ["view_full_resolution_block", "copy_prompt_block"]


def view_full_resolution_block(generation_id: str, kind: str = "image") -> ui.UINode:
    """A button that asks Webbee, in chat, to show this generation at full size.

    Sends a plain chat message rather than calling a tool directly (see the
    module docstring for why): the LLM turn that follows is what actually
    renders the returned ``image_base64`` inline, exactly like every
    generation reply already does. The generation id is spelled out so the
    request is unambiguous regardless of phrasing.
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
