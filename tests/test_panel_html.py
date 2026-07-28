"""The download button and the copy-prompt block, now native components.

Why these are tested apart from the detail view
-------------------------------------------------
Two prior implementations of both features were tried and disproven by real
use, not by these tests -- which is exactly why this suite exists as a
regression guard rather than a first line of defence:

1. A raw ``<a download>`` anchor via ``ui.Html`` -- the user reported the
   download button either did not appear, or spun forever when clicked.
2. Removing ``sandbox=True`` on that same markup -- the user reported it was
   STILL unreliable, disproving the sandbox theory.

Both features are now built exclusively from components already proven
reliable elsewhere in this panel: ``ui.Button`` + ``ui.Open`` (the same action
class used by every working "View image"/"Regenerate" button), and ``ui.Code``
for the prompt (a plain selectable text block, no custom JS). These tests pin
that no HTML/script strings crept back in, and that the download's MIME
override is the mechanism making the browser save rather than display it.
"""
from __future__ import annotations

import base64

from handlers.panel_html import (
    DOWNLOAD_CEILING_CHARS, copy_prompt_block, download_block,
)


def _walk(node):
    d = node.to_dict()
    yield d
    for v in d.get("props", {}).values():
        if isinstance(v, dict) and v.get("type"):
            yield from _walk(_Wrap(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and item.get("type"):
                    yield from _walk(_Wrap(item))


class _Wrap:
    """Lets a raw dict be re-walked through the same generator."""
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


def _find(tree, node_type):
    return next((n for n in _walk(tree) if n.get("type") == node_type), None)


def test_download_button_uses_native_open_action_not_raw_html():
    """No ``ui.Html``/custom markup -- a plain Button + Open action.

    Raw HTML was tried twice and was not reliably clickable in the real panel
    either time. This pins that the fix does not quietly reintroduce it.
    """
    node = download_block(b"some bytes", "image/jpeg", "gemini-abc.jpg")
    button = _find(node, "Button")
    assert button is not None, "must render as a native Button"
    action = button["props"].get("on_click", {})
    assert action.get("action") == "open", "must use the native Open action"
    assert "url" in action, "the Open action must carry a url"
    assert "Html" not in {n.get("type") for n in _walk(node)}, \
        "must not fall back to raw HTML"


def test_download_hands_over_the_original_byte_for_byte():
    """The bytes in the Open URL must decode back to exactly what was stored.

    Not "an image", not "a resized copy" -- the same bytes. Anything else means
    the user silently receives the preview when they asked for the original.
    """
    raw = bytes(range(256)) * 40  # arbitrary binary, not valid image data
    node = download_block(raw, "image/jpeg", "gemini-abc.jpg")
    button = _find(node, "Button")
    url = button["props"]["on_click"]["url"]

    payload = url.split("base64,", 1)[1]
    assert base64.b64decode(payload) == raw, \
        "the download URL must carry the ORIGINAL bytes, unmodified"


def test_download_forces_an_opaque_mime_so_the_browser_saves_it():
    """The MIME must NOT be the image's real type, or the browser just displays it.

    ``ui.Open`` navigates a tab to the URL; navigating to ``image/jpeg`` opens
    the picture inline instead of downloading it. ``application/octet-stream``
    is what makes the browser treat it as a file to save.
    """
    node = download_block(b"bytes", "image/jpeg", "x.jpg")
    button = _find(node, "Button")
    url = button["props"]["on_click"]["url"]
    assert url.startswith("data:application/octet-stream;base64,"), \
        "must override the real MIME so the browser downloads rather than displays"


def test_an_absurd_payload_is_refused_with_an_explanation():
    """Over the ceiling the user gets an honest message, not a dead button.

    A ceiling that silently produced nothing would reproduce the original
    complaint in a new form.
    """
    huge = b"x" * (DOWNLOAD_CEILING_CHARS)  # base64 expands ~4/3, so over it
    node = download_block(huge, "image/png", "huge.png")
    d = node.to_dict()
    assert d["type"] == "Alert", "an oversized original must explain itself"
    import json
    assert "KB" in json.dumps(d), "the message should state the actual size"


def test_copy_block_uses_native_code_component_not_raw_html():
    """No custom clipboard JS -- a plain, selectable ``ui.Code`` block.

    The previous ``onclick="navigator.clipboard..."`` button was reported not
    clickable in the real panel. A selectable text block cannot silently fail
    the way a blocked inline handler can.
    """
    node = copy_prompt_block("a lighthouse in fog")
    d = node.to_dict()
    assert d["type"] == "Code", "must render as a native Code block"
    assert "Html" not in {n.get("type") for n in _walk(node)}


def test_copy_block_carries_the_whole_prompt_not_a_truncation():
    """The point of the block: the ENTIRE prompt, verbatim.

    Showing a shortened prompt is worse than none, because the user cannot see
    that anything is missing.
    """
    prompt = "a lighthouse in fog, " + "extremely detailed, " * 40
    node = copy_prompt_block(prompt)
    assert node.to_dict()["props"]["content"] == prompt, \
        "the full, untruncated prompt text must be present"


def test_copy_block_survives_a_hostile_prompt():
    """Quotes, newlines, backslashes and markup must not need any escaping.

    Real prompts contain apostrophes and quotes constantly ('in the style of
    "..."'). A selectable text block has no JS string literal or HTML
    attribute to break out of, unlike the previous implementation -- this pins
    that the raw prompt survives completely unmodified.
    """
    prompt = (
        'she said "hi" and Denis\'s cat\nback\\slash '
        "</script><img src=x onerror=alert(1)>"
    )
    node = copy_prompt_block(prompt)
    assert node.to_dict()["props"]["content"] == prompt, \
        "the shown text must be exactly the prompt, unescaped and unmodified"


def test_copy_block_is_absent_for_an_empty_prompt():
    """No prompt, no block -- a block that shows nothing is noise."""
    assert copy_prompt_block("") is None
    assert copy_prompt_block("   ") is None
