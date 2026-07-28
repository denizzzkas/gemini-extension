"""The "view full resolution" button and the copy-prompt block.

Why these are tested apart from the detail view
-------------------------------------------------
Three prior implementations of an in-panel download were tried and disproven
by real use, not by these tests -- which is exactly why this suite exists as
a regression guard rather than a first line of defence:

1. A raw ``<a download>`` anchor via ``ui.Html`` -- the user reported the
   download button either did not appear, or spun forever when clicked.
2. Removing ``sandbox=True`` on that same markup -- the user reported it was
   STILL unreliable, disproving the sandbox theory.
3. ``ui.Button`` + ``ui.Open`` on a ``data:`` URI with an opaque MIME type --
   looked sound, but Chrome has categorically blocked top-frame navigation to
   ``data:`` URIs since 2017 regardless of payload size, so it could never
   have worked. Confirmed against real sources, not assumed.

The button now hands off to chat via ``ui.Send`` instead -- the one channel
already proven, in daily use, to render a full-resolution image
(``generate_image``'s own reply does exactly this). These tests pin that no
HTML/script/data-URI trick crept back in, and that the copy block stays a
plain, native, selectable text component.
"""
from __future__ import annotations

from handlers.panel_html import copy_prompt_block, view_full_resolution_block


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


def test_view_full_resolution_uses_a_native_button_and_send_action():
    """No ``ui.Html``/custom markup, and no ``data:`` URI navigation.

    Both prior tricks were reported unreliable (or, for the data: URI, are
    now known to be categorically impossible in Chrome). The fix is a plain
    Button + a real chat Send action instead.
    """
    node = view_full_resolution_block("gen-123", "image")
    button = _find(node, "Button")
    assert button is not None, "must render as a native Button"
    action = button["props"].get("on_click", {})
    assert action.get("action") == "send", \
        "must use ui.Send so a real chat turn renders the result"
    assert "message" in action, "the Send action must carry a message"
    assert "Html" not in {n.get("type") for n in _walk(node)}, \
        "must not fall back to raw HTML"
    assert "data:" not in action["message"], \
        "must not smuggle a data: URI back in through the message"


def test_view_full_resolution_message_names_the_generation_and_kind():
    """The chat message must be unambiguous about WHICH generation and WHAT
    kind of media, since the user may have many generations open at once."""
    node = view_full_resolution_block("gen-abc-999", "video")
    button = _find(node, "Button")
    message = button["props"]["on_click"]["message"]
    assert "gen-abc-999" in message
    assert "video" in message


def test_view_full_resolution_defaults_to_image():
    node = view_full_resolution_block("gen-xyz")
    button = _find(node, "Button")
    message = button["props"]["on_click"]["message"]
    assert "image" in message


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
    "...".'). A selectable text block has no JS string literal or HTML
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
