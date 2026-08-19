"""The raw-HTML affordances: saving the original, and copying the prompt.

Why these are tested apart from the detail view
-----------------------------------------------
Both are strings of hand-built markup, so a mistake here is not a type error --
it is a button that looks perfect and does nothing, which is exactly how the
download shipped broken THREE times in a row (see handlers/panel_html.py for
the full history of what was tried and what the evidence actually shows: a
sandboxed iframe silently blocks saves, a data: URI works fine for an
attribute-driven ``download``, and Chrome only ever blocked page-initiated
top-frame *navigation* to data:, never that mechanism). Two failures in
particular are invisible to any test that merely renders the panel:

* ``sandbox=True`` (the ``ui.Html`` DEFAULT) wraps the markup in an iframe whose
  sandbox attribute has no ``allow-downloads``, so the browser blocks the save
  silently -- the click is swallowed and the UI spins forever. Nothing about the
  rendered tree looks wrong. :func:`test_download_is_not_sandboxed` pins it.
* a prompt containing a quote or a newline can break out of the JS string
  literal in the copy handler, producing a button that throws in the console and
  does nothing. :func:`test_copy_button_survives_a_hostile_prompt` feeds it the
  characters that would do it.

The download tests also assert the thing the user asked to be certain of: what
is handed over is the ORIGINAL, byte for byte, not the preview.
"""
from __future__ import annotations

import base64
import json

from handlers.panel_html import (
    DOWNLOAD_CEILING_CHARS, copy_prompt_block, download_block,
)


def _content(node) -> str:
    return node.to_dict()["props"].get("content", "")


def _props(node) -> dict:
    return node.to_dict()["props"]


def test_download_hands_over_the_original_byte_for_byte():
    """The bytes in the anchor must decode back to exactly what was stored.

    Not "an image", not "a resized copy" -- the same bytes. Anything else means
    the user silently receives the preview when they asked for the original.
    """
    raw = bytes(range(256)) * 40  # arbitrary binary, not valid image data
    node = download_block(raw, "image/jpeg", "gemini-abc.jpg")

    content = _content(node)
    assert 'download="' in content, "without the download attribute a browser navigates"

    payload = content.split("base64,", 1)[1].split('"', 1)[0]
    assert base64.b64decode(payload) == raw, \
        "the anchor must carry the ORIGINAL bytes, unmodified"


def test_download_is_not_sandboxed():
    """THE regression guard for the download that hung forever.

    A sandboxed iframe without ``allow-downloads`` has its downloads blocked by
    the browser, with no error -- the exact reported symptom. The SDK cannot add
    sandbox tokens, so the only way the anchor works is unsandboxed.
    """
    node = download_block(b"bytes", "image/png", "x.png")
    assert _props(node).get("sandbox") is False, \
        "a sandboxed download is silently blocked -- this is the hang"


def test_download_filename_cannot_break_out_of_the_attribute():
    """A quote in the filename must not be able to inject attributes."""
    node = download_block(b"bytes", "image/png", 'evil" onclick="alert(1)')
    content = _content(node)
    assert 'onclick="alert(1)"' not in content, "filename escaped into markup"
    assert "&quot;" in content or "&#x27;" in content, "the quote must be escaped"


def test_an_absurd_payload_is_refused_with_an_explanation():
    """Over the ceiling the user gets an honest message, not a dead button.

    A ceiling that silently produced nothing would reproduce the original
    complaint in a new form.
    """
    huge = b"x" * (DOWNLOAD_CEILING_CHARS)  # base64 expands ~4/3, so over it
    node = download_block(huge, "image/png", "huge.png")
    d = node.to_dict()
    assert d["type"] == "Alert", "an oversized original must explain itself"
    assert "KB" in json.dumps(d), "the message should state the actual size"


def test_download_falls_back_to_cached_preview_when_original_missing():
    """No raw bytes (e.g. a failed/slow storage read) must not mean the panel
    lies -- it must plainly say a preview is shown and point at the real
    download path (the webhook link), instead of a second '<a download>'
    anchor over the SAME cached preview bytes that real clicks showed does
    not fire (unlike the primary anchor above it -- see download_block's own
    comment for why this one was removed rather than left as a dead button).
    """
    fallback = b"\x89PNG\r\n\x1a\n" + b"fake-preview-bytes"
    node = download_block(
        None, "image/jpeg", "gen.jpg",
        fallback_b64=base64.b64encode(fallback).decode(),
        fallback_mime="image/png",
    )
    d = node.to_dict()
    assert d["type"] == "Text", "a dead second download anchor must not be rendered"
    content = json.dumps(d)
    assert "preview" in content.lower()


def test_download_falls_back_to_cached_preview_when_original_too_large():
    """Over the ceiling, the panel must say a preview is shown rather than
    just an alert -- but not via a second download anchor that real clicks
    showed does not work (see download_block's own comment for why)."""
    huge = b"x" * DOWNLOAD_CEILING_CHARS
    fallback = base64.b64encode(b"small-preview-bytes").decode()
    node = download_block(
        huge, "image/png", "huge.png",
        fallback_b64=fallback, fallback_mime="image/png",
    )
    d = node.to_dict()
    assert d["type"] == "Text", "a dead second download anchor must not be rendered"


def test_download_is_an_honest_alert_when_nothing_is_available():
    """Neither the original nor a cached preview exist -- there really is
    nothing to hand over, and the panel must say so plainly."""
    node = download_block(None, "image/jpeg", "gen.jpg")
    d = node.to_dict()
    assert d["type"] == "Alert"
    assert "No download available" in json.dumps(d)


def test_copy_button_carries_the_whole_prompt_not_a_truncation():
    """The point of the block: the ENTIRE prompt, every word, none dropped.

    Showing a shortened prompt is worse than showing nothing, because the
    user cannot see that anything is missing. The block now hard-wraps long
    text (so a Code block never needs horizontal scrolling -- see
    ``_wrap_for_display``), which rewrites SPACING (turns some spaces into
    newlines) but must never drop or truncate a single word.
    """
    prompt = "a lighthouse in fog, " + "extremely detailed, " * 40
    node = copy_prompt_block(prompt)

    assert node.to_dict()["type"] == "Code", (
        "a real, selectable text block -- not a hand-rolled HTML/JS button "
        "(see the function's own docstring for why: no JS click handler "
        "has ever been proven to fire on this surface, in this repo or in "
        "any other published Marketplace extension's public source)"
    )
    content = _content(node)
    # Wrapping may turn spaces into newlines, so compare words, not bytes.
    assert content.split() == prompt.split(), (
        "every word of the prompt must be present, in order -- wrapping may "
        "reflow whitespace but must never drop or reorder content"
    )
    assert "…" not in content and "..." not in content, "the prompt was truncated"


def test_copy_button_survives_a_hostile_prompt():
    """Quotes, newlines, backslashes and markup must survive byte-for-byte.

    Real prompts contain apostrophes and quotes constantly ('in the style of
    "..."'), so this is ordinary input, not just an attack. ``ui.Code`` shows
    raw text content -- there is no JS string literal or HTML attribute to
    escape into any more, so the only invariant that matters is an exact,
    unmodified round-trip of whatever the caller passed in.
    """
    prompt = (
        'she said "hi" and Denis\'s cat\nback\\slash '
        "</script><img src=x onerror=alert(1)>"
    )
    node = copy_prompt_block(prompt)
    content = _content(node)

    assert content == prompt, (
        "the block must carry the prompt exactly as given, with no escaping "
        "or mangling -- ui.Code renders raw text, not markup"
    )
    assert node.to_dict()["props"].get("language") == "text"


def test_copy_button_never_needs_horizontal_scroll():
    """No line in the rendered block may exceed the wrap column.

    This is the actual point of wrapping at all: a Code block with no
    ``wrap`` prop (see ``imperal_sdk/ui/display.py``) renders whatever
    lines it is given verbatim, so a single very long line -- an ordinary
    one-paragraph image prompt -- would force horizontal scrolling. Every
    line here must stay within the wrap width, even a pathological case
    with no spaces at all (a long URL-like token) that an ordinary word-wrap
    could not break on its own.
    """
    from handlers.panel_html import _WRAP_COLUMNS

    long_paragraph = "a lighthouse in fog, " + "extremely detailed, " * 40
    no_spaces = "x" * 500  # nothing to break on except mid-word

    for prompt in (long_paragraph, no_spaces, long_paragraph + "\n" + no_spaces):
        content = _content(copy_prompt_block(prompt))
        for line in content.split("\n"):
            assert len(line) <= _WRAP_COLUMNS, (
                f"line exceeds the wrap width and would force horizontal "
                f"scroll: {line!r}"
            )


def test_copy_button_is_absent_for_an_empty_prompt():
    """No prompt, no button -- a button that copies nothing is noise."""
    assert copy_prompt_block("") is None
    assert copy_prompt_block("   ") is None


def test_broken_full_resolution_chat_button_is_not_exposed():
    """A chat button that only prints an ID must not remain in the UI."""
    import handlers.panel_html as panel_html

    assert not hasattr(panel_html, "view_full_resolution_block")
