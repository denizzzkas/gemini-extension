"""Shared helpers for the reworked-panel tests.

Not a test module -- it holds the tree walker and the record factory that both
``test_panel_tabs_refs.py`` and ``test_panel_references.py`` need, so neither
file has to import the other and neither grows past the 300-line limit the
deploy validator warns about.

The walker is the important piece. A naive recursive walk that only follows
``props`` values silently skips everything inside ``Tabs``, because a tab is a
plain ``{"label", "content"}`` dict, not a UINode. A test using such a walk
finds no form, asserts nothing, and PASSES -- that happened during development
and briefly looked like a bug in the panel itself. ``_walk`` descends through
dicts and lists alike, and ``test_the_walker_descends_into_tabs`` proves it can
see tab contents.
"""
from __future__ import annotations

import base64

from gemini_config import GENERATION_LOG_COLLECTION, MODEL_IMAGE


def _walk(node):
    """Yield every ``(type, props)`` in a rendered tree, INCLUDING inside Tabs.

    Descends through dicts and lists alike. The list case is what makes tab
    contents visible: ``Tabs`` stores ``[{"label", "content"}, ...]``, so a
    walker that only follows ``props`` values never reaches the forms.
    """
    if isinstance(node, dict):
        if node.get("type"):
            yield node["type"], node.get("props", {})
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _types(tree) -> list[str]:
    return [t for t, _ in _walk(tree)]


def _labels(tree) -> list[str]:
    return [p.get("label") for t, p in _walk(tree) if t == "Button" and p.get("label")]


async def _an_image(ctx, prompt="a red fox in snow", with_preview=True):
    data = {
        "user_id": ctx.user.imperal_id, "kind": "image", "prompt": prompt,
        "model": MODEL_IMAGE, "storage_path": "gemini/image/a.jpg",
        "mime_type": "image/jpeg", "created_at": "2026-07-27T10:00:00Z",
        "source": "generated",
    }
    if with_preview:
        data["preview_b64"] = base64.b64encode(b"fakepreviewbytes").decode()
        data["preview_mime"] = "image/png"
    return await ctx.store.create(GENERATION_LOG_COLLECTION, data)
