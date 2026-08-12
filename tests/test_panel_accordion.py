"""Open/close cycle of the generation detail view, with a HOST SIMULATOR.

Why this file exists
--------------------
The "Hide" button shipped broken once, and the existing tests all passed,
because every one of them called the panel handler with a FRESH params dict.
The real Panel app does not do that. Per docs.imperal.io/en/concepts/panels:

    "Params accumulate: the Panel app accumulates params per-panel_id.
     Re-fetches merge ``{}`` into the accumulated params."

So a param-less ``ui.Call(\"__panel__x\")`` cannot clear an already-set
``generation_id`` -- the old value survives the merge and the image stays
open. Testing each click in isolation makes that bug invisible.

:class:`PanelHost` below reproduces the accumulation, so a test can click
through the way a user does and assert on what the user would SEE.

Where the open/close cycle lives now
-------------------------------------
Full detail no longer expands INLINE inside a "gemini_quick" history card --
that design is exactly what overloaded the left panel (see handlers/
panel_history.py). Opening a generation now navigates to the "gemini_studio"
centre panel (``ui.Call(\"__panel__gemini_studio\", generation_id=...)``), and
closing it re-renders THAT panel with the close sentinel -- a self-call,
same accumulation rules, just on the other panel_id. So this suite now drives
"gemini_studio" through PanelHost instead of "gemini_quick".
"""
from __future__ import annotations

import pytest

from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx


class PanelHost:
    """Minimal stand-in for the Panel app's per-panel_id param accumulation."""

    def __init__(self, ctx, panel_id: str):
        self.ctx = ctx
        self.panel_id = panel_id
        self.params: dict = {}

    async def click(self, action: dict) -> dict:
        """Dispatch a serialized ui.Call action and return the rendered tree.

        Mirrors the documented merge: params from the action are merged INTO
        the accumulated ones, never replacing the dict wholesale.
        """
        assert action["action"] == "call", action
        assert action["function"] == f"__panel__{self.panel_id}", (
            f"button targets {action['function']!r}, not its own panel"
        )
        incoming = {k: v for k, v in action.items() if k not in ("action", "function")}
        nested = incoming.pop("params", None)
        if isinstance(nested, dict):
            incoming.update(nested)
        self.params.update(incoming)          # <-- the accumulation
        return await self.render()

    async def render(self) -> dict:
        from app import ext
        result = await ext._tools[f"__panel__{self.panel_id}"].func(
            self.ctx, **self.params,
        )
        return result["ui"] if isinstance(result, dict) and "ui" in result else result


def _walk(node):
    """Yield every dict node in a serialized UI tree."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _images(tree) -> list[dict]:
    return [n for n in _walk(tree) if n.get("type") == "Image"]


def _button(tree, label: str) -> dict | None:
    for n in _walk(tree):
        if n.get("type") == "Button" and n.get("props", {}).get("label") == label:
            return n
    return None


async def _seed_image(ctx, prompt: str = "a red apple"):
    path = "gemini/image/accordion.png"
    await ctx.storage.upload(path, b"\x89PNG\r\n\x1a\n" + b"pixels", content_type="image/png")
    return await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id,
        "kind": "image",
        "prompt": prompt,
        "model": "gemini-3.1-flash-lite-image",
        "storage_path": path,
        "mime_type": "image/png",
        "created_at": "2026-07-24T21:00:00Z",
    })


@pytest.mark.asyncio
async def test_image_info_opens_the_studio_panel_and_close_collapses_it():
    """The regression this guards: opening must show the image, and Close
    must actually collapse it back to the empty state -- not leave the
    accumulated generation_id stuck (the exact bug \"Hide\" used to have).

    Runs through PanelHost so accumulated params are in play. The "Image
    info" button now lives on the history card, which renders only inside
    gemini_studio itself (gemini_quick holds no history at all any more) --
    so it is found in studio's OWN empty-state-then-populated render, not in
    gemini_quick.
    """
    import main  # noqa: F401  (registers panels)

    ctx = make_ctx(with_key=True)
    doc = await _seed_image(ctx)

    studio = PanelHost(ctx, "gemini_studio")
    listing = await studio.render()
    assert not _images(listing), "studio's list view must not inline any image"

    info = _button(listing, "Image info")
    assert info is not None, "no 'Image info' button rendered in gemini_studio's list"
    assert info["props"]["on_click"]["function"] == "__panel__gemini_studio"

    opened = await studio.click(info["props"]["on_click"])
    assert _images(opened), "opening via 'Image info' did not show the image"

    close = _button(opened, "Close")
    assert close is not None, "no 'Close' button while a generation is open"
    closed = await studio.click(close["props"]["on_click"])

    assert not _images(closed), (
        "'Close' did not collapse the studio view -- the accumulated "
        "generation_id survived the re-fetch (see module docstring)"
    )


@pytest.mark.asyncio
async def test_open_close_open_is_repeatable():
    """Closing must not poison the state: the user can re-open afterwards."""
    import main  # noqa: F401

    ctx = make_ctx(with_key=True)
    doc = await _seed_image(ctx)

    studio = PanelHost(ctx, "gemini_studio")
    tree = await studio.render()
    info = _button(tree, "Image info")
    assert info is not None, "no 'Image info' button rendered in gemini_studio's list"
    for cycle in range(2):
        tree = await studio.click(info["props"]["on_click"])
        assert _images(tree), f"cycle {cycle}: image did not open"

        close = _button(tree, "Close")
        assert close is not None, f"cycle {cycle}: no 'Close'"
        tree = await studio.click(close["props"]["on_click"])
        assert not _images(tree), f"cycle {cycle}: image did not close"
        info = _button(tree, "Image info")
        assert info is not None, f"cycle {cycle}: 'Image info' missing after close"


@pytest.mark.asyncio
async def test_sentinel_is_never_mistaken_for_a_record_id():
    """The close sentinel must not resolve to a generation or crash lookup."""
    import main  # noqa: F401
    from handlers.panel_viewer import CLOSED_SENTINEL, _find_generation, _opened_id

    ctx = make_ctx(with_key=True)
    await _seed_image(ctx)

    assert _opened_id({"generation_id": CLOSED_SENTINEL}) == ""
    assert _opened_id({"params": {"generation_id": CLOSED_SENTINEL}}) == ""
    # _find_generation returns (doc_or_None, lookup_failed): the sentinel must
    # resolve to "no such record" WITHOUT being reported as a storage failure,
    # so it can never surface as a scary error to the user.
    doc, lookup_failed = await _find_generation(ctx, CLOSED_SENTINEL)
    assert doc is None
    assert lookup_failed is False
