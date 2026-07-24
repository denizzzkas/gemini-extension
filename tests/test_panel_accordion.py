"""Open/close cycle of the inline image accordion, with a HOST SIMULATOR.

Why this file exists
--------------------
The "Hide" button shipped broken and the existing tests all passed, because
every one of them calls the panel handler with a FRESH params dict. The real
Panel app does not do that. Per docs.imperal.io/en/concepts/panels:

    "Params accumulate: the Panel app accumulates params per-panel_id.
     Re-fetches merge ``{}`` into the accumulated params."

So a param-less ``ui.Call("__panel__x")`` cannot clear an already-set
``generation_id`` -- the old value survives the merge and the image stays
open. Testing each click in isolation makes that bug invisible.

:class:`PanelHost` below reproduces the accumulation, so a test can click
View -> Hide the way a user does and assert on what the user would SEE.
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


@pytest.mark.parametrize("panel_id", ["gemini_quick", "gemini_studio"])
@pytest.mark.asyncio
async def test_view_then_hide_collapses_the_image(panel_id):
    """The regression: View expands, Hide must actually collapse.

    Runs through PanelHost so accumulated params are in play -- the exact
    condition under which the old param-less Hide silently did nothing.
    """
    import main  # noqa: F401  (registers panels)
    ctx = make_ctx(with_key=True)
    await _seed_image(ctx)
    host = PanelHost(ctx, panel_id)

    closed = await host.render()
    assert not _images(closed), "history list must not render bytes unasked"

    view = _button(closed, "View image")
    assert view is not None, "no 'View image' button rendered"
    opened = await host.click(view["props"]["on_click"])
    assert _images(opened), "'View image' did not expand the image"

    hide = _button(opened, "Hide")
    assert hide is not None, "no 'Hide' button while the image is open"
    hidden = await host.click(hide["props"]["on_click"])

    assert not _images(hidden), (
        "'Hide' did not collapse the image -- the accumulated generation_id "
        "survived the re-fetch (see module docstring)"
    )
    assert _button(hidden, "View image") is not None, (
        "after hiding, the row must offer 'View image' again"
    )


@pytest.mark.parametrize("panel_id", ["gemini_quick", "gemini_studio"])
@pytest.mark.asyncio
async def test_view_hide_view_is_repeatable(panel_id):
    """Closing must not poison the state: the user can re-open afterwards."""
    import main  # noqa: F401
    ctx = make_ctx(with_key=True)
    await _seed_image(ctx)
    host = PanelHost(ctx, panel_id)

    tree = await host.render()
    for cycle in range(2):
        view = _button(tree, "View image")
        assert view is not None, f"cycle {cycle}: no 'View image'"
        tree = await host.click(view["props"]["on_click"])
        assert _images(tree), f"cycle {cycle}: image did not open"

        hide = _button(tree, "Hide")
        assert hide is not None, f"cycle {cycle}: no 'Hide'"
        tree = await host.click(hide["props"]["on_click"])
        assert not _images(tree), f"cycle {cycle}: image did not close"


@pytest.mark.asyncio
async def test_refresh_button_also_collapses_an_open_image():
    """Refresh claims to collapse too -- it needs the same reset, not a bare call."""
    import main  # noqa: F401
    ctx = make_ctx(with_key=True)
    await _seed_image(ctx)
    host = PanelHost(ctx, "gemini_quick")

    tree = await host.render()
    tree = await host.click(_button(tree, "View image")["props"]["on_click"])
    assert _images(tree)

    refresh = _button(tree, "Refresh")
    assert refresh is not None, "no 'Refresh' button in the panel"
    tree = await host.click(refresh["props"]["on_click"])
    assert not _images(tree), "'Refresh' left the image expanded"


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
