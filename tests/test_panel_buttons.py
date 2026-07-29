"""The no-dead-buttons rule: every button must target somewhere real.

Split out of ``test_panel.py`` to keep both files under the 300-line limit the
deploy validator enforces.

Why this rule has a whole file
------------------------------
A button that targets ANOTHER panel is only safe if that panel is actually
declared AND rendered by the host. The historical bug here was routing a
click to ``gemini_studio`` while that panel was declared without
``center_overlay=True`` (or not declared at all) -- the reported dead "View
image"/"Open Gemini Studio" buttons.

That is now fixed at the root: ``gemini_studio`` is declared on
``slot="center"`` WITH ``center_overlay=True`` (see handlers/panel.py), which
is the SDK's own documented mechanism for a center panel to actually render
(confirmed against docs.imperal.io/en/concepts/panels and the SDK's
``ext.panel()`` docstring -- not the "the host will never grant this a render
path" assumption a previous version of this file encoded). So cross-panel
navigation from "gemini_quick" to "gemini_studio" is now a SANCTIONED target,
not a dead one -- this file asserts every button targets either its own
panel, the real declared companion panel, a registered tool, or the
sanctioned send-to-chat action.
"""
from __future__ import annotations

import pytest

from handlers.panel import gemini_quick_panel
from gemini_config import GENERATION_LOG_COLLECTION
from tests.fixtures import make_ctx


def _collect_buttons(node, acc):
    """Collect (label, on_click) for every Button in a serialized tree."""
    if isinstance(node, dict):
        if node.get("type") == "Button":
            props = node.get("props", {})
            acc.append((props.get("label", ""), props.get("on_click") or {}))
        for v in node.values():
            _collect_buttons(v, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_buttons(item, acc)
    return acc


# The ONE sanctioned non-"call" action. "View full resolution in chat" uses
# ui.Send (a real chat message) rather than ui.Call, deliberately: ui.Call
# bypasses chat and invokes the tool directly with no LLM turn to render the
# image it returns, whereas ui.Send starts a normal turn that DOES render it
# -- the only channel proven to actually deliver a full-size image.
_SEND_ACTION_LABELS = {"View full resolution in chat"}

# Panels this app actually declares (see imperal.json / handlers/panel.py) --
# a cross-panel button target is only safe when it names one of these.
_REAL_PANELS = {"gemini_quick", "gemini_studio", "secrets"}


def _is_registered_tool(function_name: str) -> bool:
    """Whether a button invokes a REAL chat tool of this app.

    A button may legitimately call a tool instead of re-rendering its panel --
    "Regenerate" runs a generation. That is not the dead-button class this
    rule guards: a tool call dispatches a registered function rather than a
    panel that may never be granted a render path. It is only safe while the
    name really exists, though, so this checks the live registry instead of
    waving through anything that merely looks like a tool.
    """
    import main  # noqa: F401 -- registers every handler module
    from app import chat
    return function_name in chat.functions


@pytest.mark.asyncio
async def test_every_button_targets_something_real():
    """THE rule this UI is built on: no button may target a dead end.

    A button may re-render its OWN panel, navigate to another panel this app
    ACTUALLY declares (``_REAL_PANELS``), invoke a registered tool, or use the
    sanctioned send-to-chat action. Anything else is a silent dead click.
    """
    ctx = make_ctx(with_key=True)
    created = await ctx.store.create(GENERATION_LOG_COLLECTION, {
        "user_id": ctx.user.imperal_id, "kind": "image", "prompt": "p",
        "model": "gemini-3-pro-image", "storage_path": "gemini/image/a.png",
        "mime_type": "image/png", "created_at": "2026-07-24T00:00:00Z",
    })
    _first_id = created.id

    for panel_fn, panel_id in (
        (lambda c: gemini_quick_panel(c, generation_id=_first_id), "gemini_quick"),
    ):
        result = await panel_fn(ctx)
        tree = result["ui"] if isinstance(result, dict) else result.to_dict()

        buttons = _collect_buttons(tree, [])
        assert buttons, f"{panel_id}: expected at least one button"

        for label, on_click in buttons:
            if label in _SEND_ACTION_LABELS:
                assert on_click.get("action") == "send", (
                    f"{panel_id}: button {label!r} is the sanctioned "
                    f"send-to-chat action, expected ui.Send, got {on_click!r}"
                )
                continue
            assert on_click.get("action") == "call", \
                f"{panel_id}: button {label!r} must use ui.Call, got {on_click!r}"
            target = on_click.get("function") or ""
            if target == f"__panel__{panel_id}":
                continue
            if target.startswith("__panel__") and target[len("__panel__"):] in _REAL_PANELS:
                continue
            if _is_registered_tool(target):
                continue
            raise AssertionError(
                f"{panel_id}: button {label!r} targets {target!r} -- a button "
                "must re-render its OWN panel, navigate to a REAL declared "
                "panel, invoke a registered tool, or be the sanctioned "
                f"send-to-chat action {sorted(_SEND_ACTION_LABELS)}. Anything "
                "else silently does nothing."
            )
