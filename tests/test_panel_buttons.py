"""The no-dead-buttons rule: every button must target its OWN panel.

Split out of ``test_panel.py`` to keep both files under the 300-line limit the
deploy validator enforces.

Why this rule has a whole file
------------------------------
Per I-PANEL-RENDERING-CONTRACT a ``slot="center"`` panel is rendered as a
centre overlay only when the host grants it a render path. A button in the LEFT
panel that re-renders a DIFFERENT panel is therefore a button that can do
nothing at all, with no error and no clue why -- the class of bug that produced
a "Hide" button which silently did nothing while "View" worked.

This extension declares no center-slot panel at all any more ("gemini_studio"
was removed after the user confirmed it never opened in real testing), so
there is nothing left to escalate to -- every button must target the ONE
panel it is rendered in ("gemini_quick"), invoke a registered tool, or send a
chat message (the "View full resolution in chat" button, which hands the
original off to the one proven-working delivery channel instead of trying
another in-panel trick -- see handlers/panel_html.py).
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
async def test_every_button_targets_the_panel_it_is_rendered_in():
    """THE rule this UI is built on: no button may depend on ANOTHER panel.

    Per I-PANEL-RENDERING-CONTRACT a slot="center" panel is only rendered as a
    center-overlay when the host grants it a render path (historically a
    hardcoded allowlist: compose, email_viewer, editor, workshop). Routing a
    click to a different panel therefore silently did nothing -- the reported
    dead "View image"/"Open Gemini Studio" buttons. This extension no longer
    declares a center panel at all, so this now simply guards that every
    button in the one real panel targets ITSELF (or a registered tool, or the
    sanctioned Send-to-chat action).
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
            target = on_click.get("function")
            expected = f"__panel__{panel_id}"
            if target == expected or _is_registered_tool(target or ""):
                continue
            raise AssertionError(
                f"{panel_id}: button {label!r} targets {target!r} -- a button "
                "must re-render its OWN panel, invoke a registered tool, or be "
                f"the sanctioned send-to-chat action {sorted(_SEND_ACTION_LABELS)}. "
                "Anything else silently does nothing when the host does not "
                "grant another panel a render path."
            )
