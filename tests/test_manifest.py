"""Manifest/runtime consistency tests.

These exist because of a live bug: the ``gemini_image`` viewer panel was
added in a handler-only commit (482eb0c) and ``imperal.json`` was never
rebuilt, so the committed manifest still listed only 3 panels. The platform
reads panels from the manifest, so the "View image" button called a
``__panel__gemini_image`` action the platform had no record of and nothing
happened -- with no error anywhere.

The panel handlers themselves were fine and fully unit-tested, which is
exactly why the handler tests did not catch it. A manifest that has drifted
from the code is its own class of bug, so it gets its own test.
"""
from __future__ import annotations

import json
from pathlib import Path

import main  # noqa: F401  -- imports every handler module, registering panels
from app import ext

MANIFEST = Path(__file__).resolve().parent.parent / "imperal.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_lists_every_registered_panel():
    """Every @ext.panel must appear in imperal.json's panels block.

    Guards the whole class of "handler shipped, manifest not rebuilt" bugs:
    a panel missing here is invisible to the platform, so any button whose
    on_click targets it silently does nothing.
    """
    manifest_panels = {p["panel_id"] for p in _manifest().get("panels", [])}
    runtime_panels = set(ext.panels)

    assert runtime_panels == manifest_panels, (
        "imperal.json is out of sync with the code -- run `imperal build .`. "
        f"Missing from manifest: {sorted(runtime_panels - manifest_panels)}; "
        f"stale in manifest: {sorted(manifest_panels - runtime_panels)}"
    )


def test_manifest_panel_metadata_matches_decorators():
    """slot/center_overlay must match too, not just the panel_id set.

    A center panel is only opened on demand when the manifest carries
    ``center_overlay: true`` (I-PANEL-RENDERING-CONTRACT), and this flag has
    silently regressed on rebuild before, so it is pinned explicitly.
    """
    by_id = {p["panel_id"]: p for p in _manifest().get("panels", [])}

    for panel_id, meta in ext.panels.items():
        entry = by_id.get(panel_id)
        if entry is None:
            # Presence is asserted by test_manifest_lists_every_registered_panel;
            # skip here so that failure reports once, with its own clear message.
            continue
        assert entry["slot"] == meta["slot"], f"{panel_id}: slot drift"
        assert entry.get("center_overlay", False) == meta.get("center_overlay", False), (
            f"{panel_id}: center_overlay drift -- a center panel without this "
            "flag is never opened by the panel host"
        )


def test_every_panel_call_action_targets_a_real_panel():
    """Any ui.Call("__panel__<id>") in the UI must resolve to a live panel.

    Catches the button side of the same bug: a typo'd or removed panel id in
    an on_click is otherwise a silent no-op click.
    """
    import io
    import re
    import tokenize

    # Match only real ui.Call(...) expressions. Comments are stripped first:
    # panel.py documents the precedent from another extension in prose that
    # itself spells out a ui.Call("__panel__spotify_detail", ...) example, and
    # a comment is not a live call site.
    call_re = re.compile(r'ui\.Call\(\s*[\'"]__panel__([a-zA-Z0-9_]+)[\'"]')

    def _strip_comments(source: str) -> str:
        out: list[str] = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type != tokenize.COMMENT:
                    out.append(tok.string)
        except tokenize.TokenError:  # pragma: no cover - malformed source
            return source
        return "\n".join(out)

    panel_dir = Path(__file__).resolve().parent.parent / "handlers"
    referenced: set[str] = set()
    for path in panel_dir.glob("*.py"):
        referenced |= set(call_re.findall(_strip_comments(path.read_text())))

    unknown = referenced - set(ext.panels)
    assert not unknown, (
        f"UI calls reference panels that are not registered: {sorted(unknown)}"
    )


def test_manifest_app_id_matches_extension():
    """app_id drift between code and manifest broke deploys once already."""
    assert _manifest()["app_id"] == ext.app_id
