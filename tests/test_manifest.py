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
from app import chat, ext

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


def test_manifest_lists_every_registered_chat_function():
    """Every @chat.function must appear in imperal.json's tools block.

    The panel version of this test existed while the tool version did not,
    and the gap bit immediately: four per-model image tools were registered
    in code and absent from the manifest, so the platform could not see or
    price them. The platform reads tools from the manifest, so an unlisted
    tool simply does not exist for callers.
    """
    manifest_tools = {t["name"] for t in _manifest().get("tools", [])}
    runtime_tools = set(chat.functions)

    missing = sorted(runtime_tools - manifest_tools)
    assert not missing, (
        "imperal.json is out of sync with the code -- rebuild the manifest. "
        f"Registered in code but missing from the manifest: {missing}"
    )


def test_manifest_keeps_its_marketplace_metadata():
    """category/license/tags must survive a manifest rebuild.

    These are disk-only fields: the generator does not own them, and
    regenerating the manifest drops them unless they are carried over. That
    happened once here -- a rebuild silently removed the app's Marketplace
    category, licence and every search tag, which is invisible in tests that
    only look at tools and panels.
    """
    m = _manifest()
    for field in ("category", "license", "tags"):
        assert m.get(field), (
            f"manifest lost {field!r} -- a rebuild dropped a disk-preserved "
            "field, which silently degrades the Marketplace listing"
        )


def test_manifest_panel_metadata_matches_decorators():
    """slot/center_overlay must match too, not just the panel_id set.

    A center panel is only opened on demand when the manifest carries
    ``center_overlay: true``. This flag has silently regressed on rebuild
    before, so the deployed metadata must match the decorators exactly.
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


def test_gemini_uses_the_supported_sidebar_and_center_overlay_layout():
    """Gemini has a permanent left sidebar and an on-demand Studio overlay."""
    quick = ext.panels["gemini_quick"]
    studio = ext.panels["gemini_studio"]

    assert quick["slot"] == "left"
    assert studio["slot"] == "center"
    assert studio.get("center_overlay", False)


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


def test_every_ui_action_target_exists():
    """Every Form action / Button call in EVERY panel must resolve to something real.

    This is the test that would have caught the whole dead-button saga at
    commit time. Panels are rendered for real, then every action target found
    in the serialized tree is checked against the things the platform can
    actually invoke: registered panel tools (__panel__*), chat functions, and
    skeleton refreshers.
    """
    import asyncio
    import json
    import re

    from app import chat
    from tests.fixtures import make_ctx

    known = set(ext.tools)
    known |= set(getattr(chat, "functions", {}) or {})

    async def _render_all() -> dict[str, str]:
        """Render every panel WITH data present.

        Rendering against an empty store is useless for this test: the history
        list collapses to an Empty node, so the per-entry buttons never appear
        and a dead target cannot be observed. Seed one generation (and open it)
        so every branch of the UI is actually emitted.
        """
        from gemini_config import GENERATION_LOG_COLLECTION

        ctx = make_ctx(with_key=True)
        await ctx.storage.upload(
            "gemini/image/guard.png", b"\x89PNG\r\n\x1a\n" + b"x" * 40,
            content_type="image/png",
        )
        doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id, "kind": "image",
            "prompt": "manifest guard fixture", "model": "gemini-3-pro-image",
            "storage_path": "gemini/image/guard.png", "mime_type": "image/png",
            "created_at": "2026-07-24T00:00:00Z",
        })

        # A record whose bytes are GONE, so the failure branch ("Retry") also
        # renders. Without this the retry button is never emitted and a dead
        # target hiding in that branch stays invisible to this test -- which is
        # exactly how one slipped past an earlier version of this guard.
        broken = await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id, "kind": "image",
            "prompt": "bytes are gone", "model": "gemini-3-pro-image",
            "storage_path": "gemini/image/vanished.png", "mime_type": "image/png",
            "created_at": "2026-07-23T00:00:00Z",
        })

        out = {}
        for panel_id in ext.panels:
            tool = ext.tools.get(f"__panel__{panel_id}")
            if tool is None:
                continue
            # Render every branch: closed (View image), opened (inline image +
            # Hide) and opened-but-bytes-missing (Retry).
            blobs = [
                await tool.func(ctx),
                await tool.func(ctx, generation_id=doc.id),
                await tool.func(ctx, generation_id=broken.id),
            ]
            out[panel_id] = "".join(json.dumps(b) for b in blobs)
        return out

    trees = asyncio.run(_render_all())
    assert trees, "no panels rendered -- registration is broken"

    problems: list[str] = []
    for panel_id, blob in trees.items():
        # ui.Call(...) targets
        for target in set(re.findall(r'"function":\s*"([A-Za-z0-9_]+)"', blob)):
            if target not in known:
                problems.append(f"{panel_id}: call -> {target!r} does not exist")
        # ui.Form(action=...) targets
        for target in set(re.findall(r'"action":\s*"([A-Za-z0-9_]+)"', blob)):
            if target in ("call", "navigate", "send", "open"):
                continue  # UIAction kinds, not invocation targets
            if target not in known:
                problems.append(f"{panel_id}: form action -> {target!r} does not exist")

    assert not problems, "dead UI targets:\n  " + "\n  ".join(problems)


def test_entrypoint_registers_callable_panel_endpoints():
    """The import path used by the host must expose each declared panel.

    A boot-time exception before ``handlers.panel`` imports leaves the host
    with no callable sidebar even if direct handler unit tests pass.
    """
    for panel_id in ext.panels:
        endpoint = ext.tools.get(f"__panel__{panel_id}")
        assert endpoint is not None, f"{panel_id}: no callable panel endpoint"
        assert callable(endpoint.func), f"{panel_id}: endpoint is not callable"


def test_panels_register_before_optional_tool_modules():
    """One optional tool import must not turn the whole app into chat-only."""
    import subprocess
    import sys

    probe = r'''
import importlib
real_import = importlib.import_module

def fail_image_tools(name, *args, **kwargs):
    if name == "handlers.image_tools":
        raise RuntimeError("simulated optional tool import failure")
    return real_import(name, *args, **kwargs)

importlib.import_module = fail_image_tools
import main
from app import ext
assert {"gemini_quick", "gemini_studio"} <= set(ext.panels)
assert {"__panel__gemini_quick", "__panel__gemini_studio"} <= set(ext.tools)
'''
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=MANIFEST.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_two_panels_claim_the_same_slot():
    """Two panels on one slot make the centre unusable.

    Live bug this guards: ``gemini_image`` and ``gemini_studio`` were BOTH
    registered on slot="center". The host opens one panel per slot and picked
    gemini_image -- with no params -- so the centre permanently showed its
    "Nothing to show / this viewer opened without a generation id" dead end,
    while the real Studio (forms + history) could not be reached at all.

    Only ONE panel may own a slot. Duplicates are a wiring bug, not a layout
    preference, and nothing in the handler tests can see it.
    """
    by_slot: dict[str, list[str]] = {}
    for panel_id, cfg in ext.panels.items():
        by_slot.setdefault(cfg.get("slot", ""), []).append(panel_id)

    clashes = {slot: sorted(ids) for slot, ids in by_slot.items() if len(ids) > 1}
    assert not clashes, (
        "more than one panel registered on the same slot -- the host shows only "
        f"one, so the others are unreachable: {clashes}"
    )
