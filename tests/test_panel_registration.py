"""Runtime registration and action-target guards for Gemini panels."""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import main  # noqa: F401 -- imports every handler module
from app import chat, ext
from tests.fixtures import make_ctx

ROOT = Path(__file__).resolve().parent.parent


def test_every_ui_action_target_exists():
    """Every rendered Form/Button target resolves to a registered endpoint."""
    known = set(ext.tools)
    known |= set(getattr(chat, "functions", {}) or {})

    async def render_all() -> dict[str, str]:
        """Render populated, selected, and missing-media branches for each panel."""
        from gemini_config import GENERATION_LOG_COLLECTION

        ctx = make_ctx(with_key=True)
        await ctx.storage.upload(
            "gemini/image/guard.png", b"\x89PNG\r\n\x1a\n" + b"x" * 40,
            content_type="image/png",
        )
        doc = await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": "image",
            "prompt": "manifest guard fixture",
            "model": "gemini-3-pro-image",
            "storage_path": "gemini/image/guard.png",
            "mime_type": "image/png",
            "created_at": "2026-07-24T00:00:00Z",
        })
        broken = await ctx.store.create(GENERATION_LOG_COLLECTION, {
            "user_id": ctx.user.imperal_id,
            "kind": "image",
            "prompt": "bytes are gone",
            "model": "gemini-3-pro-image",
            "storage_path": "gemini/image/vanished.png",
            "mime_type": "image/png",
            "created_at": "2026-07-23T00:00:00Z",
        })

        rendered = {}
        for panel_id in ext.panels:
            endpoint = ext.tools.get(f"__panel__{panel_id}")
            if endpoint is None:
                continue
            responses = [
                await endpoint.func(ctx),
                await endpoint.func(ctx, generation_id=doc.id),
                await endpoint.func(ctx, generation_id=broken.id),
            ]
            rendered[panel_id] = "".join(json.dumps(item) for item in responses)
        return rendered

    trees = asyncio.run(render_all())
    assert trees, "no panels rendered -- registration is broken"

    problems: list[str] = []
    for panel_id, blob in trees.items():
        for target in set(re.findall(r'"function":\s*"([A-Za-z0-9_]+)"', blob)):
            if target not in known:
                problems.append(f"{panel_id}: call -> {target!r} does not exist")
        for target in set(re.findall(r'"action":\s*"([A-Za-z0-9_]+)"', blob)):
            if target not in {"call", "navigate", "send", "open"} and target not in known:
                problems.append(f"{panel_id}: form action -> {target!r} does not exist")

    assert not problems, "dead UI targets:\n  " + "\n  ".join(problems)


def test_entrypoint_registers_callable_panel_endpoints():
    """The host entrypoint exposes a callable endpoint for every panel."""
    for panel_id in ext.panels:
        endpoint = ext.tools.get(f"__panel__{panel_id}")
        assert endpoint is not None, f"{panel_id}: no callable panel endpoint"
        assert callable(endpoint.func), f"{panel_id}: endpoint is not callable"


def test_app_and_main_entrypoints_register_the_same_panels():
    """A host importing app.py directly must not degrade to chat-only."""
    expected = {"gemini_quick", "gemini_studio", "secrets"}
    for entrypoint in ("app", "main"):
        probe = (
            f"import {entrypoint}; from app import ext; "
            f"assert set(ext.panels) == {expected!r}; "
            "assert all(f'__panel__{p}' in ext.tools for p in ext.panels)"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{entrypoint}: {result.stderr}"


def test_panels_register_before_optional_tool_modules():
    """An optional tool failure must not turn the whole extension into chat-only."""
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
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
