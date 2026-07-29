"""Gemini v1.0.0 · Image & video generation extension for Imperal Cloud."""
from __future__ import annotations

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

# ---------------------------------------------------------------------------
# Evict every module THIS EXTENSION OWNS from sys.modules before re-importing.
#
# Why this exists at all
# -----------------------
# The production host reuses a warm Python process across deploys (it does
# NOT always cold-start), so ``import handlers.panel`` after a redeploy can
# silently return the OLD module object still cached in ``sys.modules`` --
# code changes are on disk and ``deploy_app`` reports success, but the
# running process keeps executing the previous version until something else
# forces a restart. That is a plausible root cause for "I deployed a fix but
# nothing changed in the panel" reports.
#
# Why this list is now COMPUTED, not hand-typed
# ----------------------------------------------
# The previous version was a hardcoded tuple of module names, and it rotted:
# four rounds of adding new files (handlers/panel_detail.py,
# handlers/panel_history.py, handlers/panel_html.py, handlers/panel_forms.py,
# handlers/image_core.py, core/jpeg*.py, clients/gemini_client.py, models/,
# prompt_guide.py) were never added to it, so exactly the files under active
# development were the ones NOT guaranteed to reload. Walking the package
# directory for every ``*.py`` file and deriving its dotted module name
# removes that whole class of bug: any file this extension ships is evicted,
# with no separate list to keep in sync.
# ---------------------------------------------------------------------------
_OWNED_TOP_LEVEL = ("app", "gemini_config", "return_models", "prompt_guide")
_OWNED_PACKAGES = ("core", "clients", "handlers", "models")


def _owned_module_names() -> list[str]:
    names = list(_OWNED_TOP_LEVEL)
    for pkg in _OWNED_PACKAGES:
        pkg_dir = os.path.join(_dir, pkg)
        if not os.path.isdir(pkg_dir):
            continue
        names.append(pkg)
        for fname in os.listdir(pkg_dir):
            if fname.endswith(".py") and fname != "__init__.py":
                names.append(f"{pkg}.{fname[:-3]}")
    return names


for _m in [k for k in sys.modules if k in _owned_module_names()]:
    del sys.modules[_m]

import importlib
import logging

from app import ext, chat  # noqa: F401,E402

# UI discovery must come before optional tools. The host only has something to
# render if this import finishes; previously a decorator/API mismatch in an
# unrelated generator or upload module happened first and made the entire app
# look like chat-only. Panel code deliberately has no dependency on those
# optional tool registrations.
import handlers.panel  # noqa: F401,E402

log = logging.getLogger("gemini.entrypoint")


def _load_optional(module_name: str) -> None:
    """Register a non-UI feature without making panel discovery hostage to it.

    Deployment validation still catches missing catalog tools. At runtime this
    keeps the permanent Gemini panel available and records the actual failing
    module instead of silently losing every extension surface.
    """
    try:
        importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        log.exception("optional Gemini module failed to load: %s", module_name)


for _module in (
    "handlers.generate",
    "handlers.image_tools",
    "handlers.status",
    "handlers.prompt_help",
    "handlers.uploads",
    "handlers.diagnostics",
    "handlers.skeleton",
    "handlers.panel_viewer",
):
    _load_optional(_module)

del _module
