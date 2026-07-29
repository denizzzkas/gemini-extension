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
_OWNED_TOP_LEVEL = (
    "app", "bootstrap", "gemini_config", "return_models", "prompt_guide",
)
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

# app.py imports bootstrap after constructing ``ext``. Keeping main as a thin
# compatibility entrypoint means the CLI and a host that imports app directly
# register precisely the same panels and tools.
from app import ext, chat  # noqa: F401,E402
