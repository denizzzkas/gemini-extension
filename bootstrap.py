"""Register Gemini handlers for every supported extension entrypoint."""
from __future__ import annotations

import importlib
import logging

# UI discovery must happen before optional tools. If an unrelated generator or
# upload module fails at import time, the permanent Gemini sidebar still exists.
import handlers.panel  # noqa: F401

log = logging.getLogger("gemini.bootstrap")


def _load_optional(module_name: str) -> None:
    """Register one non-UI feature without blocking panel discovery."""
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
