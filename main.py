"""Gemini v1.0.0 · Image & video generation extension for Imperal Cloud."""
from __future__ import annotations

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

for _m in [k for k in sys.modules if k in (
    "app", "gemini_config", "return_models",
    "core", "core.png", "core.jpeg", "core.preview",
    "clients", "clients.gemini_client",
    "handlers", "handlers.media", "handlers.generate", "handlers.status",
    "handlers.image_core", "handlers.image_tools", "handlers.image_loader",
    "handlers.diagnostics", "handlers.probe",
    "handlers.skeleton", "handlers.panel", "handlers.panel_viewer",
    "handlers.panel_forms",
)]:
    del sys.modules[_m]

from app import ext, chat  # noqa: F401
import handlers.generate  # noqa: F401
import handlers.image_tools  # noqa: F401  -- one tool per model, own price each
import handlers.status  # noqa: F401
import handlers.diagnostics  # noqa: F401
import handlers.probe  # noqa: F401
import handlers.skeleton  # noqa: F401
import handlers.panel  # noqa: F401
import handlers.panel_viewer  # noqa: F401
