"""The per-user Gemini API key field, rendered above generation controls.

Why this exists
---------------
``gemini_api_key`` moved from an app-level to a per-user secret
(``scope="user"``), so there is no longer one app-wide value an admin sets
once -- every user must bring their own key. The platform's built-in Secrets
tab still works (``write_mode="both"`` in app.py keeps it enabled), but the
user asked explicitly for a field INSIDE the left panel itself, above the
generation form, so setting a key does not require leaving the extension's
own surface. ``write_mode="both"`` is what makes this legal: ``write_mode=
"user"`` would make ``ctx.secrets.set()`` raise ``SecretWriteForbidden`` for
any call originating from extension code, which is exactly what the
``save_gemini_api_key`` chat function below does on submit.
"""
from __future__ import annotations

from imperal_sdk import ui


def api_key_field(configured: bool) -> ui.UINode:
    """A one-field form that saves the user's own Gemini API key.

    Deliberately its own small Card, not folded into ``generation_tabs``:
    the key is prerequisite state, not a generation parameter, and keeping
    it visually separate (and always ABOVE the generation form, per the
    user's explicit request) makes that distinction obvious at a glance.
    """
    return ui.Card(
        title="Gemini API key",
        subtitle=(
            "Connected -- paste a new key below to replace it"
            if configured else
            "Paste your key from aistudio.google.com/apikey to start generating"
        ),
        content=ui.Form(
            children=[
                ui.Password(
                    placeholder="AIza...",
                    param_name="api_key",
                ),
            ],
            action="save_gemini_api_key",
            submit_label="Update key" if configured else "Save key",
        ),
    )
