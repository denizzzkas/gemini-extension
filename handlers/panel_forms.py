"""Generation forms for the Gemini panels.

Split out of ``handlers/panel.py`` to stay under the 300-line file limit the
deploy validator enforces.

Reference images, and why the form has two controls for them
------------------------------------------------------------
A reference can only be an image the extension already holds, because the
Gemini call needs the actual BYTES and the panel form submits values, not
files. So attaching one is genuinely two steps, and the form says so:

  1. the dropzone UPLOADS a file (it becomes an image you own, stored exactly
     like a generation -- see handlers/uploads.py),
  2. the picker SELECTS which of your stored images to send with the prompt.

An upload therefore appears in the picker on the next render. Collapsing this
into a single control would mean either uploading on submit (losing the file
if generation fails) or pretending the picker can accept raw files.
"""
from __future__ import annotations

from imperal_sdk import ui

from gemini_config import (
    IMAGE_MODEL_CHOICES, MODEL_IMAGE, IMAGE_SIZE_CHOICES, DEFAULT_IMAGE_SIZE,
)

# How many recent images to offer as reference choices. The picker is a
# dropdown, not a gallery, so a long list stops being usable well before it
# stops being renderable.
REFERENCE_CHOICE_LIMIT = 24


def _reference_controls(choices: list[dict]) -> list[ui.UINode]:
    """The upload dropzone + the picker of already-stored images.

    ``choices`` is a list of ``{"value": generation_id, "label": ...}`` for the
    user's recent images. When it is empty the picker is omitted entirely
    rather than rendered blank: an empty dropdown looks broken and invites the
    conclusion that references are unavailable, when in fact none exist yet.
    """
    nodes: list[ui.UINode] = [
        ui.FileUpload(
            accept="image/png,image/jpeg",
            max_size_mb=12,
            multiple=True,
            max_files=5,
            param_name="files",
            title="Reference image (optional)",
            hint="Drop a PNG or JPEG to reuse a character, style or scene.",
            show_previews=True,
            on_upload=ui.Call("upload_reference_image"),
        ),
    ]
    if choices:
        nodes.append(ui.MultiSelect(
            options=choices,
            placeholder="Use stored image(s) as reference…",
            param_name="reference_generation_ids",
        ))
    return nodes


def _image_form(reference_choices: list[dict] | None = None) -> ui.UINode:
    """The generation form.

    ``reference_choices`` defaults to none so existing callers (and tests)
    keep working; the panel passes the user's recent images so they can be
    picked as references.
    """
    return ui.Card(
        title="Generate image",
        subtitle="Nano Banana — pick a model",
        content=ui.Form(
            children=[
                ui.TextArea(
                    placeholder="Describe the image you want...",
                    param_name="prompt", rows=3,
                ),
                ui.Select(
                    options=[
                        {"value": mid, "label": info["label"]}
                        for mid, info in IMAGE_MODEL_CHOICES.items()
                    ],
                    value=MODEL_IMAGE,
                    param_name="model",
                ),
                ui.Select(
                    options=[
                        {"value": size, "label": label}
                        for size, label in IMAGE_SIZE_CHOICES.items()
                    ],
                    value=DEFAULT_IMAGE_SIZE,
                    param_name="image_size",
                ),
                *_reference_controls(reference_choices or []),
            ],
            action="generate_image",
            submit_label="Generate image",
        ),
    )


def _video_form() -> ui.UINode:
    return ui.Card(
        title="Generate video",
        subtitle="Gemini Omni Flash",
        content=ui.Form(
            children=[
                ui.TextArea(
                    placeholder="Describe the video you want...",
                    param_name="prompt", rows=3,
                ),
            ],
            action="generate_video",
            submit_label="Generate video",
        ),
    )
