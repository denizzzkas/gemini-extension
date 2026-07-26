"""Generation forms for the Gemini panels.

Split out of ``handlers/panel.py`` to stay under the 300-line file limit the
deploy validator enforces. Both panels render the SAME form components, so
they live here instead of being duplicated.
"""
from __future__ import annotations

from imperal_sdk import ui

from gemini_config import (
    IMAGE_MODEL_CHOICES, MODEL_IMAGE, IMAGE_SIZE_CHOICES, DEFAULT_IMAGE_SIZE,
)


def _image_form() -> ui.UINode:
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
