"""Generation forms for the Gemini panels.

Split out of ``handlers/panel.py`` to stay under the 300-line file limit the
deploy validator enforces.

Image and video live in separate TABS
-------------------------------------
Both forms used to be rendered one under the other in the same column, so the
panel opened as a wall of inputs and pushed history off-screen -- even though
only one of the two is ever in use at a time. :func:`generation_tabs` puts them
behind ``ui.Tabs`` and leaves the column to history.

Reference images: chosen by SIGHT, not by prompt text
-----------------------------------------------------
The picker used to be a dropdown of prompt strings, which asked the user to
remember which wall of text produced which image -- unanswerable, and the whole
reason references went unused. Selection now happens where the image is
actually visible: each history card carries "Use as reference", and the form
shows the chosen reference as a THUMBNAIL so there is no doubt what is attached.

The dropdown survives underneath as the control that actually carries the
selection into the form submit (a ``ui.MultiSelect`` pre-selected via
``values=``), because a form submits the values of its own inputs. It is now a
confirmation of a choice already made by eye, not the place the choice is made.

Only SELECTED references are shown as images, deliberately: every thumbnail is
base64 inside the response, and a gallery of a dozen would push the panel past
the payload ceiling that made images fail to render in the first place.
"""
from __future__ import annotations

from imperal_sdk import ui

from handlers.panel_viewer import CLOSED_SENTINEL

from gemini_config import (
    IMAGE_MODEL_CHOICES, MODEL_IMAGE, IMAGE_SIZE_CHOICES, DEFAULT_IMAGE_SIZE,
)


def _selected_reference_block(selected: list[dict]) -> list[ui.UINode]:
    """Thumbnails of the references currently attached, with a way to clear them.

    ``selected`` items are ``{"id", "src", "label"}``. ``src`` may be empty when
    a record has no cached preview yet -- the entry is still shown, by label, so
    an attached reference is never invisible just because its thumbnail is
    missing.
    """
    if not selected:
        return []

    thumbs: list[ui.UINode] = []
    for ref in selected:
        src = ref.get("src") or ""
        if src:
            thumbs.append(ui.Image(
                src=src, alt=ref.get("label", "reference"), width="72px",
            ))
        else:
            thumbs.append(ui.Text(f"• {ref.get('label') or 'image'}", variant="caption"))

    return [
        ui.Text(
            f"Attached reference ({len(selected)}) — the generation will reuse "
            "this image:",
            variant="caption",
        ),
        ui.Stack(children=thumbs, direction="h", gap=2, wrap=True),
        ui.Button(
            label="Clear references",
            variant="ghost",
            icon="X",
            # CLOSED_SENTINEL, not "": the host MERGES a re-fetch's params into
            # the ones already accumulated for this panel, so an empty value
            # cannot remove an existing one -- clearing has to overwrite with a
            # non-empty value that the reader then treats as "none". This is the
            # same reason "Hide" once did nothing while "View" worked.
            on_click=ui.Call("__panel__gemini_quick", refs=CLOSED_SENTINEL),
        ),
    ]


def _reference_controls(selected: list[dict] | None = None) -> list[ui.UINode]:
    """Upload dropzone plus thumbnails of whatever is currently attached.

    ``selected`` is the shape described in :func:`_selected_reference_block`.

    There is deliberately NO dropdown of stored images here. It used to list
    prompt TEXT, which asked the user to recognise a picture from the wall of
    words that produced it. References are now chosen by sight, with the "Use as
    reference" button next to the visible image in the history below.
    """
    selected = selected or []

    # The dropzone's caption is a separate ui.Text rather than FileUpload's own
    # title/hint/show_previews: those keywords exist in the local SDK but the
    # PRODUCTION validator rejects them outright, so using them makes the
    # extension undeployable.
    nodes: list[ui.UINode] = [
        ui.Text("Reference image (optional)", variant="caption"),
        ui.Text(
            "Drop a PNG or JPEG here, or press \"Use as reference\" on any "
            "image in the history below.",
            variant="caption",
        ),
        ui.FileUpload(
            accept="image/png,image/jpeg",
            max_size_mb=12,
            multiple=True,
            max_files=5,
            param_name="files",
            on_upload=ui.Call("upload_reference_image"),
        ),
    ]
    nodes += _selected_reference_block(selected)
    return nodes


def _image_form(selected_references: list[dict] | None = None) -> ui.UINode:
    """The image generation form.

    ``selected_references`` defaults to empty so callers and tests that do not
    care about references keep working; the panel supplies it.
    """
    selected_ids = [
        r["id"] for r in (selected_references or []) if r.get("id")
    ]
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
                *_reference_controls(selected_references or []),
            ],
            action="generate_image",
            submit_label="Generate image",
            # The attached references ride along as a HIDDEN default rather than
            # as a visible picker.
            #
            # The picker used to be a MultiSelect listing PROMPT TEXT, which
            # asked the user to recognise a picture from the wall of words that
            # produced it -- unusable by design, and the reason it is gone.
            # Something still has to submit the ids though, or "Use as
            # reference" would be purely decorative: a form submits the values
            # of its own inputs, and a thumbnail is not an input. ui.Form
            # defaults are exactly that hidden carrier.
            #
            # Sent only when non-empty, and as a LIST, not a joined string:
            # the tool's parameter is list[str] and Pydantic rejects "a,b"
            # outright ("Input should be a valid list") -- verified, not
            # assumed. An empty value is omitted entirely so the field simply
            # defaults rather than arriving present-but-blank.
            defaults=(
                {"reference_generation_ids": selected_ids}
                if selected_ids else None
            ),
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


def generation_tabs(selected_references: list[dict] | None = None) -> ui.UINode:
    """Image and video generation as two tabs instead of two stacked forms."""
    return ui.Tabs(tabs=[
        {
            "label": "Image",
            "content": _image_form(selected_references),
        },
        {"label": "Video", "content": _video_form()},
    ])
