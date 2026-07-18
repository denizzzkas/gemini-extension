"""Official Google prompting guidance for Gemini image/video generation.

Every string here is transcribed (condensed, not paraphrased-into-fiction)
from Google's own developer documentation, fetched and verified 2026-07-18:

  - https://ai.google.dev/gemini-api/docs/image-generation
    -> "Prompting guide and strategies" + "Best practices" sections
  - https://ai.google.dev/gemini-api/docs/veo
    -> "Veo prompt guide" -> "Prompt writing basics" + "Prompting for audio"

These constants are injected into the ``description=`` of the
``generate_image`` / ``generate_video`` chat functions and their
``prompt`` param, so the model that calls these tools (Webbee) has the
Google-confirmed structure in hand *before* it writes the prompt argument
-- turning a vague user ask ("нарисуй кота") into a properly structured
prompt without a second heuristic/LLM pass inside this extension.
"""
from __future__ import annotations

# ─── Images: generation templates (source: image-generation.md §"Prompting guide") ─── #

IMAGE_GENERATION_TEMPLATES: dict[str, str] = {
    "photorealistic_scene": (
        "A photorealistic [type of shot] of a [subject description] in a "
        "[setting description]. [Description of the light]. Shot from a "
        "[camera angle] with a [lens type]."
    ),
    "stylized_illustration_sticker": (
        "A [style] of a [subject, with details about accessories or actions] "
        "doing [activity]. The design features [visual qualities, e.g. bold "
        "outlines, cel-shading] and [color/background preference]."
    ),
    "accurate_text_in_image": (
        "Create a [image type] for [brand/concept] with the text \"[text to "
        "render]\" in a [font style]. The design should be [style "
        "description], with a [color scheme]."
    ),
    "product_mockup": (
        "A high-resolution, studio-lit product photograph of a [product "
        "description] on a [background surface/description]. The lighting is "
        "a [lighting setup] to [lighting purpose]. The camera angle is a "
        "[angle type] to showcase [specific feature]. Ultra-realistic, with "
        "sharp focus on [key detail]. [Aspect ratio]."
    ),
    "minimalist_negative_space": (
        "A minimalist composition featuring a single [subject] positioned in "
        "the [bottom-right/top-left/etc.] of the frame. The background is a "
        "vast, empty [color] canvas, creating significant negative space. "
        "Soft, subtle lighting. [Aspect ratio]."
    ),
    "sequential_art_comic": (
        "Make a [N] panel comic in a [style]. Put the character in a [type "
        "of scene]."
    ),
}

# ─── Images: editing templates (source: image-generation.md §"Prompts for editing images") ─── #

IMAGE_EDITING_TEMPLATES: dict[str, str] = {
    "add_remove_element": (
        "Using the provided image of [subject], please [add/remove/modify] "
        "[element] to/from the scene. Ensure the change is [description of "
        "how the change should integrate]."
    ),
    "inpainting_semantic_mask": (
        "Using the provided image, change only the [specific element] to "
        "[new element/description]. Keep everything else in the image "
        "exactly the same, preserving the original style, lighting, and "
        "composition."
    ),
    "style_transfer": (
        "Transform the provided photograph of [subject] into the artistic "
        "style of [artist/art style]. Preserve the original composition but "
        "render it with [description of stylistic elements]."
    ),
    "combine_multiple_images": (
        "Create a new image by combining the elements from the provided "
        "images. Take the [element from image 1] and place it with/on the "
        "[element from image 2]. The final image should be a [description "
        "of the final scene]."
    ),
    "high_fidelity_detail_preservation": (
        "Using the provided images, place [element from image 2] onto "
        "[element from image 1]. Ensure that the features of [element from "
        "image 1] remain completely unchanged. The added element should "
        "[description of how it should integrate]."
    ),
    "bring_sketch_to_life": (
        "Turn this rough [medium] sketch of a [subject] into a [style "
        "description] photo. Keep the [specific features] from the sketch "
        "but add [new details/materials]."
    ),
    "character_consistency_360": (
        "A studio portrait of [person] against [background], [looking "
        "forward/in profile looking right/etc.]"
    ),
}

# Verbatim from image-generation.md §"Best practices"
IMAGE_BEST_PRACTICES: list[str] = [
    "Be hyper-specific: describe details (\"ornate elven plate armor, etched "
    "with silver leaf patterns\") instead of vague terms (\"fantasy armor\").",
    "Provide context and intent: state the image's purpose "
    "(\"a logo for a high-end, minimalist skincare brand\") rather than a bare "
    "request (\"create a logo\").",
    "Iterate and refine conversationally: don't expect perfection on the "
    "first try -- follow up with small changes (\"make the lighting warmer\").",
    "Use step-by-step instructions for complex scenes with many elements "
    "(\"first the background, then the foreground object, finally the "
    "highlight\").",
    "Use semantic negative prompts: describe the intended scene positively "
    "(\"an empty, deserted street\") instead of negating (\"no cars\").",
    "Control the camera with photographic/cinematic language: wide-angle "
    "shot, macro shot, low-angle perspective.",
]

# ─── Video (Veo): prompt writing basics (source: veo.md §"Veo prompt guide") ─── #

VIDEO_PROMPT_ELEMENTS: dict[str, str] = {
    "subject": "The object, person, animal, or scenery in the video (e.g. cityscape, puppies).",
    "action": "What the subject is doing (e.g. walking, running, turning their head).",
    "style": "Creative direction via film-style keywords (sci-fi, film noir, cartoon).",
    "camera_positioning_motion": "Camera location/movement (aerial view, dolly shot, worms-eye) -- optional.",
    "composition": "How the shot is framed (wide shot, close-up, two-shot) -- optional.",
    "focus_lens": "Lens effects (shallow focus, macro lens, wide-angle lens) -- optional.",
    "ambiance": "Color and light contribution to mood (blue tones, warm tones, night) -- optional.",
}

VIDEO_AUDIO_TIPS: list[str] = [
    "Dialogue: use quotes for specific speech, e.g. \"'This must be the key,' he murmured.\"",
    "Sound effects (SFX): describe explicitly, e.g. \"tires screeching loudly, engine roaring.\"",
    "Ambient noise: describe the environment's soundscape, e.g. \"a faint, eerie hum resonates in the background.\"",
]

VIDEO_EXTRA_TIPS: list[str] = [
    "Use descriptive language: adjectives and adverbs paint a clearer picture.",
    "Enhance facial detail by naming 'portrait' as a focus when a face matters.",
]


def image_prompt_guidance_text() -> str:
    """Compact guidance block for tool descriptions (image generation/editing)."""
    templates = "; ".join(f"{k}: {v}" for k, v in IMAGE_GENERATION_TEMPLATES.items())
    edits = "; ".join(f"{k}: {v}" for k, v in IMAGE_EDITING_TEMPLATES.items())
    practices = " | ".join(IMAGE_BEST_PRACTICES)
    return (
        "GOOGLE-CONFIRMED PROMPT STRUCTURE (image-generation docs): if the "
        "user's request is vague, expand it yourself into a fully-specified "
        "prompt before calling this tool -- do not pass the user's raw short "
        f"phrase through untouched. Generation templates: {templates}. "
        f"Editing templates (when an image is provided): {edits}. "
        f"Best practices: {practices}."
    )


def video_prompt_guidance_text() -> str:
    """Compact guidance block for tool descriptions (video generation)."""
    elements = "; ".join(f"{k}: {v}" for k, v in VIDEO_PROMPT_ELEMENTS.items())
    audio = " | ".join(VIDEO_AUDIO_TIPS)
    extra = " | ".join(VIDEO_EXTRA_TIPS)
    return (
        "GOOGLE-CONFIRMED PROMPT STRUCTURE (Veo prompt guide): if the user's "
        "request is vague, expand it yourself into a fully-specified prompt "
        "before calling this tool, covering these elements where relevant: "
        f"{elements}. Audio cues: {audio}. Extra tips: {extra}."
    )
