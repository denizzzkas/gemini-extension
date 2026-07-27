"""Gemini API constants and Imperal store collection names."""

# Gemini API (Interactions API — unified generateContent-style surface)
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Image model ids (Gemini API "Nano Banana" family, as of 2026-07 —
# ai.google.dev/gemini-api/docs/image-generation). All four share the exact
# same /interactions request/response shape, only the ``model`` field
# changes, so offering a choice is safe/low-risk (unlike video, see below).
MODEL_IMAGE = "gemini-3-pro-image"              # Nano Banana Pro — premium, 4K, up to 5 char-consistency refs
MODEL_IMAGE_FLASH = "gemini-3.1-flash-image"    # Nano Banana 2 — versatile workhorse, up to 4 char-consistency refs
MODEL_IMAGE_FLASH_LITE = "gemini-3.1-flash-lite-image"  # Nano Banana 2 Lite — fastest/cheapest, no multi-ref support
MODEL_IMAGE_LEGACY = "gemini-2.5-flash-image"   # Nano Banana (legacy) — Google recommends Flash Lite instead

# Default + selectable catalogue for the model= param on generate_image.
# Keys are the exact API model ids; label/description are surfaced to
# Webbee (tool description) and the Panel's model picker.
IMAGE_MODEL_CHOICES: dict[str, dict[str, str]] = {
    MODEL_IMAGE: {
        "label": "Nano Banana Pro (best quality)",
        "description": (
            "Premium: highest quality, 4K resolution, advanced text/brand "
            "accuracy, up to 5 reference images with high character fidelity."
        ),
    },
    MODEL_IMAGE_FLASH: {
        "label": "Nano Banana 2 (balanced)",
        "description": (
            "Versatile generalist: strong speed/cost/quality balance, up to "
            "4K, up to 4 reference images for character consistency."
        ),
    },
    MODEL_IMAGE_FLASH_LITE: {
        "label": "Nano Banana 2 Lite (fastest/cheapest)",
        "description": (
            "Fastest and cheapest option. Not optimized for multiple "
            "reference images or multi-turn sequential editing."
        ),
    },
    MODEL_IMAGE_LEGACY: {
        "label": "Nano Banana (legacy)",
        "description": (
            "Legacy 1024px model. Google recommends Nano Banana 2 Lite "
            "instead for new work; kept for compatibility."
        ),
    },
}

# Video model ids. Gemini Omni Flash is the only model on this same
# /interactions surface; Veo (veo-3.1-generate-preview etc.) uses a
# DIFFERENT, asynchronous predictLongRunning + polling API contract and
# is intentionally NOT offered as a drop-in model= choice here yet.
MODEL_VIDEO = "gemini-omni-flash-preview"   # Gemini Omni Flash — text/image -> video

# ── Output format ────────────────────────────────────────────────────────── #
# The output format is NOT requestable. Per
# ai.google.dev/gemini-api/docs/image-generation the response_format object
# is {"type": "image", "aspect_ratio": ..., "image_size": ...} -- there is no
# mime_type field, and sending one made the API reject every request (that is
# what broke generation entirely). The models return PNG, which every doc
# example saves straight to a .png file.
#
# That is convenient rather than limiting: PNG pixel data is plain zlib, so
# core/png.py can decode and shrink it with the standard library alone. JPEG
# would need a hand-rolled DCT/Huffman decoder, and the production runtime has
# no Pillow (verified: pillow_available=false), so a JPEG pipeline would have
# no way to build the display-sized preview the panel needs.
#
# image_size IS requestable and does reduce the render, but not nearly enough
# on its own: a 1K render still measured 1,029,068 base64 chars against a
# ~127,000 ceiling. Hence the local preview in core/preview.py.
IMAGE_SIZE_CHOICES: dict[str, str] = {
    "1K": "1K — default, lightest payload; the size that reliably displays",
    "2K": "2K — sharper, roughly 4x the pixels of 1K",
    "4K": "4K — maximum detail; very large payload, may not display inline",
}
DEFAULT_IMAGE_SIZE = "1K"

# Store collections
GENERATION_LOG_COLLECTION = "gm_generations"

# Limits
DEFAULT_HISTORY_LIMIT = 20
MAX_HISTORY_LIMIT = 50

# The panel shows the list itself (zero media I/O -- images load on demand),
# so it can afford a longer window than the chat function's default. A short
# window is why recent generations appeared to be missing entirely.
PANEL_HISTORY_LIMIT = 60
MAX_PROMPT_LEN = 4000

# HTTP
REQUEST_TIMEOUT_IMAGE = 60.0
REQUEST_TIMEOUT_VIDEO = 170.0  # video generation is slow; stay under the 180s federal cap
