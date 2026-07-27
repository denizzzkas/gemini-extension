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

# ── Output format (the payload fix, applied AT THE SOURCE) ───────────────── #
# Measured in production: a default PNG render is ~940KB raw -> ~1.25M base64
# chars, which the panel does not render, while ~127k chars does. The shrink
# path that was supposed to bridge that gap depends on Pillow, which is NOT
# installed in the production runtime (verified: pillow_available=false), so
# it never runs. Asking Gemini for a compact image in the first place needs no
# third-party library at all.
#
# Contract per ai.google.dev/gemini-api/docs/image-generation:
#   response_format = {"type": "image", "mime_type": ..., "aspect_ratio": ...,
#                      "image_size": "1K" | "2K" | "4K"}
# Gemini 3 image models default to 1K; the K must be uppercase.
#
# WHY PNG AND NOT JPEG, despite JPEG being lighter for photos: neither format
# fits inline at full size (a 1K JPEG still measured 1,029,068 base64 chars,
# only 18% below the PNG it replaced -- i.e. asking for a smaller render is
# NOT on its own enough). The payload has to be shrunk locally, and the only
# decoder available without third-party libraries is PNG's: its pixel stream
# is plain zlib (see core/png.py), whereas JPEG would need a hand-rolled
# DCT/Huffman implementation. Choosing PNG is what makes core/preview.py able
# to produce a display-sized preview in the real runtime; the full-resolution
# original stays in storage untouched.
DEFAULT_IMAGE_MIME = "image/png"    # PNG is the only format shrinkable without Pillow
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
