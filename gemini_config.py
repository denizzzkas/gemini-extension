"""Gemini API constants and Imperal store collection names."""

# Gemini API (Interactions API — unified generateContent-style surface)
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Model ids (Gemini API model catalogue, as of 2026-07)
MODEL_IMAGE = "gemini-3-pro-image"          # Nano Banana Pro — image generation/editing
MODEL_IMAGE_FLASH = "gemini-3.1-flash-image"  # Nano Banana — cheaper/faster image model
MODEL_VIDEO = "gemini-omni-flash-preview"   # Gemini Omni Flash — text/image -> video

# Store collections
GENERATION_LOG_COLLECTION = "gm_generations"

# Limits
DEFAULT_HISTORY_LIMIT = 20
MAX_HISTORY_LIMIT = 50
MAX_PROMPT_LEN = 4000

# HTTP
REQUEST_TIMEOUT_IMAGE = 60.0
REQUEST_TIMEOUT_VIDEO = 170.0  # video generation is slow; stay under the 180s federal cap
