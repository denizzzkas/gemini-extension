#!/usr/bin/env python3
"""Standalone smoke test — hits the REAL Gemini API with your own key.

No Imperal deployment needed. Just:

    export GEMINI_API_KEY="your-key-from-aistudio.google.com/apikey"
    python3 scripts/smoke_test.py image "a watercolor fox reading a book"
    python3 scripts/smoke_test.py video "a paper airplane flying through a city"

Saves the result as out.png / out.mp4 in the current directory.
"""
from __future__ import annotations

import base64
import os
import sys
import urllib.request
import urllib.error
import json

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL_IMAGE = "gemini-3-pro-image"
MODEL_VIDEO = "gemini-omni-flash-preview"


def call_gemini(api_key: str, model: str, prompt: str) -> dict:
    body = json.dumps({"model": model, "input": prompt}).encode()
    req = urllib.request.Request(
        INTERACTIONS_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=170) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        print(f"HTTP {e.code} error from Gemini API:\n{detail}", file=sys.stderr)
        sys.exit(1)


def extract_media(response: dict, kind: str) -> tuple[str, str] | None:
    """Returns (mime_type, base64_data) for the first block of `kind` found."""
    for step in response.get("steps", []):
        for block in step.get("content", []):
            if block.get("type") == kind:
                return block.get("mime_type", ""), block.get("data", "")
    return None


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/smoke_test.py <image|video> \"<prompt>\"")
        sys.exit(1)

    mode, prompt = sys.argv[1], sys.argv[2]
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY env var first: export GEMINI_API_KEY=your-key", file=sys.stderr)
        sys.exit(1)

    if mode == "image":
        model = MODEL_IMAGE
        out_path = "out.png"
    elif mode == "video":
        model = MODEL_VIDEO
        out_path = "out.mp4"
    else:
        print("First arg must be 'image' or 'video'", file=sys.stderr)
        sys.exit(1)

    print(f"Calling Gemini ({model}) with prompt: {prompt!r} ...")
    response = call_gemini(api_key, model, prompt)

    media = extract_media(response, mode)
    if media is None:
        print("No media returned. Full response:")
        print(json.dumps(response, indent=2)[:2000])
        sys.exit(1)

    mime_type, data_b64 = media
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(data_b64))

    print(f"Saved {out_path} ({mime_type}, {len(data_b64)} base64 chars)")


if __name__ == "__main__":
    main()
