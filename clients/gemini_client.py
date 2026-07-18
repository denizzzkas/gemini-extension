"""Thin client for the Gemini Interactions API (REST), used via ctx.http.

Contract (Google AI for Developers — /gemini-api/docs/get-started,
/gemini-api/docs/omni, /gemini-api/docs/image-generation — verified 2026-07):

    POST https://generativelanguage.googleapis.com/v1beta/interactions
    headers: x-goog-api-key: <API_KEY>, Content-Type: application/json
    body:    {"model": "<model-id>", "input": "<prompt>" | [...]}

Response is an ``Interaction`` resource, roughly:

    {
      "id": "...", "status": "completed",
      "steps": [
        {"type": "thought", ...},
        {"type": "model_output", "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "data": "<base64>", "mime_type": "image/png"},
            {"type": "video", "data": "<base64>", "mime_type": "video/mp4"}
        ]}
      ],
      "model": "...", "usage": {...}
    }

Both image models (Nano Banana / Nano Banana Pro) and the video model
(Gemini Omni Flash) share this exact request/response shape — only the
``model`` field changes. This module has no google-genai / SDK dependency;
it talks straight to the REST endpoint through the extension's ``ctx.http``
(the federal HTTP client — bounded timeout, no raw sockets).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gemini_config import GEMINI_API_BASE


class GeminiAPIError(Exception):
    """Raised when the Gemini API returns a non-2xx response or a malformed body."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Gemini API error {status_code}: {message}")


@dataclass
class GeneratedMedia:
    """One generated media block (image or video) extracted from a response."""

    kind: str            # "image" or "video"
    data_b64: str = ""
    mime_type: str = ""


@dataclass
class InteractionResult:
    """Normalized result of one Gemini Interactions API call."""

    interaction_id: str = ""
    status: str = ""
    model: str = ""
    text: str = ""
    media: list[GeneratedMedia] = field(default_factory=list)
    raw: dict | None = None


def _extract_error_message(body) -> str:
    """Best-effort extraction of a human-readable message from an error body."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return err.get("message") or str(err)
        if err:
            return str(err)
        return str(body)
    return str(body)


def _parse_interaction(body: dict) -> InteractionResult:
    result = InteractionResult(
        interaction_id=body.get("id", ""),
        status=body.get("status", ""),
        model=body.get("model", ""),
        raw=body,
    )
    texts: list[str] = []
    for step in body.get("steps", []) or []:
        for block in step.get("content", []) or []:
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                texts.append(block["text"])
            elif btype in ("image", "video"):
                result.media.append(
                    GeneratedMedia(
                        kind=btype,
                        data_b64=block.get("data", ""),
                        mime_type=block.get("mime_type", ""),
                    )
                )
    # Some responses also expose output_text / output_image / output_video
    # shortcuts at the top level (SDK convenience fields) — fall back to those
    # if the steps[] walk above found nothing, so we stay resilient to minor
    # response-shape variations between preview models.
    if not texts and body.get("output_text"):
        texts.append(body["output_text"])
    if not result.media:
        for key, kind in (("output_image", "image"), ("output_video", "video")):
            block = body.get(key)
            if isinstance(block, dict) and block.get("data"):
                result.media.append(
                    GeneratedMedia(
                        kind=kind,
                        data_b64=block.get("data", ""),
                        mime_type=block.get("mime_type", ""),
                    )
                )
    result.text = "\n".join(texts)
    return result


async def create_interaction(
    ctx,
    api_key: str,
    model: str,
    input_text: str,
    *,
    timeout: float = 60.0,
) -> InteractionResult:
    """Call POST /v1beta/interactions and return a normalized result.

    Raises GeminiAPIError on any non-2xx response or transport failure.
    """
    if not api_key:
        raise GeminiAPIError(401, "Missing Gemini API key")
    if not input_text or not input_text.strip():
        raise GeminiAPIError(400, "Prompt/input must not be empty")

    url = f"{GEMINI_API_BASE}/interactions"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": input_text}

    try:
        resp = await ctx.http.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — surface transport errors uniformly
        raise GeminiAPIError(0, f"Request to Gemini API failed: {e}") from e

    if resp.status_code >= 400:
        raise GeminiAPIError(resp.status_code, _extract_error_message(resp.json() if _looks_json(resp) else resp.text))

    body = resp.json() if _looks_json(resp) else {}
    if not isinstance(body, dict):
        raise GeminiAPIError(resp.status_code, "Unexpected non-object response body")
    return _parse_interaction(body)


def _looks_json(resp) -> bool:
    try:
        b = resp.json()
        return isinstance(b, (dict, list))
    except Exception:
        return False
