"""Shared test fixtures for the Gemini extension tests."""
from __future__ import annotations

import base64

from imperal_sdk.testing import MockContext, MockSecretStore
from imperal_sdk.testing.mock_context import MockHTTP

# httpx-style .request() passthrough, mirrored from other Imperal extensions'
# test suites so tests exercise the same call shape as production code.
if not hasattr(MockHTTP, "request"):
    async def _mock_http_request(self, method: str, url: str, **kwargs):
        return await getattr(self, method.lower())(url, **kwargs)
    MockHTTP.request = _mock_http_request

FAKE_IMAGE_B64 = base64.b64encode(b"fake-png-bytes").decode()
FAKE_VIDEO_B64 = base64.b64encode(b"fake-mp4-bytes").decode()

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

SAMPLE_IMAGE_RESPONSE = {
    "id": "interaction_img_1",
    "status": "completed",
    "model": "gemini-3-pro-image",
    "steps": [
        {
            "type": "model_output",
            "content": [
                {"type": "text", "text": "Here is your image."},
                {"type": "image", "data": FAKE_IMAGE_B64, "mime_type": "image/png"},
            ],
        }
    ],
}

SAMPLE_VIDEO_RESPONSE = {
    "id": "interaction_vid_1",
    "status": "completed",
    "model": "gemini-omni-flash-preview",
    "steps": [
        {
            "type": "model_output",
            "content": [
                {"type": "text", "text": "Here is your video."},
                {"type": "video", "data": FAKE_VIDEO_B64, "mime_type": "video/mp4"},
            ],
        }
    ],
}


def make_ctx(with_key: bool = True):
    """Build a MockContext wired up with a MockSecretStore for gemini_api_key."""
    ctx = MockContext(user_id="test_user")
    initial = {"gemini_api_key": "test-api-key-123"} if with_key else {}
    ctx.secrets = MockSecretStore(initial, declared={"gemini_api_key"})
    return ctx
