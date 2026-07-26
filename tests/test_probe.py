"""Tests for the payload-ceiling probe.

The probe is a measuring instrument, so what is under test is its
TRUSTWORTHINESS: it must hit the requested size, stay incompressible, emit a
decodable PNG, stay off unless asked for, and never steal the single left
panel slot from the working content.
"""
from __future__ import annotations

import base64
import zlib

import pytest

from tests.fixtures import make_ctx


def test_probe_hits_the_requested_size():
    """A ruler that lies about its own length measures nothing."""
    from handlers.probe import _noise_png

    for kb in (100, 400, 900):
        png = _noise_png(kb * 1024)
        chars = len(base64.b64encode(png))
        target = kb * 1024
        # Within 5%: the pixel grid is squared off, so exactness is impossible.
        assert abs(chars - target) / target < 0.05, (
            f"{kb}KB requested but produced {chars} base64 chars"
        )


def test_probe_payload_is_incompressible():
    """A compressible payload would report a falsely high ceiling.

    This is the exact trap that let a 68KB white square "pass" while real
    renders failed -- a flat colour flattens to almost nothing under gzip.
    """
    from handlers.probe import _noise_png

    png = _noise_png(300 * 1024)
    assert len(zlib.compress(png, 6)) >= len(png) * 0.95


def test_probe_emits_a_decodable_png():
    """If the bytes were not a real PNG, a blank panel would be ambiguous."""
    from handlers.probe import _noise_png

    png = _noise_png(100 * 1024)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    # IHDR must be the first chunk and IEND the last, per the PNG spec.
    assert png[12:16] == b"IHDR"
    assert png.endswith(b"IEND\xae\x42\x60\x82")


def test_probe_is_off_unless_requested():
    """The diagnostic must not occupy the panel during normal use."""
    from handlers.probe import probe_section

    assert probe_section("gemini_quick", {}) is None
    assert probe_section("gemini_quick", {"probe_kb": ""}) is None


def test_probe_renders_the_requested_payload():
    from handlers.probe import probe_section

    node = probe_section("gemini_quick", {"probe_kb": "200"})
    assert node is not None
    tree = node.to_dict()

    def _find(n, node_type):
        if isinstance(n, dict):
            if n.get("type") == node_type:
                return n
            for v in n.values():
                found = _find(v, node_type)
                if found:
                    return found
        elif isinstance(n, list):
            for item in n:
                found = _find(item, node_type)
                if found:
                    return found
        return None

    img = _find(tree, "Image")
    assert img is not None
    src = img["props"]["src"]
    assert src.startswith("data:image/png;base64,")
    # The rendered payload must actually BE the requested size.
    assert abs(len(src) - 200 * 1024) / (200 * 1024) < 0.05


@pytest.mark.asyncio
async def test_probe_does_not_replace_panel_content():
    """Opening the probe must keep the real panel usable.

    Regression guard: the first draft registered the probe as a SECOND left
    panel, which would have hidden the working one entirely.
    """
    from handlers.panel import gemini_quick_panel

    ctx = make_ctx(with_key=True)
    node = await gemini_quick_panel(ctx, probe_kb="100")
    tree = node.to_dict()

    types: list[str] = []

    def _collect(n):
        if isinstance(n, dict):
            if "type" in n:
                types.append(n["type"])
            for v in n.values():
                _collect(v)
        elif isinstance(n, list):
            for item in n:
                _collect(item)

    _collect(tree)
    # The generation form still has to be there alongside the probe image.
    assert "Form" in types
    assert "Image" in types
