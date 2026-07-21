"""Tests for Gemini media helpers (_absolute_url storage link normalization)."""
from __future__ import annotations

from handlers.media import _absolute_url


def test_absolute_url_passes_through_full_urls():
    assert _absolute_url("https://cdn.example.com/x.png") == "https://cdn.example.com/x.png"
    assert _absolute_url("http://cdn.example.com/x.png") == "http://cdn.example.com/x.png"


def test_absolute_url_prefixes_bare_storage_path():
    # This is the exact shape ctx.storage.upload() can return -- a bare path,
    # not a clickable link. Regression test for the "dead link in chat" bug.
    result = _absolute_url("/storage/default/gemeni/0811e960ddf94a8c926370cff6bbb7b5.jpg")
    assert result.startswith("https://")
    assert result.endswith("/storage/default/gemeni/0811e960ddf94a8c926370cff6bbb7b5.jpg")


def test_absolute_url_empty_stays_empty():
    assert _absolute_url("") == ""
