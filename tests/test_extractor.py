"""Tests for main-content extraction and boilerplate handling."""

from __future__ import annotations

from pathlib import Path

from prometheus_ns.data.extractor import (
    extract_text,
    looks_like_challenge,
    strip_gutenberg_boilerplate,
)

FIXTURES = Path(__file__).parent / "fixtures"

GUTENBERG_TEXT = """\
The Project Gutenberg eBook of A Voyage to the Sea

*** START OF THE PROJECT GUTENBERG EBOOK A VOYAGE TO THE SEA ***

The ship left Bristol on a cold morning in March and sailed south
for many weeks. The crew was small but experienced.

*** END OF THE PROJECT GUTENBERG EBOOK A VOYAGE TO THE SEA ***

Updated editions will replace the previous one—the old editions will
be renamed. This license applies to every copy of this work.
"""

CHALLENGE_HTML = """<html><head><title>Just a moment...</title></head>
<body><div class="cf-browser-verification">Checking your browser before accessing.</div></body></html>"""

PLAIN_HTML = """<html><body><article>
<h1>The Old Mill</h1>
<p>The old mill stood at the edge of the river and the wheel turned slowly every day.
The miller woke before dawn and opened the wooden doors while the mist still covered the water.
Farmers brought sacks of wheat from the surrounding valleys and left with flour before noon.</p>
</article></body></html>"""


def test_gutenberg_license_is_stripped() -> None:
    body = strip_gutenberg_boilerplate(GUTENBERG_TEXT)
    assert "START OF THE PROJECT GUTENBERG" not in body
    assert "Updated editions will replace" not in body
    assert "The ship left Bristol" in body


def test_text_without_markers_is_untouched() -> None:
    assert strip_gutenberg_boilerplate("plain text without markers") == "plain text without markers"


def test_challenge_pages_are_detected() -> None:
    assert looks_like_challenge(CHALLENGE_HTML)
    assert not looks_like_challenge(PLAIN_HTML)


def test_extract_text_returns_none_for_challenge() -> None:
    assert extract_text(CHALLENGE_HTML) is None


def test_extract_text_primary_extractor() -> None:
    text = extract_text(PLAIN_HTML)
    assert text is not None
    assert "old mill" in text.lower()


def test_extract_text_falls_back_when_primary_fails() -> None:
    calls = []

    def failing_primary(_html: str) -> str | None:
        calls.append("primary")
        return None

    def working_fallback(html: str) -> str | None:
        calls.append("fallback")
        return "recovered text from fallback extractor"

    text = extract_text(PLAIN_HTML, primary=failing_primary, fallback=working_fallback)
    assert text == "recovered text from fallback extractor"
    assert calls == ["primary", "fallback"]


def test_extract_text_none_when_both_fail() -> None:
    text = extract_text("<html></html>", primary=lambda _h: None, fallback=lambda _h: None)
    assert text is None


def test_gutenberg_source_flag_strips_license_from_extracted_text() -> None:
    def fake_primary(_html: str) -> str | None:
        return GUTENBERG_TEXT

    text = extract_text(PLAIN_HTML, source="gutenberg", primary=fake_primary)
    assert "PROJECT GUTENBERG" not in (text or "")
