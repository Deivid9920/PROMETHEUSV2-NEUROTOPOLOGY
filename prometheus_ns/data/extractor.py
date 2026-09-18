"""Main-content extraction from raw crawled documents.

Primary extractor is trafilatura; readability-lxml is the fallback.
Boilerplate that must never reach the corpus is removed here: Project
Gutenberg license blocks (identical across books, they poison MinHash
and get memorized by the model) and bot-challenge pages (Cloudflare and
friends) that would otherwise pass as clean text.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.IGNORECASE)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

_CHALLENGE_MARKERS = (
    "cf-browser-verification",
    "cf_challenge",
    "checking your browser",
    "just a moment...",
    "attention required",
    "captcha",
    "are you a robot",
)

TagStripper = Callable[[str], "str | None"]


def looks_like_challenge(text: str) -> bool:
    """Return True when the payload is a bot-challenge page, not content."""
    lowered = text.lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Gutenberg license header and footer blocks.

    Everything before the START marker and after the END marker is
    dropped; when the markers are absent the text is returned unchanged.
    """
    start = _GUTENBERG_START.search(text)
    if start is None:
        return text
    body = text[start.end():]
    end = _GUTENBERG_END.search(body)
    if end is not None:
        body = body[:end.start()]
    return body


def _trafilatura_extract(html: str) -> str | None:
    import trafilatura

    return trafilatura.extract(html, include_comments=False)


def _readability_extract(html: str) -> str | None:
    from lxml import html as lxml_html
    from readability import Document

    summary = Document(html).summary(html_partial=True)
    if not summary:
        return None
    text = lxml_html.fromstring(summary).text_content()
    return text or None


def extract_text(
    html: str,
    source: str | None = None,
    primary: TagStripper | None = None,
    fallback: TagStripper | None = None,
) -> str | None:
    """Extract the main text of a crawled document.

    Args:
        html: Raw response payload.
        source: Source identifier used for source-specific boilerplate
            removal (``gutenberg`` strips license blocks).
        primary: Extractor used first; defaults to trafilatura.
        fallback: Extractor used when the primary returns nothing;
            defaults to readability-lxml.

    Returns:
        The extracted text, or None when both extractors fail or the
        payload is a bot challenge.
    """
    if looks_like_challenge(html):
        return None
    primary_fn = primary or _trafilatura_extract
    fallback_fn = fallback or _readability_extract
    text: str | None = None
    for extractor in (primary_fn, fallback_fn):
        try:
            text = extractor(html)
        except Exception:  # noqa: BLE001 - extractor bugs must not stop the pipeline
            text = None
        if text and text.strip():
            break
        text = None
    if not text:
        return None
    if source == "gutenberg":
        text = strip_gutenberg_boilerplate(text)
    return text.strip() or None
