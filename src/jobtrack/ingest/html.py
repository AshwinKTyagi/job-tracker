"""HTML to plain-text conversion.

The single entry point, :func:`html_to_text`, is deterministic: same input, same output,
every time. M2's purity invariant (I2) depends on that, since ``RawMessage.body_text`` is
built from this function's output and the classifier must never see it vary.
"""

from __future__ import annotations

import html as html_entities
import re
from typing import Final

from bs4 import BeautifulSoup

# Tags whose entire subtree carries no readable content for a job-application email.
_DROPPED_TAGS: Final[tuple[str, ...]] = ("script", "style", "head", "title", "noscript")

# Tags that visually break a line. A trailing newline is appended after each so that
# block boundaries survive `get_text()`, which otherwise concatenates everything.
_BLOCK_TAGS: Final[tuple[str, ...]] = (
    "br",
    "p",
    "div",
    "tr",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "table",
    "ul",
    "ol",
    "section",
    "article",
    "header",
    "footer",
    "hr",
)

# Runs of horizontal whitespace (not newlines) collapse to a single space.
_HORIZONTAL_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t\f\v\r\xa0]+")

# Three or more consecutive blank lines collapse to exactly one blank line.
_EXCESS_BLANK_LINES_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text.

    Drops ``script``/``style``/``head`` content, converts ``<br>`` and block-level tag
    boundaries into newlines, unescapes HTML entities, and collapses runs of whitespace.
    Safe to call on already-plain text (nothing to strip, entities pass through unchanged).

    Args:
        html: Raw HTML (or plain text) to convert.

    Returns:
        Deterministic plain text: trimmed, with horizontal whitespace collapsed to single
        spaces and blank-line runs collapsed to one blank line. Empty input yields "".
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for tag_name in _DROPPED_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append("\n")

    text = soup.get_text()
    text = html_entities.unescape(text)
    text = _HORIZONTAL_WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


__all__ = ["html_to_text"]
