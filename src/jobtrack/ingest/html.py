"""HTML to plain text.

Marketing and ATS emails are almost always ``multipart/alternative`` with a text/plain
part, but a meaningful minority are HTML-only. Those still have to reach the classifier
as readable prose, which is what this module produces.

Determinism is a hard requirement, not a nicety: M2's purity guarantee (invariant I2) says
the same ``RawMessage`` must classify to a byte-identical ``Classification``, and the body
text that M2 sees comes out of here. So: a stdlib-backed parser (``html.parser``, never
``lxml``, whose output drifts between releases), no dict iteration, no set ordering.
"""

from __future__ import annotations

import logging
import re
from typing import Final

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

PARSER: Final[str] = "html.parser"
"""Stdlib-backed and version-stable. Do not switch to lxml — see the module docstring."""

DROPPED_TAGS: Final[tuple[str, ...]] = (
    "script",
    "style",
    "head",
    "title",
    "meta",
    "link",
    "noscript",
    "template",
)
"""Elements whose text is markup or metadata, never prose the classifier should read."""

BLOCK_TAGS: Final[tuple[str, ...]] = (
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
)
"""Elements that end a line. Inline tags (``<b>``, ``<a>``, ``<span>``) deliberately do NOT
appear here: injecting whitespace inside ``thank you for <b>applying</b>`` would break the
phrase patterns M2 matches on."""

CELL_TAGS: Final[tuple[str, ...]] = ("td", "th")
"""Table cells separate with a space rather than a newline, so a one-row layout table (the
usual email scaffolding) does not shred a sentence into one word per line."""

_HORIZONTAL_WS: Final[re.Pattern[str]] = re.compile(r"[^\S\n]+")
"""Runs of whitespace that are *not* newlines — newlines carry paragraph structure."""

_BLANK_RUN: Final[re.Pattern[str]] = re.compile(r"\n{3,}")

_INVISIBLE: Final[tuple[tuple[str, str], ...]] = (
    (" ", " "),  # non-breaking space — pervasive in HTML mail
    ("​", ""),  # zero-width space
    ("‌", ""),  # zero-width non-joiner
    ("‍", ""),  # zero-width joiner
    ("﻿", ""),  # BOM / zero-width no-break space
    ("­", ""),  # soft hyphen
)
"""Characters that render as nothing but defeat a substring match. Ordered, so the
substitution sequence is reproducible."""


def collapse_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks.

    Converts CRLF/CR to LF, replaces invisible characters, collapses runs of horizontal
    whitespace to a single space, strips each line, and caps consecutive blank lines at
    one. Deterministic and idempotent.

    Args:
        text: Any plain text.

    Returns:
        The normalized text, with no leading or trailing whitespace.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for needle, replacement in _INVISIBLE:
        normalized = normalized.replace(needle, replacement)
    normalized = _HORIZONTAL_WS.sub(" ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))
    return _BLANK_RUN.sub("\n\n", normalized).strip()


def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text.

    Drops script/style/head, converts ``<br>`` and block ends to newlines, unescapes
    entities, and collapses runs of whitespace. Deterministic — M2's purity (I2) depends
    on it.

    Entity unescaping happens exactly once, inside the parser; this function never calls
    ``html.unescape`` on the parser's output, which would turn a literal ``&amp;lt;`` into
    a real ``<``.

    Args:
        html: An HTML document or fragment. Plain text and malformed markup are accepted
            and pass through the same normalization.

    Returns:
        Plain text with paragraph breaks preserved, or "" for blank input.
    """
    if not html.strip():
        return ""

    soup = BeautifulSoup(html, PARSER)

    for dropped in soup.find_all(list(DROPPED_TAGS)):
        dropped.decompose()

    # Materialize before mutating: inserting siblings while iterating a live result set
    # is not order-stable.
    for cell in list(soup.find_all(list(CELL_TAGS))):
        cell.insert_after(" ")
    for block in list(soup.find_all(list(BLOCK_TAGS))):
        block.insert_before("\n")
        block.insert_after("\n")

    return collapse_whitespace(soup.get_text())


__all__ = [
    "BLOCK_TAGS",
    "CELL_TAGS",
    "DROPPED_TAGS",
    "PARSER",
    "collapse_whitespace",
    "html_to_text",
]
