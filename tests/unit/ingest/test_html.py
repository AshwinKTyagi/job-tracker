"""Tests for html_to_text: plain string -> string, deterministic."""

from __future__ import annotations

import pytest

from jobtrack.ingest.html import html_to_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   \n\t  ", ""),
        ("<p>Hello &amp; welcome</p>", "Hello & welcome"),
        ("Line1<br>Line2<br><br>Line3", "Line1\nLine2\n\nLine3"),
        ("<div>A</div><div>B</div>", "A\nB"),
        ("<script>evil()</script><p>Safe</p>", "Safe"),
        ("<style>.x{color:red}</style><p>Styled</p>", "Styled"),
        ("<head><title>T</title></head><body><p>Body</p></body>", "Body"),
        ("plain text   with    spaces", "plain text with spaces"),
        ("Para one\n\n\n\nPara two", "Para one\n\nPara two"),
        ("<ul><li>One</li><li>Two</li></ul>", "One\nTwo"),
        ("5 &lt; 10 and 9 &gt; 3", "5 < 10 and 9 > 3"),
        ("<b>Bold</b> and <i>italic</i>", "Bold and italic"),
        ("<table><tr><td>Row1</td></tr><tr><td>Row2</td></tr></table>", "Row1\nRow2"),
        ("<noscript>hidden</noscript><p>Visible</p>", "Visible"),
    ],
)
def test_html_to_text_cases(raw: str, expected: str) -> None:
    assert html_to_text(raw) == expected


def test_html_to_text_is_deterministic() -> None:
    raw = "<p>Hi <b>Alex</b></p><p>Next steps: <a href='https://x'>schedule</a></p>"
    assert html_to_text(raw) == html_to_text(raw)


def test_html_to_text_idempotent_on_plain_text() -> None:
    text = "Thanks for applying.\n\nBest,\nTeam"
    once = html_to_text(text)
    twice = html_to_text(once)
    assert once == twice == text


def test_html_to_text_preserves_literal_angle_brackets_in_plain_text() -> None:
    # A negative control: stray "<"/">" in plain-text bodies (e.g. "$5 < $10") must not
    # be swallowed as if they were tag syntax.
    text = "Your offer is $50 < $100 and > $10 minimum."
    assert html_to_text(text) == text
