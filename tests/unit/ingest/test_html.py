"""Tests for ``ingest.html``.

The load-bearing property is determinism: M2 promises that the same RawMessage yields a
byte-identical Classification (I2), and the body text M2 reads is produced here. Several
tests below exist only to pin that down.
"""

from __future__ import annotations

import pytest

from jobtrack.ingest.html import collapse_whitespace, html_to_text


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", "\t \r\n"])
def test_blank_input_yields_empty_string(blank: str) -> None:
    assert html_to_text(blank) == ""


def test_drops_script_style_and_head() -> None:
    html = (
        "<html><head><title>ignored</title><style>.a{color:red}</style></head>"
        "<body><script>alert('x')</script><p>Real prose.</p></body></html>"
    )
    assert html_to_text(html) == "Real prose."


def test_block_elements_become_line_breaks() -> None:
    text = html_to_text("<p>One</p><p>Two</p><div>Three</div>")
    assert text.split("\n\n") == ["One", "Two", "Three"]


def test_br_becomes_a_newline() -> None:
    assert html_to_text("first<br>second") == "first\n\nsecond"


def test_inline_tags_do_not_split_a_phrase() -> None:
    """The classifier matches on phrases; ``<b>`` must not shred one into two lines."""
    text = html_to_text("<p>Thank you for <b>applying</b> to <span>Acme</span>.</p>")
    assert text == "Thank you for applying to Acme."


def test_list_items_are_separate_lines() -> None:
    text = html_to_text("<ul><li>Screen</li><li>Onsite</li></ul>")
    assert "Screen" in text
    assert "Onsite" in text
    assert "ScreenOnsite" not in text


def test_table_cells_join_with_a_space_not_a_newline() -> None:
    """Layout tables are the standard email scaffolding; one word per line ruins matching."""
    text = html_to_text("<table><tr><td>Role:</td><td>Senior Engineer</td></tr></table>")
    assert text == "Role: Senior Engineer"


def test_entities_are_unescaped_exactly_once() -> None:
    """A literal ``&amp;amp;`` must survive as ``&amp;``, not decay into ``&``."""
    assert html_to_text("<p>Tom &amp; Jerry</p>") == "Tom & Jerry"
    assert html_to_text("<p>&amp;amp;</p>") == "&amp;"


def test_non_breaking_and_zero_width_characters_are_normalized() -> None:
    """Invisible characters render as nothing but defeat a substring match."""
    raw = "<p>Thank\u00a0you\u200b for\u00ad applying</p>"  # nbsp, ZWSP, soft hyphen
    assert html_to_text(raw) == "Thank you for applying"


def test_plain_text_input_passes_through() -> None:
    assert html_to_text("no markup &amp; some text") == "no markup & some text"


def test_malformed_markup_is_tolerated() -> None:
    text = html_to_text("<p>unclosed <b>bold<p>next")
    assert "unclosed" in text
    assert "next" in text


def test_output_is_deterministic_across_repeated_calls() -> None:
    """I2 depends on this: same input in, byte-identical text out, every time."""
    html = (
        "<html><body><table><tr><td>Hi&nbsp;Alex,</td></tr></table>"
        "<p>Thanks for <i>applying</i> to Acme.</p><br><div>Best,</div>"
        "<script>x()</script><ul><li>a</li><li>b</li></ul></body></html>"
    )
    results = {html_to_text(html) for _ in range(10)}
    assert len(results) == 1


def test_html_to_text_output_is_stable_under_reprocessing() -> None:
    """Its output is already collapsed, so collapsing again changes nothing."""
    once = html_to_text("<p>a</p><p>b</p>")
    assert collapse_whitespace(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a   \t  b", "a b"),
        ("  padded  ", "padded"),
        ("line   \n   next", "line\nnext"),
        ("a\n\n\n\n\nb", "a\n\nb"),
        ("", ""),
    ],
)
def test_collapse_whitespace_cases(raw: str, expected: str) -> None:
    assert collapse_whitespace(raw) == expected


def test_collapse_whitespace_is_idempotent() -> None:
    raw = "  Dear Alex,\r\n\r\n\r\n   Thanks   for applying.  \r\n"
    once = collapse_whitespace(raw)
    assert collapse_whitespace(once) == once


def test_paragraph_structure_survives() -> None:
    """Blank lines carry meaning; the classifier reads salutation and body separately."""
    text = html_to_text("<p>Hi Alex,</p><p>Unfortunately we are moving forward.</p><p>Best,</p>")
    assert text == "Hi Alex,\n\nUnfortunately we are moving forward.\n\nBest,"
