"""Compose the charts into one self-contained HTML dashboard.

The output file inlines plotly.js, so it opens offline with no CDN and no network of any
kind. This module is the only part of ``viz/`` that touches the filesystem.

The rendered page is a pure function of its two DataFrames: no clock is read, so building
the same frames twice produces byte-identical HTML.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.errors import ExportError
from jobtrack.models import ApplicationStatus
from jobtrack.viz.charts import (
    applications_over_time,
    compute_stage_flows,
    funnel_sankey,
    response_time_histogram,
    status_bar_chart,
    top_companies_bar,
)

logger = logging.getLogger(__name__)

PLOTLY_CONFIG: Final[dict[str, object]] = {"displaylogo": False, "responsive": True}
"""Passed to every figure. No modebar image-export host, no telemetry, no network."""

_EMPTY_HEADLINE: Final[str] = "No applications yet"
_EMPTY_BODY: Final[str] = (
    "This dashboard is built from the applications in your local database. "
    "Run <code>jobtrack sync</code> to import mail, then regenerate it with "
    "<code>jobtrack dashboard</code>."
)

_STYLESHEET: Final[str] = """
:root {
  --bg: #f7f8fa;
  --surface: #ffffff;
  --ink: #16181d;
  --muted: #6b7280;
  --border: #e5e7eb;
  --accent: #4c78a8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 24px 64px;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.5;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }
.subtitle { color: var(--muted); margin: 0 0 28px; font-size: 0.92rem; }
.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
}
.tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
.tile .value { font-size: 1.7rem; font-weight: 600; letter-spacing: -0.02em; }
.tile .label {
  color: var(--muted);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 12px 12px;
  margin-bottom: 20px;
  overflow-x: auto;
}
.placeholder { padding: 40px 24px; text-align: center; color: var(--muted); }
.placeholder h2 { color: var(--ink); font-size: 1.1rem; margin: 0 0 8px; }
code {
  background: #eef1f5;
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 0.88em;
}
footer { color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 32px; }
"""


@dataclass(frozen=True)
class _Summary:
    """Headline numbers for the dashboard header. Module-internal value object."""

    total: int
    responded: int
    response_rate: float
    median_days_to_response: float | None
    interviewing: int
    offers: int
    rejected: int
    needs_review: int


def _summarize(applications_df: pd.DataFrame) -> _Summary:
    """Derive the header tiles from the applications frame."""
    if applications_df.empty:
        return _Summary(
            total=0,
            responded=0,
            response_rate=0.0,
            median_days_to_response=None,
            interviewing=0,
            offers=0,
            rejected=0,
            needs_review=0,
        )

    total = int(len(applications_df))
    days = pd.to_numeric(applications_df["days_to_first_response"], errors="coerce").dropna()
    responded = int(len(days))
    statuses = applications_df["status"].astype("str")
    counts = statuses.value_counts()
    flags = applications_df["needs_review"].fillna(False).astype("bool")

    return _Summary(
        total=total,
        responded=responded,
        response_rate=responded / total if total else 0.0,
        median_days_to_response=float(days.median()) if responded else None,
        interviewing=int(counts.get(ApplicationStatus.INTERVIEWING.value, 0)),
        offers=int(counts.get(ApplicationStatus.OFFER.value, 0)),
        rejected=int(counts.get(ApplicationStatus.REJECTED.value, 0)),
        needs_review=int(flags.sum()),
    )


def _tile(label: str, value: str) -> str:
    """One headline tile."""
    return (
        '<div class="tile">'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="label">{html.escape(label)}</div>'
        "</div>"
    )


def _render_tiles(summary: _Summary) -> str:
    """The grid of headline tiles."""
    median = summary.median_days_to_response
    tiles = [
        _tile("applications", str(summary.total)),
        _tile("responded", str(summary.responded)),
        _tile("response rate", f"{summary.response_rate * 100:.0f}%"),
        _tile("median days to reply", "—" if median is None else f"{median:g}"),
        _tile("interviewing", str(summary.interviewing)),
        _tile("offers", str(summary.offers)),
        _tile("rejected", str(summary.rejected)),
        _tile("needs review", str(summary.needs_review)),
    ]
    return '<section class="tiles">' + "".join(tiles) + "</section>"


def _figure_card(figure: go.Figure, div_id: str, *, height: int = 420) -> str:
    """Render one Figure as an HTML card, with plotly.js assumed already inlined."""
    fragment: str = figure.to_html(
        include_plotlyjs=False,
        include_mathjax=False,
        full_html=False,
        div_id=div_id,
        default_height=f"{height}px",
        config=PLOTLY_CONFIG,
    )
    return f'<section class="card">{fragment}</section>'


def _render_body(applications_df: pd.DataFrame, events_df: pd.DataFrame) -> str:
    """Every chart card, or an explanatory placeholder when there is nothing to plot."""
    if applications_df.empty:
        return (
            '<section class="card placeholder">'
            f"<h2>{html.escape(_EMPTY_HEADLINE)}</h2><p>{_EMPTY_BODY}</p>"
            "</section>"
        )

    flows = compute_stage_flows(events_df, applications_df)
    cards = [
        _figure_card(status_bar_chart(applications_df), "chart-status"),
        _figure_card(funnel_sankey(flows), "chart-funnel", height=460),
        _figure_card(applications_over_time(applications_df), "chart-timeline"),
        _figure_card(response_time_histogram(applications_df), "chart-response"),
    ]
    companies = top_companies_bar(applications_df)
    cards.append(_figure_card(companies, "chart-companies", height=int(companies.layout.height or 420)))
    return "".join(cards)


def _render_page(title: str, applications_df: pd.DataFrame, events_df: pd.DataFrame) -> str:
    """Assemble the full standalone HTML document, plotly.js inlined."""
    summary = _summarize(applications_df)
    safe_title = html.escape(title)
    subtitle = (
        f"{summary.total} application{'' if summary.total == 1 else 's'} tracked "
        "· generated locally, no network required"
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{safe_title}</title>\n"
        f"<style>{_STYLESHEET}</style>\n"
        f"<script>{get_plotlyjs()}</script>\n"
        "</head>\n<body>\n"
        f'<div class="wrap">\n<h1>{safe_title}</h1>\n'
        f'<p class="subtitle">{html.escape(subtitle)}</p>\n'
        f"{_render_tiles(summary)}\n"
        f"{_render_body(applications_df, events_df)}\n"
        "<footer>jobtrack · self-contained dashboard</footer>\n"
        "</div>\n</body>\n</html>\n"
    )


def build_dashboard(
    applications_df: pd.DataFrame,
    events_df: pd.DataFrame,
    path: Path,
    *,
    title: str = "Job Application Tracker",
) -> Path:
    """Compose every figure into ONE self-contained HTML file.

    plotly.js is inlined, so the file renders with no network and no CDN. The page opens
    with a summary header (totals, response rate, median time-to-response) and then the
    status bar chart, the stage Sankey, the submission timeline, the response-time
    histogram, and the per-company breakdown.

    Args:
        applications_df: Applications frame carrying EXPORT_COLUMNS (I10). An empty frame
            renders an explanatory placeholder rather than raising.
        events_df: Long-format events frame carrying EVENT_COLUMNS, used for the Sankey.
        path: Destination .html file. Parent directories are created as needed.
        title: Page title and heading.

    Returns:
        The resolved path that was written.

    Raises:
        ExportError: either frame is missing a required column, or the path is not writable.
    """
    from jobtrack.viz.charts import _require_columns  # noqa: PLC0415

    _require_columns(applications_df, EXPORT_COLUMNS, what="applications")
    _require_columns(events_df, EVENT_COLUMNS, what="events")

    page = _render_page(title, applications_df, events_df)
    destination = Path(path).expanduser()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(page, encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"could not write dashboard to {destination}: {exc}") from exc

    resolved = destination.resolve()
    logger.info("wrote dashboard to %s (%d bytes)", resolved, len(page))
    return resolved


__all__ = ["PLOTLY_CONFIG", "build_dashboard"]
