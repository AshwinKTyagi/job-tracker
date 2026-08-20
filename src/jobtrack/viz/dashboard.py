"""Composes every `viz/charts.py` figure into one self-contained HTML dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from jobtrack.errors import ExportError
from jobtrack.viz.charts import (
    applications_over_time,
    compute_stage_flows,
    funnel_sankey,
    response_time_histogram,
    status_bar_chart,
    top_companies_bar,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Summary:
    """Headline numbers shown in the dashboard's summary header."""

    total: int
    responded: int
    response_rate: float
    median_response_days: float | None
    needs_review: int


def _summarize(df: pd.DataFrame) -> _Summary:
    """Compute the summary header stats from the applications frame.

    Args:
        df: applications frame shaped like `constants.EXPORT_COLUMNS`.

    Returns:
        Totals, response rate, and median time-to-response for the header tiles.
    """
    if df.empty or "application_id" not in df.columns:
        return _Summary(
            total=0, responded=0, response_rate=0.0, median_response_days=None, needs_review=0
        )

    total = len(df)
    responded_series = (
        pd.to_numeric(df["days_to_first_response"], errors="coerce").dropna()
        if "days_to_first_response" in df.columns
        else pd.Series(dtype="float64")
    )
    responded = len(responded_series)
    response_rate = responded / total if total else 0.0
    median_response = float(responded_series.median()) if responded else None
    needs_review = int(df["needs_review"].astype(bool).sum()) if "needs_review" in df.columns else 0

    return _Summary(
        total=total,
        responded=responded,
        response_rate=response_rate,
        median_response_days=median_response,
        needs_review=needs_review,
    )


def _stat_tile(label: str, value: str) -> str:
    """Render one stat tile as an HTML fragment."""
    return (
        '<div class="stat-tile">'
        f'<div class="stat-value">{escape(value)}</div>'
        f'<div class="stat-label">{escape(label)}</div>'
        "</div>"
    )


def _summary_header(summary: _Summary) -> str:
    """Render the summary stat row as an HTML fragment."""
    median_text = (
        f"{summary.median_response_days:.0f} days"
        if summary.median_response_days is not None
        else "N/A"
    )
    tiles = [
        _stat_tile("Total applications", str(summary.total)),
        _stat_tile("Responses received", str(summary.responded)),
        _stat_tile("Response rate", f"{summary.response_rate * 100:.0f}%"),
        _stat_tile("Median time to response", median_text),
        _stat_tile("Needs review", str(summary.needs_review)),
    ]
    return f'<div class="stat-row">{"".join(tiles)}</div>'


_STYLE = """
body {
  margin: 0;
  padding: 32px;
  background: #f9f9f7;
  color: #0b0b0b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { margin: 0 0 24px; font-size: 24px; }
.stat-row { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 32px; }
.stat-tile {
  background: #fcfcfb;
  border: 1px solid rgba(11, 11, 11, 0.10);
  border-radius: 8px;
  padding: 16px 20px;
  min-width: 160px;
}
.stat-value { font-size: 28px; font-weight: 600; }
.stat-label { font-size: 13px; color: #52514e; margin-top: 4px; }
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 24px;
}
.chart-card {
  background: #fcfcfb;
  border: 1px solid rgba(11, 11, 11, 0.10);
  border-radius: 8px;
  padding: 8px;
}
.placeholder {
  padding: 24px;
  margin-bottom: 24px;
  text-align: center;
  color: #898781;
  background: #fcfcfb;
  border: 1px solid rgba(11, 11, 11, 0.10);
  border-radius: 8px;
}
"""


def _chart_div(fig: go.Figure, *, chart_id: str) -> str:
    """Render one figure to an HTML fragment, assuming plotly.js is already inlined."""
    html: str = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
    return html


def build_dashboard(
    applications_df: pd.DataFrame,
    events_df: pd.DataFrame,
    path: Path,
    *,
    title: str = "Job Application Tracker",
) -> Path:
    """Compose every chart into one self-contained HTML file.

    plotly.js is inlined directly into the page (`plotly.offline.get_plotlyjs()`), so
    the file renders offline with no CDN reference. An empty `applications_df` still
    renders a full page — with an explanatory placeholder banner and each chart's own
    "no data" placeholder — rather than raising.

    Args:
        applications_df: applications frame shaped like `constants.EXPORT_COLUMNS`.
        events_df: events frame shaped like `constants.EVENT_COLUMNS`.
        path: destination file path. Parent directories are created if needed.
        title: page title and header text.

    Returns:
        The resolved path the dashboard was written to.

    Raises:
        ExportError: `path` is not writable.
    """
    summary = _summarize(applications_df)
    placeholder = (
        '<p class="placeholder">No applications recorded yet — run `jobtrack sync` first.</p>'
        if applications_df.empty
        else ""
    )

    figures = [
        status_bar_chart(applications_df),
        applications_over_time(applications_df),
        top_companies_bar(applications_df),
        response_time_histogram(applications_df),
        funnel_sankey(compute_stage_flows(events_df, applications_df)),
    ]

    chart_divs = "".join(
        f'<div class="chart-card">{_chart_div(fig, chart_id=f"chart-{i}")}</div>'
        for i, fig in enumerate(figures)
    )

    html = (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        f"<style>{_STYLE}</style>"
        f"<script>{pyo.get_plotlyjs()}</script>"
        "</head><body>"
        f"<h1>{escape(title)}</h1>"
        f"{_summary_header(summary)}"
        f"{placeholder}"
        f'<div class="chart-grid">{chart_divs}</div>'
        "</body></html>"
    )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"could not write dashboard to {path}: {exc}") from exc

    resolved = path.resolve()
    logger.info("dashboard written to %s (%d applications)", resolved, summary.total)
    return resolved


__all__ = ["build_dashboard"]
