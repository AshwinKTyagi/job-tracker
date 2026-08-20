"""Plotly figures for the application dashboard.

Every function here is a pure transform: it takes the DataFrame shapes frozen by
``constants.EXPORT_COLUMNS`` / ``constants.EVENT_COLUMNS`` (invariant I10) and returns a
configured ``plotly.graph_objects.Figure``. Nothing in this module writes a file, opens a
socket, or knows that SQLite exists — composition and output belong to ``dashboard.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from itertools import pairwise
from typing import Final

import pandas as pd
import plotly.graph_objects as go

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.errors import ExportError
from jobtrack.models import ApplicationStatus, EventType, StageFlow

logger = logging.getLogger(__name__)

STATUS_ORDER: Final[tuple[ApplicationStatus, ...]] = (
    ApplicationStatus.APPLIED,
    ApplicationStatus.ASSESSMENT,
    ApplicationStatus.INTERVIEWING,
    ApplicationStatus.OFFER,
    ApplicationStatus.REJECTED,
    ApplicationStatus.WITHDRAWN,
    ApplicationStatus.GHOSTED,
)
"""Pipeline order, not alphabetical order. Every status axis in this module uses it."""

STATUS_COLORS: Final[dict[ApplicationStatus, str]] = {
    ApplicationStatus.APPLIED: "#4c78a8",
    ApplicationStatus.ASSESSMENT: "#72b7b2",
    ApplicationStatus.INTERVIEWING: "#f58518",
    ApplicationStatus.OFFER: "#54a24b",
    ApplicationStatus.REJECTED: "#e45756",
    ApplicationStatus.WITHDRAWN: "#b279a2",
    ApplicationStatus.GHOSTED: "#9c9c9c",
}

STAGE_OUTREACH: Final[str] = "Outreach"
STAGE_APPLIED: Final[str] = "Applied"
STAGE_ASSESSMENT: Final[str] = "Assessment"
STAGE_INTERVIEW: Final[str] = "Interview"
STAGE_OFFER: Final[str] = "Offer"
STAGE_REJECTED: Final[str] = "Rejected"
STAGE_WITHDRAWN: Final[str] = "Withdrawn"
STAGE_GHOSTED: Final[str] = "Ghosted"

STAGE_ORDER: Final[tuple[str, ...]] = (
    STAGE_OUTREACH,
    STAGE_APPLIED,
    STAGE_ASSESSMENT,
    STAGE_INTERVIEW,
    STAGE_OFFER,
    STAGE_REJECTED,
    STAGE_WITHDRAWN,
    STAGE_GHOSTED,
)
"""Sankey node order. Progression stages first, terminal stages last so they sit rightmost."""

TERMINAL_STAGES: Final[frozenset[str]] = frozenset(
    {STAGE_OFFER, STAGE_REJECTED, STAGE_WITHDRAWN, STAGE_GHOSTED}
)
"""A sequence stops here: nothing follows a terminal stage."""

STAGE_BY_EVENT: Final[dict[str, str]] = {
    EventType.RECRUITER_OUTREACH.value: STAGE_OUTREACH,
    EventType.APPLICATION_RECEIVED.value: STAGE_APPLIED,
    EventType.ASSESSMENT.value: STAGE_ASSESSMENT,
    EventType.INTERVIEW.value: STAGE_INTERVIEW,
    EventType.OFFER.value: STAGE_OFFER,
    EventType.REJECTION.value: STAGE_REJECTED,
    EventType.WITHDRAWN.value: STAGE_WITHDRAWN,
}
"""EventType value -> Sankey stage. UNKNOWN is deliberately absent: it is not a stage."""

STAGE_COLORS: Final[dict[str, str]] = {
    STAGE_OUTREACH: "#8c8cd9",
    STAGE_APPLIED: "#4c78a8",
    STAGE_ASSESSMENT: "#72b7b2",
    STAGE_INTERVIEW: "#f58518",
    STAGE_OFFER: "#54a24b",
    STAGE_REJECTED: "#e45756",
    STAGE_WITHDRAWN: "#b279a2",
    STAGE_GHOSTED: "#9c9c9c",
}

_FALLBACK_COLOR: Final[str] = "#6b7280"
_GRID_COLOR: Final[str] = "#e5e7eb"
_FONT_FAMILY: Final[str] = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif"
_EMPTY_MESSAGE: Final[str] = "No data yet — run <b>jobtrack sync</b> to populate the dashboard."
_MARGIN: Final[dict[str, int]] = {"l": 60, "r": 30, "t": 60, "b": 60}


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, what: str) -> None:
    """Fail loudly when the wire format (I10) is not what this module was promised."""
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ExportError(f"{what} frame is missing required columns: {', '.join(missing)}")


def _base_figure(title: str) -> go.Figure:
    """A Figure with the shared house layout already applied."""
    figure = go.Figure()
    figure.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left"},
        font={"family": _FONT_FAMILY, "size": 13},
        margin=_MARGIN,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    figure.update_xaxes(gridcolor=_GRID_COLOR, zeroline=False)
    figure.update_yaxes(gridcolor=_GRID_COLOR, zeroline=False)
    return figure


def _empty_figure(title: str, message: str = _EMPTY_MESSAGE) -> go.Figure:
    """An axis-free placeholder figure carrying an explanatory annotation."""
    figure = _base_figure(title)
    figure.add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font={"size": 14, "color": _FALLBACK_COLOR},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _status_series(df: pd.DataFrame) -> pd.Series:
    """The status column as plain strings, so StrEnum and str frames behave identically."""
    return df["status"].astype("str")


def status_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Application count by ApplicationStatus, ordered by pipeline stage.

    Args:
        df: Applications frame carrying EXPORT_COLUMNS (I10).

    Returns:
        A bar Figure with one bar per ApplicationStatus, in STATUS_ORDER. Statuses with no
        applications are kept at zero so the chart's shape is stable between runs.

    Raises:
        ExportError: `df` is missing a required column.
    """
    _require_columns(df, EXPORT_COLUMNS, what="applications")
    if df.empty:
        return _empty_figure("Applications by status")

    counts = _status_series(df).value_counts()
    labels = [status.value for status in STATUS_ORDER]
    values = [int(counts.get(label, 0)) for label in labels]

    figure = _base_figure("Applications by status")
    figure.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[STATUS_COLORS[status] for status in STATUS_ORDER],
            text=values,
            textposition="outside",
            hovertemplate="%{x}: %{y} applications<extra></extra>",
            name="applications",
        )
    )
    figure.update_layout(showlegend=False, yaxis_title="applications")
    figure.update_xaxes(categoryorder="array", categoryarray=labels)
    return figure


def applications_over_time(df: pd.DataFrame, *, freq: str = "W") -> go.Figure:
    """Applications submitted per period, from applied_at.

    Args:
        df: Applications frame carrying EXPORT_COLUMNS (I10).
        freq: A pandas offset alias for the bucket width, e.g. "D", "W", "ME".

    Returns:
        A bar Figure of submissions per period, oldest bucket first.

    Raises:
        ExportError: `df` is missing a required column, or `freq` is not a valid alias.
    """
    _require_columns(df, EXPORT_COLUMNS, what="applications")
    title = f"Applications submitted per period ({freq})"
    if df.empty:
        return _empty_figure(title)

    applied = pd.to_datetime(df["applied_at"], utc=True)
    series = pd.Series(1, index=pd.DatetimeIndex(applied), dtype="int64").sort_index()
    try:
        counts = series.resample(freq).sum()
    except ValueError as exc:  # pandas rejects an unparseable offset alias
        raise ExportError(f"invalid resample frequency {freq!r}: {exc}") from exc

    figure = _base_figure(title)
    figure.add_trace(
        go.Bar(
            x=list(counts.index),
            y=[int(value) for value in counts.to_numpy()],
            marker_color=STATUS_COLORS[ApplicationStatus.APPLIED],
            hovertemplate="%{x|%Y-%m-%d}: %{y} applications<extra></extra>",
            name="applications",
        )
    )
    figure.update_layout(showlegend=False, xaxis_title="period start", yaxis_title="applications")
    return figure


def top_companies_bar(df: pd.DataFrame, *, top_n: int = 20) -> go.Figure:
    """Companies by application count, descending, stacked by status.

    Args:
        df: Applications frame carrying EXPORT_COLUMNS (I10).
        top_n: Keep at most this many companies. Values below 1 yield a placeholder.

    Returns:
        A horizontal stacked-bar Figure, busiest company first.

    Raises:
        ExportError: `df` is missing a required column.
    """
    _require_columns(df, EXPORT_COLUMNS, what="applications")
    title = f"Top {top_n} companies by applications"
    if df.empty or top_n < 1:
        return _empty_figure(title)

    companies = df["company"].astype("str")
    totals = companies.value_counts().head(top_n)
    # Plotly draws the first category at the bottom of a horizontal axis, so reverse to put
    # the busiest company on top.
    ordered = list(reversed(totals.index.tolist()))

    statuses = _status_series(df)
    figure = _base_figure(title)
    for status in STATUS_ORDER:
        matching = companies[statuses == status.value].value_counts()
        values = [int(matching.get(company, 0)) for company in ordered]
        if not any(values):
            continue
        figure.add_trace(
            go.Bar(
                x=values,
                y=ordered,
                name=status.value,
                orientation="h",
                marker_color=STATUS_COLORS[status],
                hovertemplate="%{y} — " + status.value + ": %{x}<extra></extra>",
            )
        )

    figure.update_layout(
        barmode="stack",
        xaxis_title="applications",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        height=max(320, 26 * len(ordered) + 140),
    )
    figure.update_yaxes(categoryorder="array", categoryarray=ordered, automargin=True)
    return figure


def response_time_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of days_to_first_response, excluding nulls.

    Args:
        df: Applications frame carrying EXPORT_COLUMNS (I10).

    Returns:
        A histogram Figure with a dashed line at the median. When no application has heard
        back yet, an explanatory placeholder is returned instead.

    Raises:
        ExportError: `df` is missing a required column.
    """
    _require_columns(df, EXPORT_COLUMNS, what="applications")
    title = "Days to first response"
    if df.empty:
        return _empty_figure(title)

    days = pd.to_numeric(df["days_to_first_response"], errors="coerce").dropna()
    if days.empty:
        return _empty_figure(title, "No application has had a first response yet.")

    values = [float(value) for value in days.to_numpy()]
    figure = _base_figure(title)
    figure.add_trace(
        go.Histogram(
            x=values,
            marker_color=STATUS_COLORS[ApplicationStatus.INTERVIEWING],
            hovertemplate="%{x} days: %{y} applications<extra></extra>",
            name="applications",
        )
    )
    median = float(days.median())
    figure.add_vline(
        x=median,
        line_dash="dash",
        line_color=_FALLBACK_COLOR,
        annotation_text=f"median {median:g}d",
        annotation_position="top",
    )
    figure.update_layout(showlegend=False, xaxis_title="days", yaxis_title="applications")
    return figure


def _ghosted_application_ids(applications_df: pd.DataFrame) -> frozenset[str]:
    """Application ids whose derived status is GHOSTED."""
    if applications_df.empty:
        return frozenset()
    ghosted = applications_df[_status_series(applications_df) == ApplicationStatus.GHOSTED.value]
    return frozenset(str(value) for value in ghosted["application_id"].tolist())


def _stage_sequences(events_df: pd.DataFrame) -> dict[str, list[str]]:
    """Per-application stage progressions, oldest first, monotonically forward."""
    if events_df.empty:
        return {}

    linked = events_df[events_df["application_id"].notna()]
    if linked.empty:
        return {}
    ordered = linked.sort_values(["application_id", "occurred_at"], kind="stable")

    sequences: dict[str, list[str]] = {}
    for application_id, event_type in zip(
        ordered["application_id"].tolist(), ordered["event_type"].tolist(), strict=True
    ):
        stage = STAGE_BY_EVENT.get(str(event_type))
        if stage is None:
            continue
        stages = sequences.setdefault(str(application_id), [])
        if stages:
            last = stages[-1]
            if last in TERMINAL_STAGES:
                continue
            # Only ever move forward: repeats and out-of-order events add no link.
            if STAGE_ORDER.index(stage) <= STAGE_ORDER.index(last):
                continue
        stages.append(stage)
    return sequences


def compute_stage_flows(events_df: pd.DataFrame, applications_df: pd.DataFrame) -> list[StageFlow]:
    """Turn per-application event sequences into Sankey links. PURE.

    Each application contributes consecutive transitions through the stages it actually
    reached (Applied -> Assessment -> Interview -> Offer/Rejected). Repeated and
    out-of-order events add no link, and nothing follows a terminal stage. Applications
    that stalled (status GHOSTED) contribute a terminal link into 'Ghosted'. Zero-count
    links are dropped.

    Args:
        events_df: Long-format events frame carrying EVENT_COLUMNS. Rows with no
            application_id (unlinked UNKNOWNs) are ignored.
        applications_df: Applications frame carrying EXPORT_COLUMNS, used only to find
            which applications are ghosted.

    Returns:
        Links sorted by (source stage, target stage) in STAGE_ORDER — deterministic for a
        given pair of frames.

    Raises:
        ExportError: either frame is missing a required column.
    """
    _require_columns(events_df, EVENT_COLUMNS, what="events")
    _require_columns(applications_df, EXPORT_COLUMNS, what="applications")

    sequences = _stage_sequences(events_df)
    ghosted = _ghosted_application_ids(applications_df)

    counts: dict[tuple[str, str], int] = {}
    for application_id in sorted(sequences):
        stages = sequences[application_id]
        for source, target in pairwise(stages):
            counts[(source, target)] = counts.get((source, target), 0) + 1
        last = stages[-1]
        if application_id in ghosted and last not in TERMINAL_STAGES:
            key = (last, STAGE_GHOSTED)
            counts[key] = counts.get(key, 0) + 1

    def _sort_key(item: tuple[tuple[str, str], int]) -> tuple[int, int]:
        source, target = item[0]
        return (STAGE_ORDER.index(source), STAGE_ORDER.index(target))

    return [
        StageFlow(source=source, target=target, count=count)
        for (source, target), count in sorted(counts.items(), key=_sort_key)
        if count > 0
    ]


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to an rgba() string at the given alpha."""
    raw = hex_color.lstrip("#")
    red, green, blue = (int(raw[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def funnel_sankey(flows: Sequence[StageFlow]) -> go.Figure:
    """Render stage flows as a plotly.graph_objects.Sankey.

    Args:
        flows: Links, typically from compute_stage_flows.

    Returns:
        A Sankey Figure whose node order follows STAGE_ORDER, so terminal nodes
        (Offer/Rejected/Withdrawn/Ghosted) sit rightmost. Empty input yields a placeholder.
    """
    title = "Application funnel"
    if not flows:
        return _empty_figure(title, "No stage transitions recorded yet.")

    present = {flow.source for flow in flows} | {flow.target for flow in flows}
    labels = [stage for stage in STAGE_ORDER if stage in present]
    labels += sorted(present - set(STAGE_ORDER))
    index = {label: position for position, label in enumerate(labels)}

    figure = _base_figure(title)
    figure.add_trace(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "color": [STAGE_COLORS.get(label, _FALLBACK_COLOR) for label in labels],
                "pad": 18,
                "thickness": 18,
                "line": {"color": "#ffffff", "width": 0.5},
            },
            link={
                "source": [index[flow.source] for flow in flows],
                "target": [index[flow.target] for flow in flows],
                "value": [flow.count for flow in flows],
                "color": [
                    _rgba(STAGE_COLORS.get(flow.target, _FALLBACK_COLOR), 0.35) for flow in flows
                ],
            },
        )
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


__all__ = [
    "STAGE_BY_EVENT",
    "STAGE_COLORS",
    "STAGE_ORDER",
    "STATUS_COLORS",
    "STATUS_ORDER",
    "TERMINAL_STAGES",
    "applications_over_time",
    "compute_stage_flows",
    "funnel_sankey",
    "response_time_histogram",
    "status_bar_chart",
    "top_companies_bar",
]
