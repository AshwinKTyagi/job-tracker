"""Chart builders — `src/jobtrack/viz/charts.py`.

Each function takes a DataFrame shaped like M4's `build_dataframe` /
`build_events_dataframe` output (`constants.EXPORT_COLUMNS` / `constants.EVENT_COLUMNS`,
see CONTRACTS.md §7-8) and returns a configured `plotly.graph_objects.Figure`. No function
writes a file and none touches SQLite — viz only ever sees a DataFrame (I10).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

import pandas as pd
import plotly.graph_objects as go

from jobtrack.models import ApplicationStatus, EventType, StageFlow

logger = logging.getLogger(__name__)

# --- palette -------------------------------------------------------------------
# Fixed categorical order, never cycled or re-derived from the data (dataviz skill,
# color-formula.md). Every status-keyed chart in the dashboard uses this same mapping,
# so a status is always the same hue wherever it appears.
STATUS_PIPELINE_ORDER: tuple[ApplicationStatus, ...] = (
    ApplicationStatus.APPLIED,
    ApplicationStatus.ASSESSMENT,
    ApplicationStatus.INTERVIEWING,
    ApplicationStatus.OFFER,
    ApplicationStatus.REJECTED,
    ApplicationStatus.WITHDRAWN,
    ApplicationStatus.GHOSTED,
)
"""Pipeline order for every status-keyed chart: how far an application progressed,
not alphabetical order."""

STATUS_COLORS: dict[ApplicationStatus, str] = {
    ApplicationStatus.APPLIED: "#2a78d6",
    ApplicationStatus.ASSESSMENT: "#eb6834",
    ApplicationStatus.INTERVIEWING: "#1baf7a",
    ApplicationStatus.OFFER: "#eda100",
    ApplicationStatus.REJECTED: "#e87ba4",
    ApplicationStatus.WITHDRAWN: "#008300",
    ApplicationStatus.GHOSTED: "#4a3aa7",
}
"""First seven slots of the validated eight-hue categorical order (dataviz skill,
palette.md) taken in STATUS_PIPELINE_ORDER, so adjacent-pair CVD separation is preserved."""

CHART_SURFACE = "#fcfcfb"
GRIDLINE_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
SEQUENTIAL_HUE = "#3987e5"
"""Single-hue blue used for charts with no status dimension (timeline, histogram)."""

_LAYOUT_DEFAULTS: dict[str, object] = {
    "paper_bgcolor": CHART_SURFACE,
    "plot_bgcolor": CHART_SURFACE,
    "font": {"color": PRIMARY_INK, "family": "system-ui, -apple-system, 'Segoe UI', sans-serif"},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
}

# --- Sankey stage labels ---------------------------------------------------------
PIPELINE_STAGE_LABELS: dict[EventType, str] = {
    EventType.APPLICATION_RECEIVED: "Applied",
    EventType.ASSESSMENT: "Assessment",
    EventType.INTERVIEW: "Interview",
    EventType.OFFER: "Offer",
    EventType.REJECTION: "Rejected",
    EventType.WITHDRAWN: "Withdrawn",
}
"""Event types that mark a pipeline-stage transition. RECRUITER_OUTREACH and UNKNOWN
carry no stage of their own and are skipped when building a Sankey sequence."""

APPLIED_STAGE = "Applied"
GHOSTED_STAGE = "Ghosted"

SANKEY_NODE_ORDER: tuple[str, ...] = (
    "Applied",
    "Assessment",
    "Interview",
    "Offer",
    "Rejected",
    "Withdrawn",
    "Ghosted",
)
"""Left-to-right node order for the funnel Sankey. Terminal stages (Offer, Rejected,
Withdrawn, Ghosted) sit rightmost."""

_SANKEY_NODE_COLORS: dict[str, str] = {
    "Applied": STATUS_COLORS[ApplicationStatus.APPLIED],
    "Assessment": STATUS_COLORS[ApplicationStatus.ASSESSMENT],
    "Interview": STATUS_COLORS[ApplicationStatus.INTERVIEWING],
    "Offer": STATUS_COLORS[ApplicationStatus.OFFER],
    "Rejected": STATUS_COLORS[ApplicationStatus.REJECTED],
    "Withdrawn": STATUS_COLORS[ApplicationStatus.WITHDRAWN],
    "Ghosted": STATUS_COLORS[ApplicationStatus.GHOSTED],
}


def _base_figure(title: str) -> go.Figure:
    """Create an empty figure pre-styled with the dashboard's chart chrome."""
    fig = go.Figure()
    fig.update_layout(title=title, **_LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor=GRIDLINE_COLOR, linecolor=AXIS_COLOR, zerolinecolor=AXIS_COLOR)
    fig.update_yaxes(gridcolor=GRIDLINE_COLOR, linecolor=AXIS_COLOR, zerolinecolor=AXIS_COLOR)
    return fig


def _empty_annotation(fig: go.Figure, message: str) -> go.Figure:
    """Add a centered placeholder annotation to an otherwise-empty figure."""
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font={"color": MUTED_INK, "size": 14},
    )
    return fig


def status_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Application count by ApplicationStatus, ordered by pipeline stage.

    Args:
        df: applications frame shaped like `constants.EXPORT_COLUMNS`.

    Returns:
        A bar chart with one bar per pipeline stage, zero-filled for stages with no
        applications, ordered APPLIED -> ASSESSMENT -> INTERVIEWING ->
        OFFER/REJECTED/WITHDRAWN/GHOSTED rather than alphabetically.
    """
    fig = _base_figure("Applications by Status")
    if df.empty or "status" not in df.columns:
        return _empty_annotation(fig, "No applications yet")

    counts = df["status"].astype(str).value_counts()
    labels = [str(status) for status in STATUS_PIPELINE_ORDER]
    values = [int(counts.get(str(status), 0)) for status in STATUS_PIPELINE_ORDER]
    colors = [STATUS_COLORS[status] for status in STATUS_PIPELINE_ORDER]

    fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors, showlegend=False))
    fig.update_yaxes(title="Applications", rangemode="tozero")
    return fig


def applications_over_time(df: pd.DataFrame, *, freq: str = "W") -> go.Figure:
    """Applications submitted per period, from `applied_at`.

    Args:
        df: applications frame shaped like `constants.EXPORT_COLUMNS`.
        freq: pandas offset alias for the bucket width (default weekly).

    Returns:
        A filled line chart of application counts per period.
    """
    fig = _base_figure("Applications Over Time")
    if df.empty or "applied_at" not in df.columns:
        return _empty_annotation(fig, "No applications yet")

    applied_at = pd.to_datetime(df["applied_at"], utc=True, errors="coerce").dropna()
    if applied_at.empty:
        return _empty_annotation(fig, "No applications yet")

    # Bucketing is calendar-relative, not timezone-relative — drop the (already-UTC) tz
    # explicitly rather than let pandas warn about doing it implicitly. resample() (not
    # Period.to_period()) accepts every modern offset alias, including "ME"/"YE".
    naive = applied_at.dt.tz_localize(None).sort_values()
    counts = pd.Series(1, index=pd.DatetimeIndex(naive)).resample(freq).sum()

    fig.add_trace(
        go.Scatter(
            x=counts.index,
            y=counts.to_numpy(),
            mode="lines+markers",
            line={"color": SEQUENTIAL_HUE, "width": 2},
            marker={"size": 8, "color": SEQUENTIAL_HUE},
            fill="tozeroy",
            fillcolor="rgba(57, 135, 229, 0.12)",
            showlegend=False,
        )
    )
    fig.update_yaxes(title="Applications", rangemode="tozero")
    fig.update_xaxes(title="Period")
    return fig


def top_companies_bar(df: pd.DataFrame, *, top_n: int = 20) -> go.Figure:
    """Companies by application count, descending, stacked by status.

    Args:
        df: applications frame shaped like `constants.EXPORT_COLUMNS`.
        top_n: maximum number of companies to show.

    Returns:
        A horizontal stacked bar chart, one segment per status per company, companies
        ordered by total application count descending.
    """
    fig = _base_figure("Top Companies")
    required = {"company", "status"}
    if df.empty or not required.issubset(df.columns):
        return _empty_annotation(fig, "No applications yet")

    work = df.loc[:, ["company", "status"]].copy()
    work["status"] = work["status"].astype(str)
    totals = work.groupby("company").size().sort_values(ascending=False)
    top_companies = list(totals.head(max(top_n, 0)).index)
    if not top_companies:
        return _empty_annotation(fig, "No applications yet")

    pivot = (
        work[work["company"].isin(top_companies)]
        .groupby(["company", "status"])
        .size()
        .unstack("status", fill_value=0)
        .reindex(index=top_companies)
    )

    for status in STATUS_PIPELINE_ORDER:
        column = str(status)
        if column not in pivot.columns:
            continue
        values = pivot[column]
        if values.sum() == 0:
            continue
        fig.add_trace(
            go.Bar(
                y=list(pivot.index),
                x=values.to_numpy(),
                name=column.replace("_", " ").title(),
                marker_color=STATUS_COLORS[status],
                orientation="h",
            )
        )

    fig.update_layout(barmode="stack", legend={"title": {"text": "Status"}})
    # Highest-count company on top; plotly's default category order is bottom-up.
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(top_companies)))
    fig.update_xaxes(title="Applications")
    return fig


def response_time_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of `days_to_first_response`, excluding nulls.

    Args:
        df: applications frame shaped like `constants.EXPORT_COLUMNS`.

    Returns:
        A histogram of response times in days.
    """
    fig = _base_figure("Time to First Response")
    if df.empty or "days_to_first_response" not in df.columns:
        return _empty_annotation(fig, "No response data yet")

    values = pd.to_numeric(df["days_to_first_response"], errors="coerce").dropna()
    if values.empty:
        return _empty_annotation(fig, "No response data yet")

    fig.add_trace(go.Histogram(x=values.to_numpy(), marker_color=SEQUENTIAL_HUE, showlegend=False))
    fig.update_xaxes(title="Days to first response")
    fig.update_yaxes(title="Applications", rangemode="tozero")
    return fig


def _record_transition(
    counts: Counter[tuple[str, str]], first_seen: list[tuple[str, str]], source: str, target: str
) -> None:
    """Increment a transition's count, recording its first-seen order once."""
    key = (source, target)
    if key not in counts:
        first_seen.append(key)
    counts[key] += 1


def compute_stage_flows(events_df: pd.DataFrame, applications_df: pd.DataFrame) -> list[StageFlow]:
    """Turn per-application event sequences into Sankey links.

    PURE — unit-testable directly, no plotly involved. Every application, by virtue of
    existing in `applications_df`, has reached the "Applied" stage. From there, each
    application contributes consecutive transitions through the further stages it
    actually reached (Applied -> Assessment -> Interview -> Offer/Rejected/Withdrawn),
    collapsing repeated consecutive events into a single stage. Applications whose
    derived status is GHOSTED additionally contribute one terminal link from the last
    stage they reached into 'Ghosted'.

    Args:
        events_df: events frame shaped like `constants.EVENT_COLUMNS`.
        applications_df: applications frame shaped like `constants.EXPORT_COLUMNS`.

    Returns:
        One StageFlow per distinct (source, target) transition that actually occurred,
        in first-seen order. Transitions with a zero count are never produced, so none
        need dropping.
    """
    required_app_cols = {"application_id", "status"}
    if applications_df.empty or not required_app_cols.issubset(applications_df.columns):
        return []

    required_event_cols = {"application_id", "event_type", "occurred_at"}
    has_events = not events_df.empty and required_event_cols.issubset(events_df.columns)

    if has_events:
        events = events_df.loc[:, ["application_id", "event_type", "occurred_at"]].copy()
        events["event_type"] = events["event_type"].astype(str)
        events = events.sort_values("occurred_at")
    else:
        events = pd.DataFrame(columns=["application_id", "event_type", "occurred_at"])

    stage_by_event_value = {
        str(event_type): label for event_type, label in PIPELINE_STAGE_LABELS.items()
    }

    transition_counts: Counter[tuple[str, str]] = Counter()
    first_seen: list[tuple[str, str]] = []

    for _, application in applications_df.iterrows():
        application_id = application["application_id"]
        status = str(application["status"])

        app_events = events[events["application_id"] == application_id] if has_events else events
        stages = [APPLIED_STAGE]
        for event_type in app_events["event_type"]:
            label = stage_by_event_value.get(event_type)
            if label is not None and stages[-1] != label:
                stages.append(label)

        for source, target in pairwise(stages):
            _record_transition(transition_counts, first_seen, source, target)

        if status == str(ApplicationStatus.GHOSTED):
            _record_transition(transition_counts, first_seen, stages[-1], GHOSTED_STAGE)

    return [
        StageFlow(source=source, target=target, count=transition_counts[(source, target)])
        for source, target in first_seen
    ]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a `#rrggbb` hex color to an `rgba()` string at the given alpha."""
    stripped = hex_color.lstrip("#")
    r, g, b = (int(stripped[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def funnel_sankey(flows: Sequence[StageFlow]) -> go.Figure:
    """Render stage flows as a `plotly.graph_objects.Sankey`.

    Args:
        flows: StageFlow links, typically from `compute_stage_flows`.

    Returns:
        A Sankey figure. Node order follows pipeline stage; terminal nodes
        (Offer/Rejected/Withdrawn/Ghosted) sit rightmost.
    """
    fig = _base_figure("Application Funnel")
    if not flows:
        return _empty_annotation(fig, "No stage transitions yet")

    present_nodes = {flow.source for flow in flows} | {flow.target for flow in flows}
    nodes = [label for label in SANKEY_NODE_ORDER if label in present_nodes]
    nodes.extend(sorted(present_nodes - set(nodes)))
    node_index = {label: i for i, label in enumerate(nodes)}
    node_colors = [_SANKEY_NODE_COLORS.get(label, MUTED_INK) for label in nodes]

    fig.add_trace(
        go.Sankey(
            arrangement="snap",
            node={
                "label": nodes,
                "color": node_colors,
                "pad": 16,
                "thickness": 18,
                "line": {"color": AXIS_COLOR, "width": 0.5},
            },
            link={
                "source": [node_index[flow.source] for flow in flows],
                "target": [node_index[flow.target] for flow in flows],
                "value": [flow.count for flow in flows],
                "color": [
                    _hex_to_rgba(_SANKEY_NODE_COLORS.get(flow.target, MUTED_INK), 0.35)
                    for flow in flows
                ],
            },
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


__all__ = [
    "PIPELINE_STAGE_LABELS",
    "SANKEY_NODE_ORDER",
    "STATUS_COLORS",
    "STATUS_PIPELINE_ORDER",
    "applications_over_time",
    "compute_stage_flows",
    "funnel_sankey",
    "response_time_histogram",
    "status_bar_chart",
    "top_companies_bar",
]
