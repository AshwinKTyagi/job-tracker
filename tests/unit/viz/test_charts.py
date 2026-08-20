"""Unit tests for the figure builders and the pure stage-flow transform.

No figure function may write a file, and none of them may reach for the clock or the
network. `compute_stage_flows` is tested as a pure function, independently of any figure.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.errors import ExportError
from jobtrack.models import ApplicationStatus, EventType, StageFlow
from jobtrack.viz.charts import (
    STAGE_BY_EVENT,
    STAGE_COLORS,
    STAGE_ORDER,
    STATUS_COLORS,
    STATUS_ORDER,
    TERMINAL_STAGES,
    applications_over_time,
    compute_stage_flows,
    funnel_sankey,
    response_time_histogram,
    status_bar_chart,
    top_companies_bar,
)

from .conftest import (
    BASE_TIME,
    applications_frame,
    events_frame,
    linear_pipeline,
    make_application,
    make_event,
)


def _annotation_texts(figure: go.Figure) -> list[str]:
    """Every annotation string on a figure, for placeholder assertions."""
    return [str(annotation.text) for annotation in figure.layout.annotations]


# --- module tables ---------------------------------------------------------------------


def test_status_order_covers_every_status() -> None:
    assert set(STATUS_ORDER) == set(ApplicationStatus)
    assert len(STATUS_ORDER) == len(ApplicationStatus)


def test_status_colors_cover_every_status() -> None:
    assert set(STATUS_COLORS) == set(ApplicationStatus)


def test_stage_table_covers_every_event_type_except_unknown() -> None:
    mapped = {EventType(value) for value in STAGE_BY_EVENT}
    assert mapped == set(EventType) - {EventType.UNKNOWN}
    assert set(STAGE_BY_EVENT.values()) <= set(STAGE_ORDER)


def test_terminal_stages_are_the_rightmost_stages() -> None:
    assert set(STAGE_ORDER) >= TERMINAL_STAGES
    assert set(STAGE_ORDER[-len(TERMINAL_STAGES) :]) == TERMINAL_STAGES
    assert set(STAGE_COLORS) == set(STAGE_ORDER)


# --- status_bar_chart ------------------------------------------------------------------


def test_status_bar_chart_orders_by_pipeline_stage(sample_applications: pd.DataFrame) -> None:
    figure = status_bar_chart(sample_applications)
    assert list(figure.data[0].x) == [status.value for status in STATUS_ORDER]
    assert list(figure.data[0].y) == [1, 0, 1, 1, 1, 0, 1]


def test_status_bar_chart_keeps_empty_statuses_at_zero() -> None:
    df = applications_frame([make_application(status=ApplicationStatus.OFFER)])
    figure = status_bar_chart(df)
    assert sum(figure.data[0].y) == 1
    assert len(figure.data[0].y) == len(STATUS_ORDER)


def test_status_bar_chart_accepts_raw_strenum_cells() -> None:
    """A frame carrying StrEnum objects rather than plain strings must behave identically."""
    df = applications_frame([make_application(status=ApplicationStatus.OFFER)])
    enum_df = df.assign(status=[ApplicationStatus.OFFER])
    assert list(status_bar_chart(enum_df).data[0].y) == list(status_bar_chart(df).data[0].y)


def test_status_bar_chart_empty_renders_placeholder(empty_applications: pd.DataFrame) -> None:
    figure = status_bar_chart(empty_applications)
    assert len(figure.data) == 0
    assert "No data yet" in _annotation_texts(figure)[0]


def test_status_bar_chart_rejects_a_missing_column(sample_applications: pd.DataFrame) -> None:
    with pytest.raises(ExportError, match="status"):
        status_bar_chart(sample_applications.drop(columns=["status"]))


# --- applications_over_time ------------------------------------------------------------


def test_applications_over_time_counts_every_application(
    sample_applications: pd.DataFrame,
) -> None:
    figure = applications_over_time(sample_applications)
    assert sum(figure.data[0].y) == len(sample_applications)


def test_applications_over_time_buckets_by_frequency() -> None:
    df = applications_frame(
        [
            make_application(application_id="a", applied_at=BASE_TIME),
            make_application(application_id="b", applied_at=BASE_TIME + timedelta(hours=3)),
            make_application(application_id="c", applied_at=BASE_TIME + timedelta(days=1)),
        ]
    )
    daily = applications_over_time(df, freq="D")
    assert list(daily.data[0].y) == [2, 1]
    weekly = applications_over_time(df, freq="W")
    assert list(weekly.data[0].y) == [3]


def test_applications_over_time_keeps_buckets_tz_aware_utc() -> None:
    df = applications_frame([make_application(applied_at=BASE_TIME)])
    figure = applications_over_time(df, freq="D")
    assert all(stamp.tzinfo is not None for stamp in figure.data[0].x)
    assert str(figure.data[0].x[0].tzinfo) == "UTC"


def test_applications_over_time_rejects_a_bad_frequency(
    sample_applications: pd.DataFrame,
) -> None:
    with pytest.raises(ExportError, match="invalid resample frequency"):
        applications_over_time(sample_applications, freq="not-a-frequency")


def test_applications_over_time_empty_renders_placeholder(
    empty_applications: pd.DataFrame,
) -> None:
    figure = applications_over_time(empty_applications)
    assert len(figure.data) == 0
    assert _annotation_texts(figure)


# --- top_companies_bar -----------------------------------------------------------------


def test_top_companies_bar_is_stacked_by_status(sample_applications: pd.DataFrame) -> None:
    figure = top_companies_bar(sample_applications)
    assert figure.layout.barmode == "stack"
    names = [trace.name for trace in figure.data]
    assert names == ["applied", "interviewing", "offer", "rejected", "ghosted"]
    assert sum(sum(trace.x) for trace in figure.data) == len(sample_applications)


def test_top_companies_bar_puts_the_busiest_company_last_in_y_order(
    sample_applications: pd.DataFrame,
) -> None:
    """Plotly draws horizontal categories bottom-up, so the busiest company comes last."""
    figure = top_companies_bar(sample_applications)
    assert list(figure.data[0].y)[-1] == "Acme Robotics"


def test_top_companies_bar_honours_top_n() -> None:
    df = applications_frame(
        [make_application(application_id=f"a{index}", company=f"Co {index}") for index in range(5)]
    )
    figure = top_companies_bar(df, top_n=2)
    assert len(figure.data[0].y) == 2


def test_top_companies_bar_with_non_positive_top_n_renders_placeholder(
    sample_applications: pd.DataFrame,
) -> None:
    figure = top_companies_bar(sample_applications, top_n=0)
    assert len(figure.data) == 0


def test_top_companies_bar_empty_renders_placeholder(empty_applications: pd.DataFrame) -> None:
    assert len(top_companies_bar(empty_applications).data) == 0


# --- response_time_histogram -----------------------------------------------------------


def test_response_time_histogram_excludes_nulls(sample_applications: pd.DataFrame) -> None:
    figure = response_time_histogram(sample_applications)
    assert sorted(figure.data[0].x) == [4.0, 6.0, 8.0]


def test_response_time_histogram_marks_the_median(sample_applications: pd.DataFrame) -> None:
    figure = response_time_histogram(sample_applications)
    assert any("median 6d" in text for text in _annotation_texts(figure))


def test_response_time_histogram_with_no_responses_renders_placeholder() -> None:
    df = applications_frame([make_application(days_to_first_response=None)])
    figure = response_time_histogram(df)
    assert len(figure.data) == 0
    assert "first response" in _annotation_texts(figure)[0]


def test_response_time_histogram_empty_renders_placeholder(
    empty_applications: pd.DataFrame,
) -> None:
    assert len(response_time_histogram(empty_applications).data) == 0


# --- compute_stage_flows (pure) --------------------------------------------------------


def _flow_pairs(flows: list[StageFlow]) -> list[tuple[str, str, int]]:
    """Flows as plain tuples, so assertions read as the diagram they describe."""
    return [(flow.source, flow.target, flow.count) for flow in flows]


def test_compute_stage_flows_on_empty_frames(
    empty_events: pd.DataFrame, empty_applications: pd.DataFrame
) -> None:
    assert compute_stage_flows(empty_events, empty_applications) == []


def test_compute_stage_flows_follows_a_linear_pipeline() -> None:
    events = events_frame(
        linear_pipeline(
            "app-1",
            [
                EventType.APPLICATION_RECEIVED,
                EventType.ASSESSMENT,
                EventType.INTERVIEW,
                EventType.OFFER,
            ],
        )
    )
    apps = applications_frame([make_application(application_id="app-1")])
    assert _flow_pairs(compute_stage_flows(events, apps)) == [
        ("Applied", "Assessment", 1),
        ("Assessment", "Interview", 1),
        ("Interview", "Offer", 1),
    ]


def test_compute_stage_flows_aggregates_across_applications(
    sample_events: pd.DataFrame, sample_applications: pd.DataFrame
) -> None:
    assert _flow_pairs(compute_stage_flows(sample_events, sample_applications)) == [
        ("Applied", "Assessment", 1),
        ("Applied", "Interview", 2),
        ("Applied", "Ghosted", 1),
        ("Assessment", "Interview", 1),
        ("Interview", "Offer", 1),
        ("Interview", "Rejected", 1),
    ]


def test_compute_stage_flows_ignores_repeated_stages() -> None:
    events = events_frame(
        linear_pipeline(
            "app-1",
            [
                EventType.APPLICATION_RECEIVED,
                EventType.INTERVIEW,
                EventType.INTERVIEW,
                EventType.INTERVIEW,
            ],
        )
    )
    apps = applications_frame([make_application(application_id="app-1")])
    assert _flow_pairs(compute_stage_flows(events, apps)) == [("Applied", "Interview", 1)]


def test_compute_stage_flows_never_moves_backwards() -> None:
    """A late confirmation email must not draw an Interview -> Applied link."""
    events = events_frame(
        linear_pipeline(
            "app-1",
            [
                EventType.APPLICATION_RECEIVED,
                EventType.INTERVIEW,
                EventType.APPLICATION_RECEIVED,
            ],
        )
    )
    apps = applications_frame([make_application(application_id="app-1")])
    assert _flow_pairs(compute_stage_flows(events, apps)) == [("Applied", "Interview", 1)]


def test_compute_stage_flows_stops_at_a_terminal_stage() -> None:
    events = events_frame(
        linear_pipeline(
            "app-1",
            [EventType.APPLICATION_RECEIVED, EventType.REJECTION, EventType.INTERVIEW],
        )
    )
    apps = applications_frame([make_application(application_id="app-1")])
    assert _flow_pairs(compute_stage_flows(events, apps)) == [("Applied", "Rejected", 1)]


def test_compute_stage_flows_orders_events_by_occurred_at() -> None:
    """Row order in the frame is irrelevant; occurred_at decides the sequence."""
    rows = linear_pipeline(
        "app-1", [EventType.APPLICATION_RECEIVED, EventType.INTERVIEW, EventType.OFFER]
    )
    shuffled = events_frame([rows[2], rows[0], rows[1]])
    apps = applications_frame([make_application(application_id="app-1")])
    assert _flow_pairs(compute_stage_flows(shuffled, apps)) == [
        ("Applied", "Interview", 1),
        ("Interview", "Offer", 1),
    ]


def test_compute_stage_flows_ignores_unknown_and_unlinked_events() -> None:
    events = events_frame(
        [
            *linear_pipeline("app-1", [EventType.APPLICATION_RECEIVED, EventType.INTERVIEW]),
            make_event(
                event_id=50,
                application_id="app-1",
                message_id="noise",
                event_type=EventType.UNKNOWN,
                occurred_at=BASE_TIME + timedelta(days=5),
            ),
            make_event(
                event_id=51,
                application_id=None,
                message_id="unlinked",
                event_type=EventType.INTERVIEW,
                occurred_at=BASE_TIME + timedelta(days=6),
            ),
        ]
    )
    apps = applications_frame([make_application(application_id="app-1")])
    assert _flow_pairs(compute_stage_flows(events, apps)) == [("Applied", "Interview", 1)]


def test_compute_stage_flows_with_only_unlinked_events(
    empty_applications: pd.DataFrame,
) -> None:
    events = events_frame(
        [
            make_event(event_id=1, application_id=None, message_id="a"),
            make_event(event_id=2, application_id=None, message_id="b"),
        ]
    )
    assert compute_stage_flows(events, empty_applications) == []


def test_compute_stage_flows_sends_stalled_applications_to_ghosted() -> None:
    events = events_frame(linear_pipeline("app-1", [EventType.APPLICATION_RECEIVED]))
    apps = applications_frame(
        [make_application(application_id="app-1", status=ApplicationStatus.GHOSTED)]
    )
    assert _flow_pairs(compute_stage_flows(events, apps)) == [("Applied", "Ghosted", 1)]


def test_compute_stage_flows_does_not_ghost_a_terminal_application() -> None:
    events = events_frame(
        linear_pipeline("app-1", [EventType.APPLICATION_RECEIVED, EventType.REJECTION])
    )
    apps = applications_frame(
        [make_application(application_id="app-1", status=ApplicationStatus.GHOSTED)]
    )
    assert _flow_pairs(compute_stage_flows(events, apps)) == [("Applied", "Rejected", 1)]


def test_compute_stage_flows_handles_a_recruiter_outreach_start() -> None:
    events = events_frame(
        linear_pipeline(
            "app-1",
            [
                EventType.RECRUITER_OUTREACH,
                EventType.APPLICATION_RECEIVED,
                EventType.WITHDRAWN,
            ],
        )
    )
    apps = applications_frame([make_application(application_id="app-1")])
    assert _flow_pairs(compute_stage_flows(events, apps)) == [
        ("Outreach", "Applied", 1),
        ("Applied", "Withdrawn", 1),
    ]


def test_compute_stage_flows_drops_zero_counts_and_returns_stage_flows(
    sample_events: pd.DataFrame, sample_applications: pd.DataFrame
) -> None:
    flows = compute_stage_flows(sample_events, sample_applications)
    assert all(isinstance(flow, StageFlow) for flow in flows)
    assert all(flow.count > 0 for flow in flows)


def test_compute_stage_flows_is_deterministic(
    sample_events: pd.DataFrame, sample_applications: pd.DataFrame
) -> None:
    first = compute_stage_flows(sample_events, sample_applications)
    second = compute_stage_flows(sample_events, sample_applications)
    assert [flow.model_dump() for flow in first] == [flow.model_dump() for flow in second]


def test_compute_stage_flows_does_not_mutate_its_inputs(
    sample_events: pd.DataFrame, sample_applications: pd.DataFrame
) -> None:
    events_before = sample_events.copy()
    apps_before = sample_applications.copy()
    compute_stage_flows(sample_events, sample_applications)
    pd.testing.assert_frame_equal(sample_events, events_before)
    pd.testing.assert_frame_equal(sample_applications, apps_before)


@pytest.mark.parametrize("column", ["application_id", "occurred_at", "event_type"])
def test_compute_stage_flows_rejects_a_missing_event_column(
    sample_events: pd.DataFrame, sample_applications: pd.DataFrame, column: str
) -> None:
    with pytest.raises(ExportError, match=column):
        compute_stage_flows(sample_events.drop(columns=[column]), sample_applications)


def test_compute_stage_flows_rejects_a_missing_application_column(
    sample_events: pd.DataFrame, sample_applications: pd.DataFrame
) -> None:
    with pytest.raises(ExportError, match="status"):
        compute_stage_flows(sample_events, sample_applications.drop(columns=["status"]))


def test_wire_format_columns_are_the_frozen_tuples(
    sample_applications: pd.DataFrame, sample_events: pd.DataFrame
) -> None:
    """I10: the frames M5 reads are exactly the tuples M0 froze, in order."""
    assert tuple(sample_applications.columns) == EXPORT_COLUMNS
    assert tuple(sample_events.columns) == EVENT_COLUMNS


# --- funnel_sankey ---------------------------------------------------------------------


def test_funnel_sankey_orders_nodes_by_pipeline_stage(
    sample_events: pd.DataFrame, sample_applications: pd.DataFrame
) -> None:
    flows = compute_stage_flows(sample_events, sample_applications)
    figure = funnel_sankey(flows)
    labels = list(figure.data[0].node.label)
    assert labels == [stage for stage in STAGE_ORDER if stage in labels]
    terminal_positions = [labels.index(stage) for stage in labels if stage in TERMINAL_STAGES]
    other_positions = [labels.index(stage) for stage in labels if stage not in TERMINAL_STAGES]
    assert min(terminal_positions) > max(other_positions)


def test_funnel_sankey_maps_link_indices_to_node_labels() -> None:
    flows = [
        StageFlow(source="Applied", target="Interview", count=3),
        StageFlow(source="Interview", target="Rejected", count=2),
    ]
    link = funnel_sankey(flows).data[0].link
    labels = list(funnel_sankey(flows).data[0].node.label)
    assert [labels[index] for index in link.source] == ["Applied", "Interview"]
    assert [labels[index] for index in link.target] == ["Interview", "Rejected"]
    assert list(link.value) == [3, 2]


def test_funnel_sankey_tolerates_an_unrecognised_stage_label() -> None:
    flows = [StageFlow(source="Applied", target="Mystery", count=1)]
    labels = list(funnel_sankey(flows).data[0].node.label)
    assert labels == ["Applied", "Mystery"]


def test_funnel_sankey_with_no_flows_renders_placeholder() -> None:
    figure = funnel_sankey([])
    assert len(figure.data) == 0
    assert "No stage transitions" in _annotation_texts(figure)[0]


# --- cross-cutting ---------------------------------------------------------------------


def test_no_chart_function_writes_a_file(
    tmp_path: Path, sample_applications: pd.DataFrame, sample_events: pd.DataFrame
) -> None:
    """Charts build figures; only dashboard.py is allowed to touch the filesystem."""
    before = sorted(tmp_path.rglob("*"))
    status_bar_chart(sample_applications)
    applications_over_time(sample_applications)
    top_companies_bar(sample_applications)
    response_time_histogram(sample_applications)
    funnel_sankey(compute_stage_flows(sample_events, sample_applications))
    assert sorted(tmp_path.rglob("*")) == before


def test_figures_are_deterministic(
    sample_applications: pd.DataFrame, sample_events: pd.DataFrame
) -> None:
    """Same frame in, byte-identical figure JSON out."""
    flows = compute_stage_flows(sample_events, sample_applications)
    for build in (
        lambda: status_bar_chart(sample_applications),
        lambda: applications_over_time(sample_applications),
        lambda: top_companies_bar(sample_applications),
        lambda: response_time_histogram(sample_applications),
        lambda: funnel_sankey(flows),
    ):
        assert build().to_json() == build().to_json()
