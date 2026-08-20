"""Unit tests for jobtrack.viz.charts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import pytest

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.models import ApplicationStatus, EventType, StageFlow
from jobtrack.viz.charts import (
    STATUS_COLORS,
    STATUS_PIPELINE_ORDER,
    applications_over_time,
    compute_stage_flows,
    funnel_sankey,
    response_time_histogram,
    status_bar_chart,
    top_companies_bar,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _applications_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an applications frame shaped exactly like M4's `build_dataframe` output."""
    if not rows:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS)


def _events_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an events frame shaped exactly like M4's `build_events_dataframe` output."""
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _application_row(**overrides: Any) -> dict[str, Any]:
    """One EXPORT_COLUMNS-shaped row with sensible defaults."""
    row: dict[str, Any] = {
        "application_id": "app-1",
        "company": "Acme Robotics",
        "role": "Software Engineer",
        "location": "Remote",
        "ats": "greenhouse",
        "status": ApplicationStatus.APPLIED,
        "applied_at": _T0,
        "last_event_at": _T0,
        "last_event_type": EventType.APPLICATION_RECEIVED,
        "event_count": 1,
        "days_to_first_response": None,
        "days_since_last_event": 1,
        "needs_review": False,
    }
    row.update(overrides)
    return row


def _populated_applications_df() -> pd.DataFrame:
    return _applications_df(
        [
            _application_row(
                application_id="app-1",
                company="Acme Robotics",
                status=ApplicationStatus.INTERVIEWING,
                applied_at=_T0,
                days_to_first_response=3,
            ),
            _application_row(
                application_id="app-2",
                company="Acme Robotics",
                status=ApplicationStatus.REJECTED,
                applied_at=_T0 + timedelta(days=8),
                days_to_first_response=10,
            ),
            _application_row(
                application_id="app-3",
                company="Globex",
                status=ApplicationStatus.APPLIED,
                applied_at=_T0 + timedelta(days=15),
                days_to_first_response=None,
            ),
        ]
    )


class TestStatusBarChart:
    def test_populated_returns_figure_ordered_by_pipeline_stage(self) -> None:
        df = _populated_applications_df()
        fig = status_bar_chart(df)

        assert isinstance(fig, go.Figure)
        (bar,) = fig.data
        assert list(bar.x) == [str(status) for status in STATUS_PIPELINE_ORDER]
        # two INTERVIEWING+REJECTED go into their own slots; APPLIED gets the third
        applied_idx = list(bar.x).index(str(ApplicationStatus.APPLIED))
        assert bar.y[applied_idx] == 1

    def test_colors_follow_fixed_pipeline_order(self) -> None:
        df = _populated_applications_df()
        fig = status_bar_chart(df)
        (bar,) = fig.data
        assert list(bar.marker.color) == [STATUS_COLORS[status] for status in STATUS_PIPELINE_ORDER]

    def test_empty_dataframe_does_not_raise(self) -> None:
        fig = status_bar_chart(_applications_df([]))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) == 1

    def test_totally_empty_dataframe_does_not_raise(self) -> None:
        fig = status_bar_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)


class TestApplicationsOverTime:
    def test_populated_returns_figure(self) -> None:
        df = _populated_applications_df()
        fig = applications_over_time(df, freq="W")
        assert isinstance(fig, go.Figure)
        (trace,) = fig.data
        assert sum(trace.y) == 3

    def test_empty_dataframe_does_not_raise(self) -> None:
        fig = applications_over_time(_applications_df([]))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_totally_empty_dataframe_does_not_raise(self) -> None:
        fig = applications_over_time(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_unparseable_applied_at_does_not_raise(self) -> None:
        df = _applications_df([_application_row(application_id="app-1", applied_at="not-a-date")])
        fig = applications_over_time(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestTopCompaniesBar:
    def test_populated_orders_by_total_descending(self) -> None:
        df = _populated_applications_df()
        fig = top_companies_bar(df, top_n=20)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        # Acme Robotics (2 applications) should out-total Globex (1)
        categoryarray = list(fig.layout.yaxis.categoryarray)
        assert categoryarray.index("Acme Robotics") > categoryarray.index("Globex")

    def test_stacked_by_status(self) -> None:
        df = _populated_applications_df()
        fig = top_companies_bar(df, top_n=20)
        trace_names = {trace.name for trace in fig.data}
        assert str(ApplicationStatus.INTERVIEWING).replace("_", " ").title() in trace_names
        assert str(ApplicationStatus.REJECTED).replace("_", " ").title() in trace_names

    def test_top_n_zero_does_not_raise(self) -> None:
        fig = top_companies_bar(_populated_applications_df(), top_n=0)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_empty_dataframe_does_not_raise(self) -> None:
        fig = top_companies_bar(_applications_df([]))
        assert isinstance(fig, go.Figure)

    def test_totally_empty_dataframe_does_not_raise(self) -> None:
        fig = top_companies_bar(pd.DataFrame())
        assert isinstance(fig, go.Figure)


class TestResponseTimeHistogram:
    def test_populated_excludes_nulls(self) -> None:
        df = _populated_applications_df()
        fig = response_time_histogram(df)
        assert isinstance(fig, go.Figure)
        (trace,) = fig.data
        assert sorted(trace.x) == [3, 10]

    def test_all_null_does_not_raise(self) -> None:
        df = _applications_df(
            [_application_row(application_id="app-1", days_to_first_response=None)]
        )
        fig = response_time_histogram(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_empty_dataframe_does_not_raise(self) -> None:
        fig = response_time_histogram(_applications_df([]))
        assert isinstance(fig, go.Figure)

    def test_totally_empty_dataframe_does_not_raise(self) -> None:
        fig = response_time_histogram(pd.DataFrame())
        assert isinstance(fig, go.Figure)


class TestComputeStageFlows:
    def test_hand_constructed_history_including_ghosted(self) -> None:
        applications = _applications_df(
            [
                _application_row(application_id="app-1", status=ApplicationStatus.OFFER),
                _application_row(application_id="app-2", status=ApplicationStatus.REJECTED),
                _application_row(application_id="app-3", status=ApplicationStatus.GHOSTED),
                _application_row(application_id="app-4", status=ApplicationStatus.GHOSTED),
                # negative case: recruiter outreach alone must not fabricate a stage
                # transition, and a non-ghosted status must not add a terminal link
                _application_row(application_id="app-5", status=ApplicationStatus.APPLIED),
            ]
        )
        events = _events_df(
            [
                # app-1: full happy path through to an offer
                {
                    "application_id": "app-1",
                    "message_id": "m1",
                    "event_type": EventType.APPLICATION_RECEIVED,
                    "occurred_at": _T0,
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Thanks for applying",
                },
                {
                    "application_id": "app-1",
                    "message_id": "m2",
                    "event_type": EventType.ASSESSMENT,
                    "occurred_at": _T0 + timedelta(days=1),
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Coding challenge",
                },
                {
                    "application_id": "app-1",
                    "message_id": "m3",
                    "event_type": EventType.INTERVIEW,
                    "occurred_at": _T0 + timedelta(days=2),
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Interview scheduling",
                },
                {
                    "application_id": "app-1",
                    "message_id": "m4",
                    "event_type": EventType.OFFER,
                    "occurred_at": _T0 + timedelta(days=3),
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Offer",
                },
                # app-2: skips assessment, interviewed then rejected
                {
                    "application_id": "app-2",
                    "message_id": "m5",
                    "event_type": EventType.APPLICATION_RECEIVED,
                    "occurred_at": _T0,
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Thanks for applying",
                },
                {
                    "application_id": "app-2",
                    "message_id": "m6",
                    "event_type": EventType.INTERVIEW,
                    "occurred_at": _T0 + timedelta(days=1),
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Interview scheduling",
                },
                {
                    "application_id": "app-2",
                    "message_id": "m7",
                    "event_type": EventType.REJECTION,
                    "occurred_at": _T0 + timedelta(days=2),
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Not moving forward",
                },
                # app-3: stalled after assessment -> ghosted
                {
                    "application_id": "app-3",
                    "message_id": "m8",
                    "event_type": EventType.APPLICATION_RECEIVED,
                    "occurred_at": _T0,
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Thanks for applying",
                },
                {
                    "application_id": "app-3",
                    "message_id": "m9",
                    "event_type": EventType.ASSESSMENT,
                    "occurred_at": _T0 + timedelta(days=1),
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Take-home",
                },
                # app-4: no events recorded at all -> ghosted straight from Applied
                # app-5: recruiter outreach only, not ghosted -> no transition at all
                {
                    "application_id": "app-5",
                    "message_id": "m10",
                    "event_type": EventType.RECRUITER_OUTREACH,
                    "occurred_at": _T0,
                    "confidence": 0.9,
                    "needs_review": False,
                    "subject": "Are you open to new roles?",
                },
            ]
        )

        flows = compute_stage_flows(events, applications)

        assert flows == [
            StageFlow(source="Applied", target="Assessment", count=2),
            StageFlow(source="Assessment", target="Interview", count=1),
            StageFlow(source="Interview", target="Offer", count=1),
            StageFlow(source="Applied", target="Interview", count=1),
            StageFlow(source="Interview", target="Rejected", count=1),
            StageFlow(source="Assessment", target="Ghosted", count=1),
            StageFlow(source="Applied", target="Ghosted", count=1),
        ]

    def test_deterministic_on_repeat_calls(self) -> None:
        applications = _applications_df(
            [_application_row(application_id="app-1", status=ApplicationStatus.GHOSTED)]
        )
        events = _events_df([])
        first = compute_stage_flows(events, applications)
        second = compute_stage_flows(events, applications)
        assert first == second

    def test_empty_applications_returns_empty_list(self) -> None:
        assert compute_stage_flows(_events_df([]), _applications_df([])) == []

    def test_totally_empty_dataframes_do_not_raise(self) -> None:
        assert compute_stage_flows(pd.DataFrame(), pd.DataFrame()) == []


class TestFunnelSankey:
    def test_populated_flows_return_figure_with_terminal_nodes_rightmost(self) -> None:
        flows = [
            StageFlow(source="Applied", target="Assessment", count=3),
            StageFlow(source="Assessment", target="Interview", count=2),
            StageFlow(source="Interview", target="Offer", count=1),
            StageFlow(source="Assessment", target="Ghosted", count=1),
        ]
        fig = funnel_sankey(flows)
        assert isinstance(fig, go.Figure)
        (sankey,) = fig.data
        labels = list(sankey.node.label)
        assert labels.index("Applied") < labels.index("Assessment")
        assert labels.index("Assessment") < labels.index("Interview")
        assert labels.index("Interview") < labels.index("Offer")
        # terminal nodes sit to the right of every non-terminal node they connect from
        assert labels.index("Offer") > labels.index("Interview")
        assert labels.index("Ghosted") > labels.index("Assessment")

    def test_empty_flows_does_not_raise(self) -> None:
        fig = funnel_sankey([])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) == 1


@pytest.mark.parametrize("freq", ["D", "W", "ME"])
def test_applications_over_time_accepts_various_frequencies(freq: str) -> None:
    fig = applications_over_time(_populated_applications_df(), freq=freq)
    assert isinstance(fig, go.Figure)
