"""Unit tests for the self-contained HTML dashboard.

The load-bearing assertion here is that the generated page references nothing external:
plotly.js must be inlined, so the file opens from disk with the network unplugged. Tests
run under ``--disable-socket``, which means a CDN reference would be a silent landmine at
runtime rather than a loud failure here — hence the explicit check on the markup.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from jobtrack.errors import ExportError
from jobtrack.models import ApplicationStatus, EventType
from jobtrack.viz.charts import (
    applications_over_time,
    compute_stage_flows,
    funnel_sankey,
    response_time_histogram,
    status_bar_chart,
    top_companies_bar,
)
from jobtrack.viz.dashboard import build_dashboard

from .conftest import (
    BASE_TIME,
    applications_frame,
    events_frame,
    linear_pipeline,
    make_application,
)

EXTERNAL_SCRIPT = re.compile(r"<script[^>]*\bsrc\s*=", re.IGNORECASE)
"""Any <script src=...> at all: a self-contained page has none."""

EXTERNAL_LINK = re.compile(r"<link[^>]*\bhref\s*=", re.IGNORECASE)
"""Any <link href=...>: stylesheets and fonts must be inlined too."""

CHART_DIV_IDS = ("chart-status", "chart-funnel", "chart-timeline", "chart-response")


def _populated_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three applications with a full spread of statuses and response times."""
    applications = applications_frame(
        [
            make_application(
                application_id="app-1",
                company="Acme Robotics",
                status=ApplicationStatus.REJECTED,
                applied_at=BASE_TIME,
                last_event_at=BASE_TIME + timedelta(days=10),
                last_event_type=EventType.REJECTION,
                event_count=3,
                days_to_first_response=4,
                days_since_last_event=2,
            ),
            make_application(
                application_id="app-2",
                company="Globex",
                status=ApplicationStatus.INTERVIEWING,
                applied_at=BASE_TIME + timedelta(days=2),
                last_event_at=BASE_TIME + timedelta(days=8),
                last_event_type=EventType.INTERVIEW,
                event_count=2,
                days_to_first_response=6,
                days_since_last_event=3,
                needs_review=True,
            ),
            make_application(
                application_id="app-3",
                company="Initech",
                status=ApplicationStatus.OFFER,
                applied_at=BASE_TIME + timedelta(days=20),
                last_event_at=BASE_TIME + timedelta(days=35),
                last_event_type=EventType.OFFER,
                event_count=4,
                days_to_first_response=8,
                days_since_last_event=1,
            ),
        ]
    )
    events = events_frame(
        [
            *linear_pipeline(
                "app-1",
                [EventType.APPLICATION_RECEIVED, EventType.INTERVIEW, EventType.REJECTION],
            ),
            *linear_pipeline("app-2", [EventType.APPLICATION_RECEIVED, EventType.INTERVIEW]),
            *linear_pipeline(
                "app-3",
                [EventType.APPLICATION_RECEIVED, EventType.ASSESSMENT, EventType.OFFER],
            ),
        ]
    )
    return applications, events


@pytest.fixture(scope="module")
def dashboard_html(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A dashboard rendered once for the whole module — the page is several megabytes."""
    applications, events = _populated_frames()
    destination = tmp_path_factory.mktemp("dashboard") / "dashboard.html"
    written = build_dashboard(applications, events, destination)
    return written.read_text(encoding="utf-8")


# --- offline self-containment ----------------------------------------------------------


def test_dashboard_has_no_external_script_src(dashboard_html: str) -> None:
    """plotly.js is inlined; the file must fetch nothing when opened offline."""
    assert EXTERNAL_SCRIPT.search(dashboard_html) is None


def test_dashboard_has_no_external_stylesheet_link(dashboard_html: str) -> None:
    assert EXTERNAL_LINK.search(dashboard_html) is None


def test_dashboard_inlines_the_plotly_bundle(dashboard_html: str) -> None:
    assert "Plotly.newPlot" in dashboard_html
    # The bundle itself is megabytes; a CDN stub would be a few hundred bytes.
    assert len(dashboard_html) > 1_000_000


def test_dashboard_renders_no_geo_trace() -> None:
    """The one cdn.plot.ly string in the bundle is plotly's topojsonURL default, which only
    geo traces ever fetch. Rendering none of them keeps the page genuinely offline."""
    applications, events = _populated_frames()
    flows = compute_stage_flows(events, applications)
    figures = [
        status_bar_chart(applications),
        funnel_sankey(flows),
        applications_over_time(applications),
        response_time_histogram(applications),
        top_companies_bar(applications),
    ]
    rendered = {trace.type for figure in figures for trace in figure.data}
    assert rendered <= {"bar", "histogram", "sankey"}


# --- structure -------------------------------------------------------------------------


def test_dashboard_contains_every_chart(dashboard_html: str) -> None:
    for div_id in (*CHART_DIV_IDS, "chart-companies"):
        assert div_id in dashboard_html


def test_dashboard_is_a_complete_html_document(dashboard_html: str) -> None:
    assert dashboard_html.startswith("<!doctype html>")
    assert dashboard_html.rstrip().endswith("</html>")


def test_dashboard_summary_header_reports_the_headline_numbers(dashboard_html: str) -> None:
    header = dashboard_html.split('<section class="card"', 1)[0]
    assert "response rate" in header
    assert "100%" in header  # 3 of 3 applications have a first response
    assert "median days to reply" in header
    assert ">6<" in header  # median of 4, 6, 8
    assert "3 applications tracked" in header


def test_dashboard_uses_the_default_title(dashboard_html: str) -> None:
    assert "<title>Job Application Tracker</title>" in dashboard_html


# --- writing ---------------------------------------------------------------------------


def test_build_dashboard_returns_the_resolved_path(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    destination = tmp_path / "sub" / ".." / "sub" / "dash.html"
    written = build_dashboard(applications, events, destination)
    assert written == destination.resolve()
    assert written.is_file()


def test_build_dashboard_creates_missing_parent_directories(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    destination = tmp_path / "deep" / "nested" / "dash.html"
    build_dashboard(applications, events, destination)
    assert destination.is_file()


def test_build_dashboard_writes_nothing_outside_the_given_path(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    destination = tmp_path / "dash.html"
    build_dashboard(applications, events, destination)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["dash.html"]


def test_build_dashboard_raises_export_error_on_an_unwritable_path(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    blocked = tmp_path / "dash.html"
    blocked.mkdir()  # a directory where the file should go
    with pytest.raises(ExportError, match="could not write dashboard"):
        build_dashboard(applications, events, blocked)


def test_build_dashboard_is_deterministic(tmp_path: Path) -> None:
    """No clock is read, so the same frames render byte-identical HTML."""
    applications, events = _populated_frames()
    first = build_dashboard(applications, events, tmp_path / "one.html")
    second = build_dashboard(applications, events, tmp_path / "two.html")
    assert first.read_bytes() == second.read_bytes()


# --- titles and escaping ---------------------------------------------------------------


def test_build_dashboard_escapes_the_title(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    destination = build_dashboard(
        applications, events, tmp_path / "dash.html", title="<script>alert(1)</script>"
    )
    html = destination.read_text(encoding="utf-8")
    assert "<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>" in html
    assert EXTERNAL_SCRIPT.search(html) is None


# --- empty input -----------------------------------------------------------------------


def test_build_dashboard_renders_a_placeholder_for_empty_frames(
    tmp_path: Path, empty_applications: pd.DataFrame, empty_events: pd.DataFrame
) -> None:
    destination = build_dashboard(empty_applications, empty_events, tmp_path / "empty.html")
    html = destination.read_text(encoding="utf-8")
    assert "No applications yet" in html
    assert "jobtrack sync" in html
    assert "0 applications tracked" in html
    for div_id in CHART_DIV_IDS:
        assert div_id not in html


def test_empty_dashboard_is_still_self_contained(
    tmp_path: Path, empty_applications: pd.DataFrame, empty_events: pd.DataFrame
) -> None:
    destination = build_dashboard(empty_applications, empty_events, tmp_path / "empty.html")
    html = destination.read_text(encoding="utf-8")
    assert EXTERNAL_SCRIPT.search(html) is None
    assert EXTERNAL_LINK.search(html) is None


def test_build_dashboard_handles_a_single_application(tmp_path: Path) -> None:
    """The subtitle pluralization and the median tile both have a one-row edge case."""
    applications = applications_frame([make_application(days_to_first_response=3)])
    events = events_frame(linear_pipeline("app-0001", [EventType.APPLICATION_RECEIVED]))
    destination = build_dashboard(applications, events, tmp_path / "one.html")
    html = destination.read_text(encoding="utf-8")
    assert "1 application tracked" in html


def test_build_dashboard_handles_applications_with_no_responses(tmp_path: Path) -> None:
    applications = applications_frame([make_application(days_to_first_response=None)])
    events = events_frame(linear_pipeline("app-0001", [EventType.APPLICATION_RECEIVED]))
    destination = build_dashboard(applications, events, tmp_path / "none.html")
    html = destination.read_text(encoding="utf-8")
    assert "0%" in html
    assert "—" in html  # median tile has nothing to report


# --- wire format -----------------------------------------------------------------------


def test_build_dashboard_rejects_a_missing_application_column(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    with pytest.raises(ExportError, match="applications frame is missing"):
        build_dashboard(applications.drop(columns=["status"]), events, tmp_path / "d.html")


def test_build_dashboard_rejects_a_missing_event_column(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    with pytest.raises(ExportError, match="events frame is missing"):
        build_dashboard(applications, events.drop(columns=["occurred_at"]), tmp_path / "d.html")


def test_build_dashboard_does_not_mutate_its_inputs(tmp_path: Path) -> None:
    applications, events = _populated_frames()
    applications_before = applications.copy()
    events_before = events.copy()
    build_dashboard(applications, events, tmp_path / "dash.html")
    pd.testing.assert_frame_equal(applications, applications_before)
    pd.testing.assert_frame_equal(events, events_before)
