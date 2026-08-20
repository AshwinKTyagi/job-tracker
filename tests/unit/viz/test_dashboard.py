"""Unit tests for jobtrack.viz.dashboard."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.errors import ExportError
from jobtrack.models import ApplicationStatus, EventType
from jobtrack.viz.dashboard import build_dashboard

_T0 = datetime(2026, 1, 1, tzinfo=UTC)

_EXTERNAL_SCRIPT_SRC = re.compile(r'<script[^>]+src=["\']https?://', re.IGNORECASE)
"""Matches a <script src="http(s)://..."> tag — the only thing that would actually
reach a network at render time. Plotly's own inlined bundle mentions the literal
substring "cdn" internally (a dead default-config value), so grepping for that
substring alone is not a reliable inlining check; this is."""


def _application_row(**overrides: Any) -> dict[str, Any]:
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
    return pd.DataFrame(
        [
            _application_row(
                application_id="app-1",
                status=ApplicationStatus.INTERVIEWING,
                applied_at=_T0,
                days_to_first_response=3,
            ),
            _application_row(
                application_id="app-2",
                company="Globex",
                status=ApplicationStatus.REJECTED,
                applied_at=_T0 + timedelta(days=5),
                days_to_first_response=None,
            ),
        ],
        columns=EXPORT_COLUMNS,
    )


def _populated_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
                "event_type": EventType.INTERVIEW,
                "occurred_at": _T0 + timedelta(days=1),
                "confidence": 0.9,
                "needs_review": False,
                "subject": "Interview scheduling",
            },
        ],
        columns=EVENT_COLUMNS,
    )


class TestBuildDashboard:
    def test_writes_a_real_file_with_inlined_plotly(self, tmp_path: Path) -> None:
        out = tmp_path / "dashboard.html"
        result = build_dashboard(_populated_applications_df(), _populated_events_df(), out)

        assert result == out.resolve()
        assert out.exists()
        html = out.read_text(encoding="utf-8")

        # No <script src="http(s)://..."> — nothing is fetched from a network at render
        # time, CDN or otherwise.
        assert not _EXTERNAL_SCRIPT_SRC.search(html)
        # The plotly bundle itself, inlined and large enough to be the real thing.
        assert "Plotly.newPlot" in html
        assert len(html) > 500_000

    def test_summary_header_reflects_the_data(self, tmp_path: Path) -> None:
        out = tmp_path / "dashboard.html"
        build_dashboard(_populated_applications_df(), _populated_events_df(), out)
        html = out.read_text(encoding="utf-8")

        assert ">2<" in html  # total applications tile
        assert "50%" in html  # 1 of 2 responded

    def test_custom_title_appears_in_output(self, tmp_path: Path) -> None:
        out = tmp_path / "dashboard.html"
        build_dashboard(
            _populated_applications_df(), _populated_events_df(), out, title="My Search"
        )
        html = out.read_text(encoding="utf-8")
        assert "My Search" in html
        assert "<title>My Search</title>" in html

    def test_empty_dataframes_render_placeholder_not_traceback(self, tmp_path: Path) -> None:
        out = tmp_path / "dashboard.html"
        result = build_dashboard(
            pd.DataFrame(columns=EXPORT_COLUMNS), pd.DataFrame(columns=EVENT_COLUMNS), out
        )

        assert result.exists()
        html = result.read_text(encoding="utf-8")
        assert "No applications recorded yet" in html
        assert not _EXTERNAL_SCRIPT_SRC.search(html)

    def test_totally_empty_dataframes_do_not_raise(self, tmp_path: Path) -> None:
        out = tmp_path / "dashboard.html"
        result = build_dashboard(pd.DataFrame(), pd.DataFrame(), out)
        assert result.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "dashboard.html"
        result = build_dashboard(_populated_applications_df(), _populated_events_df(), out)
        assert result.exists()

    def test_unwritable_path_raises_export_error(self, tmp_path: Path) -> None:
        # A file where a directory needs to be is not writable-through.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        out = blocker / "dashboard.html"

        with pytest.raises(ExportError):
            build_dashboard(_populated_applications_df(), _populated_events_df(), out)
