"""Unit tests for M4 export.

The frame shape is the M4<->M5 wire format (I10), so the column assertions here are
deliberately exact: a tuple comparison, not a set comparison.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.errors import ExportError, JobTrackError
from jobtrack.export.tabular import (
    APPLICATIONS_SHEET,
    APPLICATION_DATETIME_COLUMNS,
    APPLICATION_DTYPES,
    EVENTS_SHEET,
    EVENT_DATETIME_COLUMNS,
    EVENT_DTYPES,
    build_dataframe,
    build_events_dataframe,
    write_csv,
    write_xlsx,
)
from jobtrack.models import ApplicationRow, ApplicationStatus, EventRow, EventType


def make_application(now: datetime, **overrides: Any) -> ApplicationRow:
    """Build an ApplicationRow with sensible defaults, clock injected."""
    defaults: dict[str, Any] = {
        "application_id": "app-0001",
        "company": "Acme Robotics, Inc.",
        "company_key": "acme robotics",
        "role": "Software Engineer",
        "location": "Remote — US",
        "ats": "greenhouse",
        "status": ApplicationStatus.INTERVIEWING,
        "applied_at": now - timedelta(days=10),
        "last_event_at": now - timedelta(days=2),
        "last_event_type": EventType.INTERVIEW,
        "event_count": 3,
        "days_to_first_response": 4,
        "days_since_last_event": 2,
        "needs_review": False,
        "source_thread_ids": ["thread-0001"],
    }
    defaults.update(overrides)
    return ApplicationRow.model_validate(defaults)


def make_event(now: datetime, **overrides: Any) -> EventRow:
    """Build an EventRow with sensible defaults, clock injected."""
    defaults: dict[str, Any] = {
        "event_id": 1,
        "application_id": "app-0001",
        "message_id": "msg-0001",
        "event_type": EventType.APPLICATION_RECEIVED,
        "occurred_at": now - timedelta(days=10),
        "confidence": 0.92,
        "needs_review": False,
        "is_overridden": False,
        "subject": "Thanks for applying to Acme Robotics",
        "from_email": "no-reply@greenhouse.io",
    }
    defaults.update(overrides)
    return EventRow.model_validate(defaults)


def sample_applications(now: datetime) -> list[ApplicationRow]:
    """Two rows: one fully populated, one with every nullable field missing."""
    return [
        make_application(now),
        make_application(
            now,
            application_id="app-0002",
            company="Globex",
            company_key="globex",
            role=None,
            location=None,
            ats=None,
            status=ApplicationStatus.REJECTED,
            last_event_type=EventType.REJECTION,
            event_count=2,
            days_to_first_response=None,
            days_since_last_event=40,
            needs_review=True,
            source_thread_ids=["thread-0002", "thread-0003"],
        ),
    ]


def sample_events(now: datetime) -> list[EventRow]:
    """Two events, one of them unlinked (UNKNOWN)."""
    return [
        make_event(now),
        make_event(
            now,
            event_id=2,
            application_id=None,
            message_id="msg-0002",
            event_type=EventType.UNKNOWN,
            occurred_at=now - timedelta(days=1),
            confidence=0.0,
            needs_review=True,
            subject="Weekly newsletter",
        ),
    ]


# --- I10: the frozen wire format -------------------------------------------------------


def test_columns_are_export_columns_exactly(frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    assert tuple(df.columns) == EXPORT_COLUMNS


def test_empty_input_keeps_export_columns_and_dtypes(frozen_now: datetime) -> None:
    empty = build_dataframe([])
    full = build_dataframe(sample_applications(frozen_now))
    assert tuple(empty.columns) == EXPORT_COLUMNS
    assert len(empty) == 0
    assert empty.dtypes.to_dict() == full.dtypes.to_dict()


def test_event_columns_are_event_columns_exactly(frozen_now: datetime) -> None:
    df = build_events_dataframe(sample_events(frozen_now))
    assert tuple(df.columns) == EVENT_COLUMNS


def test_empty_events_keeps_columns_and_dtypes(frozen_now: datetime) -> None:
    empty = build_events_dataframe([])
    full = build_events_dataframe(sample_events(frozen_now))
    assert tuple(empty.columns) == EVENT_COLUMNS
    assert len(empty) == 0
    assert empty.dtypes.to_dict() == full.dtypes.to_dict()


def test_dtype_tables_cover_every_column() -> None:
    assert set(APPLICATION_DTYPES) | set(APPLICATION_DATETIME_COLUMNS) == set(EXPORT_COLUMNS)
    assert set(EVENT_DTYPES) | set(EVENT_DATETIME_COLUMNS) == set(EVENT_COLUMNS)


# --- values and dtypes -----------------------------------------------------------------


def test_scalar_dtypes(frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    assert df["event_count"].dtype == "int64"
    assert df["days_since_last_event"].dtype == "int64"
    assert df["days_to_first_response"].dtype == "Int64"
    assert df["needs_review"].dtype == "bool"


def test_datetime_columns_are_tz_aware_utc(frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    for column in ("applied_at", "last_event_at"):
        dtype = df[column].dtype
        assert isinstance(dtype, pd.DatetimeTZDtype)
        assert str(dtype.tz) == "UTC"
    assert df.loc[0, "applied_at"] == pd.Timestamp(frozen_now - timedelta(days=10))


def test_event_occurred_at_is_tz_aware_utc(frozen_now: datetime) -> None:
    df = build_events_dataframe(sample_events(frozen_now))
    dtype = df["occurred_at"].dtype
    assert isinstance(dtype, pd.DatetimeTZDtype)
    assert str(dtype.tz) == "UTC"
    assert df["confidence"].dtype == "float64"


def test_enums_become_plain_strings(frozen_now: datetime) -> None:
    """StrEnum members must not survive into the frame — M5 groups on these labels."""
    df = build_dataframe(sample_applications(frozen_now))
    events = build_events_dataframe(sample_events(frozen_now))
    status = df.loc[0, "status"]
    assert type(status) is str
    assert status == "interviewing"
    assert type(df.loc[0, "last_event_type"]) is str
    assert df.loc[1, "last_event_type"] == "rejection"
    assert type(events.loc[1, "event_type"]) is str
    assert events.loc[1, "event_type"] == "unknown"


def test_nullable_fields_are_missing_not_none_strings(frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    assert df["role"].isna().tolist() == [False, True]
    assert df["location"].isna().tolist() == [False, True]
    assert df["ats"].isna().tolist() == [False, True]
    assert df["days_to_first_response"].isna().tolist() == [False, True]
    assert df.loc[0, "days_to_first_response"] == 4


def test_unlinked_event_has_missing_application_id(frozen_now: datetime) -> None:
    df = build_events_dataframe(sample_events(frozen_now))
    assert df["application_id"].isna().tolist() == [False, True]


def test_company_key_is_not_exported(frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    assert "company_key" not in df.columns
    assert "source_thread_ids" not in df.columns


def test_row_order_is_preserved(frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    assert df["application_id"].tolist() == ["app-0001", "app-0002"]
    assert df.index.tolist() == [0, 1]


def test_build_dataframe_is_deterministic(frozen_now: datetime) -> None:
    rows = sample_applications(frozen_now)
    assert_frame_equal(build_dataframe(rows), build_dataframe(rows))
    assert_frame_equal(build_events_dataframe(sample_events(frozen_now)), build_events_dataframe(
        sample_events(frozen_now)
    ))


# --- CSV -------------------------------------------------------------------------------


def test_write_csv_header_and_iso_dates(tmp_path: Path, frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    out = write_csv(df, tmp_path / "apps.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(EXPORT_COLUMNS)
    assert "2026-08-08T12:00:00Z" in lines[1]
    assert out == (tmp_path / "apps.csv").resolve()


def test_write_csv_leaves_missing_values_empty(tmp_path: Path, frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    out = write_csv(df, tmp_path / "apps.csv")
    reread = pd.read_csv(out, dtype="str", keep_default_na=True)
    assert reread.loc[1, "role"] != reread.loc[1, "role"]  # NaN, not the text "None"
    assert "None" not in out.read_text(encoding="utf-8")


def test_write_csv_round_trips_values(tmp_path: Path, frozen_now: datetime) -> None:
    rows = sample_applications(frozen_now)
    out = write_csv(build_dataframe(rows), tmp_path / "apps.csv")
    reread = pd.read_csv(out, dtype="str")
    assert tuple(reread.columns) == EXPORT_COLUMNS
    assert reread.loc[0, "company"] == rows[0].company
    assert reread.loc[0, "location"] == rows[0].location  # non-ASCII survives UTF-8
    assert reread.loc[0, "status"] == "interviewing"


def test_write_csv_creates_parent_directories(tmp_path: Path, frozen_now: datetime) -> None:
    out = write_csv(build_dataframe([]), tmp_path / "nested" / "deeper" / "apps.csv")
    assert out.is_file()


def test_write_csv_is_deterministic(tmp_path: Path, frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    first = write_csv(df, tmp_path / "one.csv").read_bytes()
    second = write_csv(df, tmp_path / "two.csv").read_bytes()
    assert first == second


def test_write_csv_events(tmp_path: Path, frozen_now: datetime) -> None:
    out = write_csv(build_events_dataframe(sample_events(frozen_now)), tmp_path / "events.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(EVENT_COLUMNS)


def test_write_csv_unwritable_path_raises_export_error(
    tmp_path: Path, frozen_now: datetime
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ExportError) as excinfo:
        write_csv(build_dataframe([]), blocker / "apps.csv")
    assert isinstance(excinfo.value, JobTrackError)


def test_write_csv_to_a_directory_raises_export_error(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        write_csv(build_dataframe([]), tmp_path)


# --- XLSX ------------------------------------------------------------------------------


def read_back(path: Path, sheet: str) -> pd.DataFrame:
    """Read a written sheet back into the same shape ``build_dataframe`` produces."""
    raw = pd.read_excel(path, sheet_name=sheet, dtype="str")
    datetime_columns = (
        APPLICATION_DATETIME_COLUMNS if sheet == APPLICATIONS_SHEET else EVENT_DATETIME_COLUMNS
    )
    dtypes = APPLICATION_DTYPES if sheet == APPLICATIONS_SHEET else EVENT_DTYPES
    out = raw.astype(dict(dtypes))
    for column in datetime_columns:
        out[column] = pd.to_datetime(raw[column], utc=True, format="%Y-%m-%dT%H:%M:%SZ").astype(
            "datetime64[ns, UTC]"
        )
    return out[list(raw.columns)]


def test_write_xlsx_round_trips(tmp_path: Path, frozen_now: datetime) -> None:
    df = build_dataframe(sample_applications(frozen_now))
    out = write_xlsx(df, tmp_path / "apps.xlsx")
    assert_frame_equal(read_back(out, APPLICATIONS_SHEET), df)


def test_write_xlsx_events_sheet_round_trips(tmp_path: Path, frozen_now: datetime) -> None:
    apps = build_dataframe(sample_applications(frozen_now))
    events = build_events_dataframe(sample_events(frozen_now))
    out = write_xlsx(apps, tmp_path / "apps.xlsx", events=events)
    with pd.ExcelFile(out) as handle:
        assert handle.sheet_names == [APPLICATIONS_SHEET, EVENTS_SHEET]
    assert_frame_equal(read_back(out, EVENTS_SHEET), events)


def test_write_xlsx_without_events_has_one_sheet(tmp_path: Path, frozen_now: datetime) -> None:
    out = write_xlsx(build_dataframe(sample_applications(frozen_now)), tmp_path / "apps.xlsx")
    with pd.ExcelFile(out) as handle:
        assert handle.sheet_names == [APPLICATIONS_SHEET]


def test_write_xlsx_formats_the_sheet(tmp_path: Path, frozen_now: datetime) -> None:
    apps = build_dataframe(sample_applications(frozen_now))
    out = write_xlsx(apps, tmp_path / "apps.xlsx", events=build_events_dataframe([]))
    with pd.ExcelFile(out, engine="openpyxl") as handle:
        book: Any = handle.book
        sheet = book[APPLICATIONS_SHEET]
        assert sheet.freeze_panes == "A2"
        assert sheet.auto_filter.ref == sheet.dimensions
        assert sheet.column_dimensions["A"].width >= len("application_id")
        assert sheet.column_dimensions["M"].width > 0
        assert book[EVENTS_SHEET].freeze_panes == "A2"


def test_write_xlsx_empty_frame_writes_headers_only(tmp_path: Path) -> None:
    out = write_xlsx(build_dataframe([]), tmp_path / "empty.xlsx")
    reread = pd.read_excel(out, sheet_name=APPLICATIONS_SHEET)
    assert tuple(reread.columns) == EXPORT_COLUMNS
    assert len(reread) == 0


def test_write_xlsx_handles_more_than_26_columns(tmp_path: Path) -> None:
    """Column sizing must survive the AA.. range, so the letter helper is exercised."""
    wide = pd.DataFrame([{f"col{i:02d}": i for i in range(30)}])
    out = write_xlsx(wide, tmp_path / "wide.xlsx")
    with pd.ExcelFile(out, engine="openpyxl") as handle:
        book: Any = handle.book
        sheet = book[APPLICATIONS_SHEET]
        assert sheet.column_dimensions["AD"].width > 0


def test_write_xlsx_creates_parent_directories(tmp_path: Path) -> None:
    out = write_xlsx(build_dataframe([]), tmp_path / "nested" / "apps.xlsx")
    assert out.is_file()


def test_write_xlsx_unwritable_path_raises_export_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ExportError):
        write_xlsx(build_dataframe([]), blocker / "apps.xlsx")


def test_write_xlsx_to_a_directory_raises_export_error(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        write_xlsx(build_dataframe([]), tmp_path)
