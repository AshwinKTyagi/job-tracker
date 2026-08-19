"""Tabular export: ``ApplicationRow[]`` -> ``pandas.DataFrame`` -> ``.csv`` / ``.xlsx``.

Column names, order, and dtypes are fixed by ``constants.EXPORT_COLUMNS`` and
``constants.EVENT_COLUMNS`` (invariant I10). M5 reads exactly these frames, so nothing here
may reorder, rename, or drop a column — not even for an empty input.

Datetimes are tz-aware UTC in the frame and ISO-8601 UTC on disk (invariant I7).
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.errors import ExportError
from jobtrack.models import ApplicationRow, EventRow

logger = logging.getLogger(__name__)

ISO_8601_UTC: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
"""On-disk datetime format (I7). Every frame datetime is UTC, so the 'Z' suffix is exact."""

UTC_DATETIME_DTYPE: Final[str] = "datetime64[ns, UTC]"
STRING_DTYPE: Final[str] = "str"
NULLABLE_INT_DTYPE: Final[str] = "Int64"

APPLICATIONS_SHEET: Final[str] = "Applications"
EVENTS_SHEET: Final[str] = "Events"

APPLICATION_DTYPES: Final[Mapping[str, str]] = {
    "application_id": STRING_DTYPE,
    "company": STRING_DTYPE,
    "role": STRING_DTYPE,
    "location": STRING_DTYPE,
    "ats": STRING_DTYPE,
    "status": STRING_DTYPE,
    "last_event_type": STRING_DTYPE,
    "event_count": "int64",
    "days_to_first_response": NULLABLE_INT_DTYPE,
    "days_since_last_event": "int64",
    "needs_review": "bool",
}
"""Non-datetime dtypes for the applications frame. Pinned so an empty frame matches a full one."""

APPLICATION_DATETIME_COLUMNS: Final[tuple[str, ...]] = ("applied_at", "last_event_at")

EVENT_DTYPES: Final[Mapping[str, str]] = {
    "application_id": STRING_DTYPE,
    "message_id": STRING_DTYPE,
    "event_type": STRING_DTYPE,
    "confidence": "float64",
    "needs_review": "bool",
    "subject": STRING_DTYPE,
}
"""Non-datetime dtypes for the long-format events frame."""

EVENT_DATETIME_COLUMNS: Final[tuple[str, ...]] = ("occurred_at",)

_FREEZE_HEADER_ROW: Final[str] = "A2"
_MIN_COLUMN_WIDTH: Final[int] = 10
_MAX_COLUMN_WIDTH: Final[int] = 60
_COLUMN_PADDING: Final[int] = 2
_EXCEL_ALPHABET_SIZE: Final[int] = 26


def build_dataframe(applications: Sequence[ApplicationRow]) -> pd.DataFrame:
    """Applications -> DataFrame with exactly ``EXPORT_COLUMNS``, in order (I10).

    Datetimes are tz-aware UTC. Empty input yields an empty frame with the correct columns
    and dtypes — downstream code must never special-case emptiness.

    Args:
        applications: Rows to export, in the order they should appear.

    Returns:
        A DataFrame whose columns are ``EXPORT_COLUMNS`` exactly, with a fresh RangeIndex.
    """
    records = [_application_record(row) for row in applications]
    return _build_frame(
        records,
        columns=EXPORT_COLUMNS,
        dtypes=APPLICATION_DTYPES,
        datetime_columns=APPLICATION_DATETIME_COLUMNS,
    )


def build_events_dataframe(events: Sequence[EventRow]) -> pd.DataFrame:
    """Events -> long-format DataFrame with exactly ``EVENT_COLUMNS``, in order (I10).

    Columns: application_id, message_id, event_type, occurred_at, confidence, needs_review,
    subject. ``application_id`` is missing for unlinked (UNKNOWN) events.

    Args:
        events: Event rows, normally oldest first.

    Returns:
        A DataFrame whose columns are ``EVENT_COLUMNS`` exactly, with a fresh RangeIndex.
    """
    records = [_event_record(row) for row in events]
    return _build_frame(
        records,
        columns=EVENT_COLUMNS,
        dtypes=EVENT_DTYPES,
        datetime_columns=EVENT_DATETIME_COLUMNS,
    )


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write UTF-8 CSV (ISO-8601 dates, no index).

    Args:
        df: Frame to write, normally from ``build_dataframe``.
        path: Destination file. Missing parent directories are created.

    Returns:
        The resolved path that was written.

    Raises:
        ExportError: path is not writable.
    """
    target = _prepare_target(path)
    try:
        _isoformat_datetimes(df).to_csv(target, index=False, encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"could not write {target}: {exc}") from exc
    logger.debug("wrote %d rows to %s", len(df), target)
    return target


def write_xlsx(df: pd.DataFrame, path: Path, *, events: pd.DataFrame | None = None) -> Path:
    """Write an .xlsx via openpyxl: sheet 'Applications', plus 'Events' when provided.

    Freezes the header row, autosizes columns, and adds an autofilter. Datetimes are written
    as ISO-8601 UTC text so the UTC offset survives the trip through Excel (I7).

    Args:
        df: Applications frame, normally from ``build_dataframe``.
        path: Destination .xlsx file. Missing parent directories are created.
        events: Optional long-format events frame for the second sheet.

    Returns:
        The resolved path that was written.

    Raises:
        ExportError: path not writable or openpyxl failed.
    """
    target = _prepare_target(path)
    try:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            _write_sheet(writer, df, APPLICATIONS_SHEET)
            if events is not None:
                _write_sheet(writer, events, EVENTS_SHEET)
    except (OSError, ValueError) as exc:
        raise ExportError(f"could not write {target}: {exc}") from exc
    logger.debug("wrote %d rows to %s", len(df), target)
    return target


def _application_record(row: ApplicationRow) -> dict[str, Any]:
    """Project one ApplicationRow onto EXPORT_COLUMNS, coercing enums to plain str."""
    return {
        "application_id": row.application_id,
        "company": row.company,
        "role": row.role,
        "location": row.location,
        "ats": row.ats,
        "status": str(row.status),
        "applied_at": row.applied_at,
        "last_event_at": row.last_event_at,
        "last_event_type": str(row.last_event_type),
        "event_count": row.event_count,
        "days_to_first_response": row.days_to_first_response,
        "days_since_last_event": row.days_since_last_event,
        "needs_review": row.needs_review,
    }


def _event_record(row: EventRow) -> dict[str, Any]:
    """Project one EventRow onto EVENT_COLUMNS, coercing enums to plain str."""
    return {
        "application_id": row.application_id,
        "message_id": row.message_id,
        "event_type": str(row.event_type),
        "occurred_at": row.occurred_at,
        "confidence": row.confidence,
        "needs_review": row.needs_review,
        "subject": row.subject,
    }


def _build_frame(
    records: list[dict[str, Any]],
    *,
    columns: tuple[str, ...],
    dtypes: Mapping[str, str],
    datetime_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Assemble a frame with a frozen column order and pinned dtypes, empty input included."""
    frame = pd.DataFrame.from_records(records, columns=list(columns))
    typed = frame.astype(dict(dtypes))
    for column in datetime_columns:
        typed[column] = typed[column].astype(UTC_DATETIME_DTYPE)
    return typed


def _isoformat_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every tz-aware datetime column rendered as ISO-8601 UTC text."""
    out = df.copy()
    for column, dtype in df.dtypes.items():
        if isinstance(dtype, pd.DatetimeTZDtype):
            rendered = df[column].dt.tz_convert("UTC").dt.strftime(ISO_8601_UTC)
            out[column] = rendered.astype(STRING_DTYPE)
    return out


def _prepare_target(path: Path) -> Path:
    """Resolve the destination and make sure its parent directory exists."""
    target = path.expanduser().resolve()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"could not create {target.parent}: {exc}") from exc
    return target


def _write_sheet(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
    """Write one frame as a sheet, then freeze the header, autofilter, and size the columns."""
    formatted = _isoformat_datetimes(df)
    formatted.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet: Any = writer.sheets[sheet_name]
    worksheet.freeze_panes = _FREEZE_HEADER_ROW
    worksheet.auto_filter.ref = worksheet.dimensions
    for offset, column in enumerate(formatted.columns, start=1):
        width = _column_width(column, formatted[column])
        worksheet.column_dimensions[_column_letter(offset)].width = width


def _column_width(column: Hashable, values: pd.Series[Any]) -> int:
    """Pick a display width wide enough for the header and the widest rendered cell."""
    widths: list[int] = [len(str(column))]
    rendered = values.dropna().astype(STRING_DTYPE)
    if len(rendered) > 0:
        widths.append(int(rendered.str.len().max()))
    return min(max(max(widths) + _COLUMN_PADDING, _MIN_COLUMN_WIDTH), _MAX_COLUMN_WIDTH)


def _column_letter(index: int) -> str:
    """Convert a 1-based column index to its spreadsheet letter (1 -> A, 27 -> AA)."""
    letters: list[str] = []
    while index > 0:
        index, remainder = divmod(index - 1, _EXCEL_ALPHABET_SIZE)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


__all__ = [
    "APPLICATIONS_SHEET",
    "APPLICATION_DTYPES",
    "EVENTS_SHEET",
    "EVENT_DTYPES",
    "ISO_8601_UTC",
    "build_dataframe",
    "build_events_dataframe",
    "write_csv",
    "write_xlsx",
]
