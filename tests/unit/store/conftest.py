"""Fixtures shared by store unit tests.

Every test here uses a real SQLite file under ``tmp_path`` (CLAUDE.md forbids mocking the
DB) and a fixed clock rather than ``datetime.now()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobtrack.models import Classification, EventType
from jobtrack.store import Store

NOW: datetime = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)
"""A fixed, tz-aware UTC clock, distinct from tests/conftest.py's FROZEN_NOW so a store test
failure can't be masked by accidentally importing the wrong constant."""


@pytest.fixture
def now() -> datetime:
    """A fixed, tz-aware UTC instant for deterministic assertions."""
    return NOW


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A jobtrack.db path under tmp_path, not yet created."""
    return tmp_path / "jobtrack.db"


@pytest.fixture
def store(db_path: Path) -> Iterator[Store]:
    """An opened and migrated Store, closed automatically at teardown."""
    with Store.open(db_path) as opened:
        opened.migrate()
        yield opened


@pytest.fixture
def make_classification() -> Callable[..., Classification]:
    """Factory for Classifications with sensible defaults.

    Lets a test name only the fields it cares about::

        make_classification(event_type=EventType.REJECTION, company="Acme")
    """

    def _make(**overrides: Any) -> Classification:
        defaults: dict[str, Any] = {
            "message_id": "msg-0001",
            "event_type": EventType.APPLICATION_RECEIVED,
            "company": "Acme Robotics",
            "company_key": "acme robotics",
            "role": "Backend Engineer",
            "location": None,
            "ats": "greenhouse",
            "confidence": 0.9,
            "needs_review": False,
            "evidence": [],
            "classifier_name": "rules",
            "classifier_version": "1.0.0",
        }
        defaults.update(overrides)
        return Classification.model_validate(defaults)

    return _make
