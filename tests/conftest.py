"""Shared test fixtures.

No test in this repo may touch the network (CLAUDE.md). Mailbox tests run against the
recorded JSON fixtures in ``tests/fixtures/emails/``; store tests run against a real
SQLite file in ``tmp_path``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobtrack.config import Config
from jobtrack.models import RawMessage

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EMAIL_DIR = FIXTURE_DIR / "emails"
EXPECTED_PATH = FIXTURE_DIR / "expected.jsonl"

FROZEN_NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)
"""A fixed clock. Anything that needs 'now' takes it as a parameter (CLAUDE.md), so tests
pass this rather than patching the clock."""


def load_email_fixture(name: str) -> RawMessage:
    """Load one recorded email fixture by stem.

    Args:
        name: Filename stem, e.g. "greenhouse_rejection".

    Returns:
        The parsed RawMessage.

    Raises:
        FileNotFoundError: no such fixture.
    """
    path = EMAIL_DIR / f"{name}.json"
    return RawMessage.model_validate_json(path.read_text())


def all_email_fixtures() -> list[tuple[str, RawMessage]]:
    """Load every email fixture, sorted by stem for deterministic test ordering."""
    return [
        (p.stem, RawMessage.model_validate_json(p.read_text()))
        for p in sorted(EMAIL_DIR.glob("*.json"))
    ]


def expected_classifications() -> dict[str, dict[str, Any]]:
    """Load the golden expectations keyed by fixture stem.

    Returns:
        Mapping of fixture stem to its expected fields. Empty if expected.jsonl is absent.
    """
    if not EXPECTED_PATH.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in EXPECTED_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        out[row["fixture"]] = row
    return out


@pytest.fixture
def frozen_now() -> datetime:
    """A fixed, tz-aware UTC clock for deterministic assertions."""
    return FROZEN_NOW


@pytest.fixture
def email_fixtures() -> list[tuple[str, RawMessage]]:
    """Every recorded email fixture as (stem, RawMessage)."""
    return all_email_fixtures()


@pytest.fixture
def expected() -> dict[str, dict[str, Any]]:
    """Golden classification expectations keyed by fixture stem."""
    return expected_classifications()


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """A Config rooted at tmp_path, so no test can touch the real JOBTRACK_HOME."""
    home = tmp_path / "jobtrack-home"
    home.mkdir(parents=True, exist_ok=True)
    return Config(home=home)


@pytest.fixture
def make_message() -> Callable[..., RawMessage]:
    """Factory for RawMessages with sensible defaults.

    Lets a test name only the fields it cares about::

        msg = make_message(
            subject="Thanks for applying to Acme",
            from_email="no-reply@greenhouse.io",
        )
    """

    counter = {"n": 0}

    def _make(**overrides: Any) -> RawMessage:
        counter["n"] += 1
        n = counter["n"]
        defaults: dict[str, Any] = {
            "message_id": f"msg-{n:04d}",
            "thread_id": f"thread-{n:04d}",
            "received_at": FROZEN_NOW,
            "from_email": "no-reply@example.com",
            "from_name": "Example Careers",
            "to_email": "candidate@example.com",
            "subject": "",
            "body_text": "",
            "snippet": "",
            "labels": [],
            "headers": {},
        }
        defaults.update(overrides)
        return RawMessage.model_validate(defaults)

    return _make


@pytest.fixture(autouse=True)
def _no_real_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Guard: JOBTRACK_HOME always points somewhere disposable during tests."""
    monkeypatch.setenv("JOBTRACK_HOME", str(tmp_path / "env-home"))
    yield
