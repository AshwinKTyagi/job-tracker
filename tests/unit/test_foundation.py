"""Tests for the M0 foundation: models, constants, config, and the fixture harness.

These lock down the invariants every Phase 1 module builds on.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from jobtrack.config import (
    DEFAULT_JOBTRACK_HOME,
    Config,
    ensure_home,
    load_config,
    resolve_home,
)
from jobtrack.constants import (
    EVENT_PRECEDENCE,
    EXPORT_COLUMNS,
    GMAIL_SCOPES,
    TERMINAL_EVENTS,
)
from jobtrack.errors import ConfigError, JobTrackError, TransientIngestError
from jobtrack.models import ApplicationStatus, Classification, EventType, RawMessage
from tests.conftest import all_email_fixtures, expected_classifications


class TestConstants:
    def test_precedence_covers_every_event_type(self) -> None:
        assert set(EVENT_PRECEDENCE) == set(EventType)

    def test_precedence_has_no_duplicates(self) -> None:
        assert len(EVENT_PRECEDENCE) == len(set(EVENT_PRECEDENCE))

    def test_rejection_outranks_application_received(self) -> None:
        """I3: the whole reason precedence exists. Rejections restate the confirmation wording."""
        assert EVENT_PRECEDENCE.index(EventType.REJECTION) < EVENT_PRECEDENCE.index(
            EventType.APPLICATION_RECEIVED
        )

    def test_rejection_outranks_interview(self) -> None:
        """Post-interview rejections restate the interview."""
        assert EVENT_PRECEDENCE.index(EventType.REJECTION) < EVENT_PRECEDENCE.index(
            EventType.INTERVIEW
        )

    def test_unknown_is_lowest_precedence(self) -> None:
        assert EVENT_PRECEDENCE[-1] is EventType.UNKNOWN

    def test_terminal_events(self) -> None:
        assert {EventType.REJECTION, EventType.OFFER, EventType.WITHDRAWN} == TERMINAL_EVENTS

    def test_export_columns_frozen(self) -> None:
        """I10: the M4 <-> M5 wire format. Changing this breaks a parallel agent's work."""
        assert EXPORT_COLUMNS == (
            "application_id",
            "company",
            "role",
            "location",
            "ats",
            "status",
            "applied_at",
            "last_event_at",
            "last_event_type",
            "event_count",
            "days_to_first_response",
            "days_since_last_event",
            "needs_review",
        )

    def test_gmail_scope_is_readonly_only(self) -> None:
        """I11: read-only is the entire scope budget."""
        assert GMAIL_SCOPES == ["https://www.googleapis.com/auth/gmail.readonly"]
        assert all("readonly" in s for s in GMAIL_SCOPES)


class TestModels:
    def test_raw_message_is_frozen(self, make_message: Callable[..., RawMessage]) -> None:
        msg = make_message()
        with pytest.raises(ValidationError):
            msg.subject = "mutated"  # type: ignore[misc]

    def test_classification_is_frozen(self) -> None:
        c = Classification(
            message_id="m1",
            event_type=EventType.REJECTION,
            confidence=0.9,
            classifier_name="rules",
            classifier_version="1.0.0",
        )
        with pytest.raises(ValidationError):
            c.confidence = 0.1  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_confidence_bounds_enforced(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            Classification(
                message_id="m1",
                event_type=EventType.UNKNOWN,
                confidence=bad,
                classifier_name="rules",
                classifier_version="1.0.0",
            )

    def test_enums_are_str_valued(self) -> None:
        """Store and export both round-trip these through strings."""
        assert EventType.REJECTION.value == "rejection"
        assert ApplicationStatus.INTERVIEWING.value == "interviewing"

    def test_mutable_defaults_are_not_shared(self, make_message: Callable[..., RawMessage]) -> None:
        a, b = make_message(), make_message()
        assert a.labels is not b.labels

    def test_message_ids_are_unique_per_factory_call(
        self, make_message: Callable[..., RawMessage]
    ) -> None:
        """I1 depends on message_id uniqueness; the factory must not hand out collisions."""
        ids = {make_message().message_id for _ in range(50)}
        assert len(ids) == 50


class TestConfig:
    def test_defaults_need_no_file(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path)
        assert cfg.classify.min_confidence == 0.60
        assert cfg.store.ghost_after_days == 30
        assert cfg.classify.backend == "rules"

    def test_toml_overrides_merge_over_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text(
            "[classify]\nmin_confidence = 0.8\n\n[store]\nghost_after_days = 14\n"
        )
        cfg = load_config(tmp_path)
        assert cfg.classify.min_confidence == 0.8
        assert cfg.store.ghost_after_days == 14
        assert cfg.gmail.lookback_days == 400  # untouched default survives

    def test_malformed_toml_raises_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[classify\nbroken")
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_invalid_value_raises_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[classify]\nmin_confidence = 5.0\n")
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_paths_derive_from_home(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path)
        assert cfg.db_path == cfg.home / "jobtrack.db"
        assert cfg.credentials_path == cfg.home / "credentials.json"
        assert cfg.token_path == cfg.home / "token.json"

    def test_secrets_never_land_in_the_repo(self, tmp_path: Path) -> None:
        """The repo must never be a plausible home for tokens or the database."""
        cfg = load_config(tmp_path)
        repo = Path(__file__).resolve().parents[2]
        for p in (cfg.db_path, cfg.token_path, cfg.credentials_path):
            assert repo not in p.parents

    def test_env_var_is_honoured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JOBTRACK_HOME", str(tmp_path / "from-env"))
        assert resolve_home() == (tmp_path / "from-env").resolve()

    def test_explicit_argument_beats_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JOBTRACK_HOME", str(tmp_path / "from-env"))
        assert resolve_home(tmp_path / "explicit") == (tmp_path / "explicit").resolve()

    def test_default_home_is_outside_the_repo(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        assert repo not in DEFAULT_JOBTRACK_HOME.parents

    def test_ensure_home_creates_directory(self, tmp_path: Path) -> None:
        cfg = Config(home=tmp_path / "nested" / "home")
        assert ensure_home(cfg).is_dir()


class TestErrors:
    def test_all_errors_share_a_base(self) -> None:
        assert issubclass(TransientIngestError, JobTrackError)
        assert issubclass(ConfigError, JobTrackError)

    def test_transient_is_distinguishable(self) -> None:
        """cli.py maps transient failures to exit 4; that requires the distinction to hold."""
        with pytest.raises(TransientIngestError):
            raise TransientIngestError("429 from Gmail")


class TestFixtureHarness:
    def test_fixtures_load_as_raw_messages(self) -> None:
        fixtures = all_email_fixtures()
        assert len(fixtures) >= 3
        assert all(isinstance(m, RawMessage) for _, m in fixtures)

    def test_every_fixture_has_an_expectation(self) -> None:
        """CLAUDE.md: a fixture without a golden expectation is dead weight."""
        stems = {stem for stem, _ in all_email_fixtures()}
        assert stems == set(expected_classifications())

    def test_fixture_datetimes_are_utc_aware(self) -> None:
        """I7: a naive datetime crossing a boundary is a bug."""
        for _, msg in all_email_fixtures():
            assert msg.received_at.tzinfo is not None
            assert msg.received_at.utcoffset() == timedelta(0)

    def test_the_confusable_pair_is_present(self) -> None:
        """The confirmation and rejection must share a subject but differ in expected type.

        This is the fixture that justifies the whole precedence design; if it ever stops
        being confusable, M2's hardest case has silently vanished from the suite.
        """
        exp = expected_classifications()
        confirm = next(m for s, m in all_email_fixtures() if s == "greenhouse_confirmation")
        reject = next(m for s, m in all_email_fixtures() if s == "greenhouse_rejection")

        assert confirm.subject == reject.subject
        assert "thanks for applying" in reject.body_text.lower()
        assert exp["greenhouse_confirmation"]["event_type"] == "application_received"
        assert exp["greenhouse_rejection"]["event_type"] == "rejection"

    def test_expected_event_types_are_valid(self) -> None:
        for stem, row in expected_classifications().items():
            EventType(row["event_type"])  # raises if the golden file drifts from the enum
            assert row["fixture"] == stem
