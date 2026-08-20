"""End-to-end: a fixture mailbox through ingest → classify → store → export → dashboard.

The only fake here is the mailbox. Everything downstream is the real thing: the real
rules classifier, a real SQLite file in tmp_path, the real DataFrame builders, and the
real Plotly page. That is the point — this is the test that would have caught the
integration seams the unit suites each pass on their own side of.

No network: ``FakeSource`` replays the recorded fixtures in ``tests/fixtures/emails/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jobtrack.classify import CompositeClassifier, RulesClassifier
from jobtrack.classify.base import Classifier
from jobtrack.config import Config
from jobtrack.export import build_dataframe, build_events_dataframe, write_csv, write_xlsx
from jobtrack.ingest.source import FetchResult
from jobtrack.models import EventType, Override, RawMessage
from jobtrack.store.db import Store
from jobtrack.viz.dashboard import build_dashboard
from tests.conftest import all_email_fixtures

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)


class FakeSource:
    """An EmailSource that replays recorded fixtures instead of calling Gmail.

    Records every ``fetch`` call so the test can assert on how the orchestration drove
    it — which cursor it passed, and whether it honoured the limit.
    """

    name = "gmail"

    def __init__(self, messages: list[RawMessage], *, next_cursor: str | None = "cursor-1") -> None:
        """Args:
        messages: The batch to hand back.
        next_cursor: The cursor to report, or None to report no resume point.
        """
        self.messages = messages
        self.next_cursor = next_cursor
        self.calls: list[dict[str, object]] = []

    def fetch(
        self,
        *,
        query: str,
        since: datetime | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> FetchResult:
        """Return the recorded batch, truncated to `limit`."""
        self.calls.append({"query": query, "since": since, "cursor": cursor, "limit": limit})
        batch = self.messages if limit is None else self.messages[:limit]
        return FetchResult(
            messages=batch,
            next_cursor=self.next_cursor,
            fetched_at=NOW,
            truncated=limit is not None and len(batch) < len(self.messages),
        )


@pytest.fixture
def fixtures() -> list[RawMessage]:
    """Every recorded email fixture, in a deterministic order."""
    return [message for _, message in all_email_fixtures()]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A Config rooted in tmp_path so nothing touches the real JOBTRACK_HOME."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    return Config(home=home)


@pytest.fixture
def store(config: Config) -> Iterator[Store]:
    """A migrated store in the throwaway home."""
    with Store.open_from_config(config) as opened:
        opened.migrate()
        yield opened


@pytest.fixture
def classifier() -> Classifier:
    """The real Phase 1 classifier stack."""
    return CompositeClassifier(RulesClassifier(), None)


def test_sync_ingests_the_whole_fixture_mailbox(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """Every fixture is fetched, classified, and recorded exactly once."""
    from jobtrack.cli import run_sync

    report = run_sync(FakeSource(fixtures), classifier, store, config, now=NOW)

    assert report.fetched == len(fixtures)
    assert report.new_messages == len(fixtures)
    assert report.events_created == len(fixtures)
    assert len(store.list_events()) == len(fixtures)


def test_sync_is_idempotent(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """I1: replaying the same mailbox creates no second event and no second application."""
    from jobtrack.cli import run_sync

    first = run_sync(FakeSource(fixtures), classifier, store, config, now=NOW)
    events_after_first = store.list_events()
    applications_after_first = store.list_applications(now=NOW)

    second = run_sync(FakeSource(fixtures), classifier, store, config, now=NOW)

    assert second.new_messages == 0
    assert second.events_created == 0
    assert second.applications_created == 0
    assert store.list_events() == events_after_first
    assert store.list_applications(now=NOW) == applications_after_first
    assert first.fetched == second.fetched


def test_sync_advances_the_cursor_only_after_the_batch(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """I9: the cursor is persisted after the messages land, and reused next run."""
    from jobtrack.cli import run_sync

    source = FakeSource(fixtures, next_cursor="history-42")
    run_sync(source, classifier, store, config, now=NOW)
    assert store.get_cursor("gmail") == "history-42"

    run_sync(source, classifier, store, config, now=NOW)
    assert source.calls[1]["cursor"] == "history-42"


def test_sync_dry_run_writes_nothing(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """A dry run reports what it would do, including the cursor it would not advance."""
    from jobtrack.cli import run_sync

    report = run_sync(FakeSource(fixtures), classifier, store, config, now=NOW, dry_run=True)

    assert report.new_messages == len(fixtures)
    assert report.events_created == 0
    assert store.list_events() == []
    assert store.get_cursor("gmail") is None


def test_sync_full_ignores_the_stored_cursor(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """--full re-scans the lookback window rather than resuming."""
    from jobtrack.cli import run_sync

    source = FakeSource(fixtures)
    run_sync(source, classifier, store, config, now=NOW)
    run_sync(source, classifier, store, config, now=NOW, full=True)

    assert source.calls[1]["cursor"] is None
    assert source.calls[1]["since"] is not None


def test_sync_honours_the_limit_and_reports_truncation(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """A capped batch says so, so the operator knows to run again."""
    from jobtrack.cli import run_sync

    report = run_sync(FakeSource(fixtures), classifier, store, config, now=NOW, limit=2)

    assert report.fetched == 2
    assert any("truncated" in problem for problem in report.errors)


def test_sync_defaults_since_to_the_configured_lookback(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """With no cursor and no explicit --since, the lookback window applies."""
    from jobtrack.cli import run_sync

    source = FakeSource(fixtures)
    run_sync(source, classifier, store, config, now=NOW)

    expected = NOW - timedelta(days=config.gmail.lookback_days)
    assert source.calls[0]["since"] == expected


def test_sync_passes_an_explicit_since_through(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """An explicit lower bound overrides the configured lookback."""
    from jobtrack.cli import run_sync

    source = FakeSource(fixtures)
    bound = datetime(2026, 6, 1, tzinfo=UTC)
    run_sync(source, classifier, store, config, now=NOW, since=bound)

    assert source.calls[0]["since"] == bound


def test_sync_is_deterministic(
    tmp_path: Path, classifier: Classifier, fixtures: list[RawMessage]
) -> None:
    """The same mailbox into two fresh databases yields identical applications."""
    from jobtrack.cli import run_sync

    def sync_into(name: str) -> list[str]:
        home = tmp_path / name
        home.mkdir()
        config = Config(home=home)
        with Store.open_from_config(config) as store:
            store.migrate()
            run_sync(FakeSource(fixtures), classifier, store, config, now=NOW)
            return [row.model_dump_json() for row in store.list_applications(now=NOW)]

    assert sync_into("a") == sync_into("b")


def test_full_pipeline_reaches_export_and_dashboard(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage], tmp_path: Path
) -> None:
    """The whole chain: sync, then a spreadsheet and a dashboard off the same store."""
    from jobtrack.cli import run_sync

    run_sync(FakeSource(fixtures), classifier, store, config, now=NOW)
    applications = build_dataframe(store.list_applications(now=NOW))
    events = build_events_dataframe(store.list_events())

    csv_path = write_csv(applications, tmp_path / "apps.csv")
    xlsx_path = write_xlsx(applications, tmp_path / "apps.xlsx", events=events)
    html_path = build_dashboard(applications, events, tmp_path / "dash.html")

    assert csv_path.is_file()
    assert xlsx_path.is_file()
    assert html_path.is_file()
    assert len(applications) == len(store.list_applications(now=NOW))


def test_reclassify_after_sync_preserves_an_override(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """I6: a human correction survives a full reclassify of the stored mailbox."""
    from jobtrack.cli import run_sync

    run_sync(FakeSource(fixtures), classifier, store, config, now=NOW)
    target = store.list_events()[0]
    store.set_override(
        Override(
            message_id=target.message_id,
            event_type=EventType.OFFER,
            corrected_at=NOW,
        )
    )

    store.clear_classifications(only_unreviewed=True)
    for message in store.list_messages():
        store.reapply_classification(message, classifier.classify(message), now=NOW)

    corrected = {event.message_id: event for event in store.list_events()}[target.message_id]
    assert corrected.event_type is EventType.OFFER
    assert corrected.is_overridden is True


def test_sync_flags_low_confidence_messages_for_review(
    store: Store, classifier: Classifier, config: Config, fixtures: list[RawMessage]
) -> None:
    """Whatever the classifier is unsure about ends up walkable by `jobtrack review`."""
    from jobtrack.cli import run_sync

    report = run_sync(FakeSource(fixtures), classifier, store, config, now=NOW)

    assert len(store.pending_review()) == report.needs_review
