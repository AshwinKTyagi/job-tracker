"""Tests for the Typer surface and the exit-code mapping in ``main``.

No command here touches the network: the Gmail source is replaced wherever a command
would build one. ``conftest`` points JOBTRACK_HOME at tmp_path for every test, so these
run against a disposable home.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jobtrack import cli
from jobtrack.errors import (
    AuthError,
    ClassificationError,
    ConfigError,
    StoreError,
    TransientIngestError,
)
from jobtrack.ingest.source import FetchResult
from jobtrack.models import Classification, EventType, RawMessage
from jobtrack.store.db import Store

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)

MessageFactory = Callable[..., RawMessage]

runner = CliRunner()

#: A <script> tag that pulls from the network. Matches tags only, never JS string bodies.
_EXTERNAL_SCRIPT_RE = re.compile(r"<script\b[^>]*\bsrc\s*=", re.IGNORECASE)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A disposable JOBTRACK_HOME the CLI will resolve to."""
    path = tmp_path / "home"
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("JOBTRACK_HOME", str(path))
    return path


def classify(message: RawMessage, **overrides: Any) -> Classification:
    """Build a Classification for a message, standing in for M2."""
    fields: dict[str, Any] = {
        "message_id": message.message_id,
        "event_type": EventType.APPLICATION_RECEIVED,
        "company": "Acme Robotics",
        "company_key": "acme robotics",
        "role": "Software Engineer",
        "location": "Remote",
        "ats": "greenhouse",
        "confidence": 0.9,
        "needs_review": False,
        "evidence": ["ack.body.application_received"],
        "classifier_name": "rules",
        "classifier_version": "1.0.0",
    }
    fields.update(overrides)
    return Classification.model_validate(fields)


@pytest.fixture
def seeded(home: Path, make_message: MessageFactory) -> Iterator[Path]:
    """A home whose database already holds one linked application."""
    from jobtrack.config import load_config

    config = load_config()
    with Store.open_from_config(config) as store:
        store.migrate()
        message = make_message(subject="Thanks for applying to Acme")
        store.link_and_record_event(message, classify(message), now=NOW)
    yield home


# --- read-only commands -----------------------------------------------------


def test_list_reports_an_empty_database(home: Path) -> None:
    """An empty store is a normal outcome, not an error."""
    result = runner.invoke(cli.app, ["list"])
    assert result.exit_code == 0
    assert "no applications match" in result.output


def test_list_shows_a_seeded_application(seeded: Path) -> None:
    """The table names the company the store derived."""
    result = runner.invoke(cli.app, ["list"])
    assert result.exit_code == 0
    assert "Acme Robotics" in result.output


def test_list_rejects_an_unknown_status(seeded: Path) -> None:
    """A bad --status is a usage error, and it names the valid values."""
    result = runner.invoke(cli.app, ["list", "--status", "bogus"])
    assert result.exit_code == cli.EXIT_USAGE
    assert "applied" in result.output


def test_list_filters_by_company(seeded: Path) -> None:
    """--company matches against company_key, so it is normalization-insensitive."""
    hit = runner.invoke(cli.app, ["list", "--company", "ACME"])
    miss = runner.invoke(cli.app, ["list", "--company", "zzz"])
    assert "Acme Robotics" in hit.output
    assert "no applications match" in miss.output


def test_stats_reports_an_empty_database(home: Path) -> None:
    """Stats on an empty store points at sync rather than dividing by zero."""
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0
    assert "no applications yet" in result.output


def test_stats_reports_a_response_rate(seeded: Path) -> None:
    """One acknowledgement and no reply is a 0% response rate."""
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0
    assert "response rate" in result.output


def test_db_migrate_reports_the_schema_version(home: Path) -> None:
    """Migrate is idempotent and prints where the schema landed."""
    first = runner.invoke(cli.app, ["db", "migrate"])
    second = runner.invoke(cli.app, ["db", "migrate"])
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "schema at version" in second.output


def test_auth_status_reports_a_missing_token(home: Path) -> None:
    """A missing token is a reported state, never an exception."""
    result = runner.invoke(cli.app, ["auth", "status"])
    assert result.exit_code == 0
    assert "auth login" in result.output


# --- export and dashboard ---------------------------------------------------


def test_export_writes_csv(seeded: Path, tmp_path: Path) -> None:
    """--format csv writes a real file at the requested path."""
    target = tmp_path / "out.csv"
    result = runner.invoke(cli.app, ["export", "--format", "csv", "-o", str(target)])
    assert result.exit_code == 0
    assert target.is_file()
    assert "application_id" in target.read_text(encoding="utf-8")


def test_export_writes_xlsx(seeded: Path, tmp_path: Path) -> None:
    """xlsx is the configured default format."""
    target = tmp_path / "out.xlsx"
    result = runner.invoke(cli.app, ["export", "-o", str(target)])
    assert result.exit_code == 0
    assert target.is_file()


def test_export_rejects_an_unknown_format(seeded: Path) -> None:
    """Anything but csv or xlsx is a usage error."""
    result = runner.invoke(cli.app, ["export", "--format", "pdf"])
    assert result.exit_code == cli.EXIT_USAGE


def test_export_defaults_into_jobtrack_home(seeded: Path) -> None:
    """With no -o the snapshot lands in JOBTRACK_HOME, never in the repo."""
    result = runner.invoke(cli.app, ["export", "--format", "csv"])
    assert result.exit_code == 0
    assert (seeded / "applications.csv").is_file()


def test_dashboard_writes_self_contained_html(seeded: Path, tmp_path: Path) -> None:
    """The dashboard inlines plotly.js so it renders with the network unplugged."""
    target = tmp_path / "dash.html"
    result = runner.invoke(cli.app, ["dashboard", "-o", str(target)])
    assert result.exit_code == 0
    page = target.read_text(encoding="utf-8")
    assert "<html" in page.lower()
    # Inlined, not linked. Matching on tags rather than raw substrings: the plotly bundle
    # is full of CDN URLs in its own JS, so a substring search would false-positive.
    assert _EXTERNAL_SCRIPT_RE.search(page) is None
    assert "<script" in page


def test_dashboard_opens_the_browser_when_asked(
    seeded: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--open hands the written file to the browser, and nothing else."""
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
    target = tmp_path / "dash.html"

    result = runner.invoke(cli.app, ["dashboard", "-o", str(target), "--open"])

    assert result.exit_code == 0
    assert opened == [target.resolve().as_uri()]


# --- reclassify -------------------------------------------------------------


def test_reclassify_reruns_every_stored_message(seeded: Path) -> None:
    """Reclassify walks the stored mailbox and reports what it touched."""
    result = runner.invoke(cli.app, ["reclassify"])
    assert result.exit_code == 0
    assert "reclassified 1 message(s)" in result.output


def test_reclassify_all_clears_reviewed_rows_too(seeded: Path) -> None:
    """--all discards human labels as well; the default keeps them (I6)."""
    result = runner.invoke(cli.app, ["reclassify", "--all"])
    assert result.exit_code == 0
    assert "cleared" in result.output


# --- review -----------------------------------------------------------------


@pytest.fixture
def queued(home: Path, make_message: MessageFactory) -> Path:
    """A home holding one low-confidence message awaiting review."""
    from jobtrack.config import load_config

    config = load_config()
    with Store.open_from_config(config) as store:
        store.migrate()
        message = make_message(subject="Something ambiguous")
        store.link_and_record_event(
            message, classify(message, confidence=0.2, needs_review=True), now=NOW
        )
    return home


def test_review_reports_an_empty_queue(seeded: Path) -> None:
    """Nothing flagged means nothing to walk."""
    result = runner.invoke(cli.app, ["review"])
    assert result.exit_code == 0
    assert "review queue is empty" in result.output


def test_review_accepts_a_guess(queued: Path) -> None:
    """Accepting clears the flag without changing any field."""
    result = runner.invoke(cli.app, ["review"], input="a\n")
    assert result.exit_code == 0
    assert "accepted" in result.output

    after = runner.invoke(cli.app, ["review"])
    assert "review queue is empty" in after.output


def test_review_shows_the_evidence_rule_ids(queued: Path) -> None:
    """The queue explains *why* the classifier guessed what it did."""
    result = runner.invoke(cli.app, ["review"], input="s\n")
    assert "ack.body.application_received" in result.output


def test_review_skip_leaves_the_item_queued(queued: Path) -> None:
    """Skipping is not a decision — the item comes back next time."""
    runner.invoke(cli.app, ["review"], input="s\n")
    again = runner.invoke(cli.app, ["review"], input="s\n")
    assert "Something ambiguous" in again.output


def test_review_quit_stops_the_walk(queued: Path) -> None:
    """Quitting leaves the rest of the queue untouched."""
    result = runner.invoke(cli.app, ["review"], input="q\n")
    assert result.exit_code == 0
    again = runner.invoke(cli.app, ["review"], input="s\n")
    assert "Something ambiguous" in again.output


def test_review_records_a_correction(queued: Path) -> None:
    """A correction becomes an Override, and the flag clears."""
    result = runner.invoke(cli.app, ["review"], input="c\nrejection\nAcme Robotics\n\n\n")
    assert result.exit_code == 0
    assert "correction saved" in result.output

    listed = runner.invoke(cli.app, ["list"])
    assert "rejected" in listed.output


def test_review_reprompts_on_an_unknown_answer(queued: Path) -> None:
    """A typo re-asks rather than silently skipping the item."""
    result = runner.invoke(cli.app, ["review"], input="x\na\n")
    assert result.exit_code == 0
    assert "expected a, c, s, or q" in result.output


def test_review_rejects_an_unknown_event_type(queued: Path) -> None:
    """A bad event type is reported and dropped, leaving the other corrections intact."""
    result = runner.invoke(cli.app, ["review"], input="c\nnonsense\nAcme\n\n\n")
    assert result.exit_code == 0
    assert "not an event type" in result.output


# --- exit-code mapping ------------------------------------------------------


def _main_with(monkeypatch: pytest.MonkeyPatch, exc: BaseException | None) -> int:
    """Run ``main`` with the Typer app replaced by something that raises ``exc``."""

    def fake_app() -> None:
        if exc is not None:
            raise exc

    monkeypatch.setattr(cli, "app", fake_app)
    return cli.main()


def test_main_returns_ok_when_nothing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean run is exit 0."""
    assert _main_with(monkeypatch, None) == cli.EXIT_OK


def test_main_maps_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """AuthError is exit 3 and points at the login command."""
    assert _main_with(monkeypatch, AuthError("no token")) == cli.EXIT_AUTH


def test_main_maps_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retryable network or quota failure is exit 4."""
    assert _main_with(monkeypatch, TransientIngestError("429")) == cli.EXIT_TRANSIENT


def test_main_maps_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed config is a usage problem, exit 2."""
    assert _main_with(monkeypatch, ConfigError("bad toml")) == cli.EXIT_USAGE


def test_main_maps_other_jobtrack_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything else in the hierarchy is exit 1."""
    assert _main_with(monkeypatch, StoreError("disk full")) == cli.EXIT_ERROR


def test_main_passes_through_a_system_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typer reports usage errors itself; main forwards the code it chose."""
    assert _main_with(monkeypatch, SystemExit(2)) == cli.EXIT_USAGE


def test_main_treats_a_bare_system_exit_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """SystemExit(None) — what --help raises — is exit 0."""
    assert _main_with(monkeypatch, SystemExit(None)) == cli.EXIT_OK


def test_main_treats_a_string_system_exit_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SystemExit carrying a message, not a code, is exit 1."""
    assert _main_with(monkeypatch, SystemExit("boom")) == cli.EXIT_ERROR


def test_main_does_not_swallow_unexpected_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JobTrackError bug must surface with its traceback, not a tidy exit code."""
    with pytest.raises(RuntimeError):
        _main_with(monkeypatch, RuntimeError("bug"))


# --- sync wiring ------------------------------------------------------------


class StubSource:
    """An EmailSource that hands back a fixed batch, standing in for Gmail."""

    name = "gmail"

    def __init__(self, messages: list[RawMessage]) -> None:
        """Args:
        messages: The batch every fetch returns.
        """
        self.messages = messages
        self.calls: list[dict[str, object]] = []

    def fetch(
        self,
        *,
        query: str,
        since: datetime | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> FetchResult:
        """Record the call and return the batch."""
        self.calls.append({"query": query, "since": since, "cursor": cursor, "limit": limit})
        return FetchResult(
            messages=self.messages, next_cursor="cursor-1", fetched_at=NOW, truncated=False
        )


@pytest.fixture
def stub_source(monkeypatch: pytest.MonkeyPatch, make_message: MessageFactory) -> StubSource:
    """Replace the Gmail source the sync command would otherwise build."""
    source = StubSource(
        [
            make_message(
                subject="Thanks for applying to Acme Robotics",
                body_text="We have received your application for Software Engineer.",
                from_email="no-reply@greenhouse.io",
            )
        ]
    )
    monkeypatch.setattr(cli, "_build_source", lambda config: source)
    return source


def test_sync_records_a_message_end_to_end(home: Path, stub_source: StubSource) -> None:
    """The sync command wires source, classifier, and store together."""
    result = runner.invoke(cli.app, ["sync"])
    assert result.exit_code == 0
    assert "sync complete" in result.output

    # Assert on the count, not the company text: rich truncates columns to the terminal
    # width, so a narrow run would drop the name without anything being wrong.
    listed = runner.invoke(cli.app, ["list"])
    assert "1 application(s)" in listed.output


def test_sync_dry_run_writes_nothing(home: Path, stub_source: StubSource) -> None:
    """--dry-run reports without touching the database."""
    result = runner.invoke(cli.app, ["sync", "--dry-run"])
    assert result.exit_code == 0
    assert "dry run" in result.output

    listed = runner.invoke(cli.app, ["list"])
    assert "no applications match" in listed.output


def test_sync_passes_a_day_count_since(home: Path, stub_source: StubSource) -> None:
    """--since 30 is read as 30 days ago, not as a date."""
    result = runner.invoke(cli.app, ["sync", "--since", "30"])
    assert result.exit_code == 0
    since = stub_source.calls[0]["since"]
    assert isinstance(since, datetime)
    assert (datetime.now(UTC) - since).days == 30


def test_sync_passes_an_iso_date_since(home: Path, stub_source: StubSource) -> None:
    """A bare ISO date is read as midnight UTC."""
    result = runner.invoke(cli.app, ["sync", "--since", "2026-06-01"])
    assert result.exit_code == 0
    assert stub_source.calls[0]["since"] == datetime(2026, 6, 1, tzinfo=UTC)


def test_sync_rejects_an_unparseable_since(home: Path, stub_source: StubSource) -> None:
    """Anything else is a usage error, not a silent full scan."""
    result = runner.invoke(cli.app, ["sync", "--since", "last tuesday"])
    assert result.exit_code == cli.EXIT_USAGE


def test_sync_honours_the_limit(home: Path, stub_source: StubSource) -> None:
    """--limit is forwarded to the source rather than applied after the fetch."""
    result = runner.invoke(cli.app, ["sync", "--limit", "5"])
    assert result.exit_code == 0
    assert stub_source.calls[0]["limit"] == 5


def test_sync_full_ignores_the_cursor(home: Path, stub_source: StubSource) -> None:
    """--full re-scans instead of resuming from the stored cursor."""
    runner.invoke(cli.app, ["sync"])
    runner.invoke(cli.app, ["sync", "--full"])
    assert stub_source.calls[1]["cursor"] is None


def test_sync_reports_a_classification_failure_without_aborting(
    home: Path, stub_source: StubSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unclassifiable message is a reported warning, not a failed sync."""

    class Failing:
        name = "broken"
        version = "0"

        def classify(self, message: RawMessage) -> Classification:
            raise ClassificationError("no verdict")

        def classify_batch(self, messages: list[RawMessage]) -> list[Classification]:
            return [self.classify(m) for m in messages]

    monkeypatch.setattr(cli, "_build_classifier", lambda config: Failing())

    result = runner.invoke(cli.app, ["sync"])

    assert result.exit_code == 0
    assert "no verdict" in result.output


def test_sync_surfaces_an_auth_failure(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing token reaches main and becomes exit 3."""

    def boom(config: object) -> None:
        raise AuthError("no token")

    monkeypatch.setattr(cli, "_build_source", boom)
    monkeypatch.setattr(cli, "app", cli.app)
    monkeypatch.setattr("sys.argv", ["jobtrack", "sync"])

    assert cli.main() == cli.EXIT_AUTH


# --- auth login -------------------------------------------------------------


def test_auth_login_runs_the_consent_flow(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Login delegates to the OAuth flow and reports where the token landed."""
    called: list[Path] = []
    monkeypatch.setattr(cli, "run_oauth_flow", lambda config: called.append(config.token_path))

    result = runner.invoke(cli.app, ["auth", "login"])

    assert result.exit_code == 0
    assert "authorized" in result.output
    assert called == [home / "token.json"]
