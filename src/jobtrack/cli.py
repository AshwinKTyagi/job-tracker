"""Typer command line for jobtrack — the composition root.

This is the only layer that wires modules together, the only layer that prints, and the
only layer that catches :class:`~jobtrack.errors.JobTrackError` and maps it to an exit
code. Library modules raise; ``main`` decides what that is worth to the shell.

Exit codes (PLAN.md §6)::

    0  ok
    1  unexpected failure
    2  usage or configuration error
    3  authentication failure
    4  transient network / quota failure — retry later
"""

from __future__ import annotations

import logging
import statistics
import webbrowser
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Final

import typer
from rich.console import Console
from rich.table import Table

from jobtrack.classify import CompositeClassifier, RulesClassifier
from jobtrack.classify.base import Classifier
from jobtrack.config import Config, load_config
from jobtrack.errors import (
    AuthError,
    ClassificationError,
    ConfigError,
    JobTrackError,
    TransientIngestError,
)
from jobtrack.export import build_dataframe, build_events_dataframe, write_csv, write_xlsx
from jobtrack.ingest.auth import credential_status, load_credentials, run_oauth_flow
from jobtrack.ingest.gmail import GmailSource
from jobtrack.ingest.source import EmailSource
from jobtrack.models import ApplicationStatus, EventType, Override, SyncReport
from jobtrack.store import Store
from jobtrack.viz.dashboard import build_dashboard

logger = logging.getLogger(__name__)

app = typer.Typer(name="jobtrack", help="Track job applications from your Gmail inbox.")

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_AUTH, EXIT_TRANSIENT = 0, 1, 2, 3, 4

#: Longest body excerpt shown for one message in the review queue.
REVIEW_EXCERPT_CHARS: Final[int] = 400

#: Filenames used when a command is given no explicit ``-o`` path.
DEFAULT_EXPORT_STEM: Final[str] = "applications"
DEFAULT_DASHBOARD_NAME: Final[str] = "dashboard.html"

_VALID_EXPORT_FORMATS: Final[frozenset[str]] = frozenset({"csv", "xlsx"})

#: Single-letter answers accepted by the review prompt.
_REVIEW_ACTIONS: Final[frozenset[str]] = frozenset({"a", "c", "s", "q"})

console = Console()


# --- shared plumbing --------------------------------------------------------


def _configure_logging(verbose: bool = False) -> None:
    """Send library logs to stderr at WARNING, or INFO when verbose."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _load_config() -> Config:
    """Resolve runtime configuration, letting ConfigError reach ``main``."""
    return load_config()


def _open_store(config: Config) -> Store:
    """Open the configured database and bring it up to the current schema.

    Args:
        config: Resolved runtime configuration.

    Returns:
        An open, migrated store.

    Raises:
        StoreError: the database could not be opened.
        MigrationError: a pending migration failed.
    """
    store = Store.open_from_config(config)
    store.migrate()
    return store


def _build_classifier(config: Config) -> Classifier:
    """Construct the configured classifier stack.

    Phase 1 ships rules-only; the composite wrapper is what M7 slots an Ollama backend
    into without touching any caller.

    Args:
        config: Resolved runtime configuration.

    Returns:
        The classifier the CLI should hand to ``run_sync``.
    """
    rules = RulesClassifier(min_confidence=config.classify.min_confidence)
    return CompositeClassifier(rules, None, min_confidence=config.classify.min_confidence)


def _build_source(config: Config) -> EmailSource:
    """Construct the Gmail source from stored credentials.

    Args:
        config: Resolved runtime configuration.

    Returns:
        A ready ``EmailSource``.

    Raises:
        AuthError: no token, or the stored token could not be refreshed.
    """
    return GmailSource(load_credentials(config))


def _parse_since(value: str | None) -> datetime | None:
    """Parse a ``--since`` value into a tz-aware UTC datetime.

    Accepts an ISO-8601 date or datetime (``2026-01-31``, ``2026-01-31T09:00:00``) or a
    bare day count (``30`` meaning "30 days ago").

    Args:
        value: The raw option text, or None.

    Returns:
        The lower bound, or None when no bound was given.

    Raises:
        typer.BadParameter: the value is neither a day count nor an ISO-8601 timestamp.
    """
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return datetime.now(UTC) - timedelta(days=int(text))
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{value!r} is neither a day count nor an ISO-8601 date", param_hint="--since"
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _parse_status(value: str | None) -> ApplicationStatus | None:
    """Parse a ``--status`` value into an ApplicationStatus.

    Args:
        value: The raw option text, or None.

    Returns:
        The status, or None when no filter was given.

    Raises:
        typer.BadParameter: the value is not an ApplicationStatus member.
    """
    if value is None:
        return None
    try:
        return ApplicationStatus(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(sorted(member.value for member in ApplicationStatus))
        raise typer.BadParameter(
            f"{value!r} is not a status; expected one of: {allowed}", param_hint="--status"
        ) from exc


def _parse_event_type(value: str) -> EventType | None:
    """Parse a typed event-type correction, returning None for an empty answer."""
    text = value.strip().lower()
    if not text:
        return None
    try:
        return EventType(text)
    except ValueError:
        allowed = ", ".join(sorted(member.value for member in EventType))
        console.print(f"[red]not an event type[/red]; expected one of: {allowed}")
        return None


def _prompt_action() -> str:
    """Ask what to do with the queued message, re-asking until the answer is valid.

    Returns:
        One of ``"a"`` (accept), ``"c"`` (correct), ``"s"`` (skip), or ``"q"`` (quit).
    """
    while True:
        answer: str = typer.prompt("[a]ccept / [c]orrect / [s]kip / [q]uit", default="a")
        choice = answer.strip().lower()[:1]
        if choice in _REVIEW_ACTIONS:
            return choice
        console.print("[red]expected a, c, s, or q[/red]")


def _format_datetime(value: datetime | None) -> str:
    """Render a datetime as a short UTC date, or an em dash when absent."""
    return "—" if value is None else value.astimezone(UTC).strftime("%Y-%m-%d")


# --- sync orchestration -----------------------------------------------------


def run_sync(
    source: EmailSource,
    classifier: Classifier,
    store: Store,
    config: Config,
    *,
    now: datetime,
    dry_run: bool = False,
    limit: int | None = None,
    since: datetime | None = None,
    full: bool = False,
) -> SyncReport:
    """Orchestrate one sync: fetch → dedupe → classify → link → record → advance cursor.

    Takes every collaborator as a parameter so the e2e test can drive it with a fake
    source, a tmp_path store, and a frozen clock. ``dry_run`` classifies and reports but
    writes nothing, including the cursor.

    Per invariant I1 a message already in the store is skipped outright, which is what
    makes a re-run — or a crash mid-batch — a no-op. Per I9 the cursor advances only
    after every message in the batch has committed.

    Args:
        source: The mailbox to pull from.
        classifier: The classifier applied to each new message.
        store: The destination store, already migrated.
        config: Resolved runtime configuration, supplying query and lookback defaults.
        now: Injected tz-aware UTC clock.
        dry_run: Classify and count, but write nothing.
        limit: Cap on messages fetched. Defaults to ``config.gmail.max_per_sync``.
        since: Explicit lower bound for the dated fetch path. Defaults to
            ``config.gmail.lookback_days`` before ``now`` when there is no cursor.
        full: Ignore any stored cursor and re-scan the lookback window.

    Returns:
        Counts describing what the sync did, plus any per-message errors it survived.

    Raises:
        AuthError: credentials are missing, expired, or revoked.
        TransientIngestError: rate limited or a 5xx — worth retrying.
        StoreError: a write failed.
    """
    cursor = None if full else store.get_cursor(source.name)
    effective_since = since
    if effective_since is None and cursor is None:
        effective_since = now - timedelta(days=config.gmail.lookback_days)
    effective_limit = config.gmail.max_per_sync if limit is None else limit

    result = source.fetch(
        query=config.gmail.query,
        since=effective_since,
        cursor=cursor,
        limit=effective_limit,
    )

    known_applications = {row.application_id for row in store.list_applications(now=now)}
    errors: list[str] = []
    new_messages = 0
    events_created = 0
    applications_created = 0
    needs_review = 0
    unknown = 0

    for message in result.messages:
        if store.has_message(message.message_id):
            continue
        new_messages += 1
        try:
            classification = classifier.classify(message)
        except ClassificationError as exc:
            errors.append(f"{message.message_id}: {exc}")
            continue

        if classification.event_type is EventType.UNKNOWN:
            unknown += 1
        if classification.needs_review:
            needs_review += 1
        if dry_run:
            continue

        store.record_message(message)
        store.record_classification(classification)
        event = store.link_and_record_event(message, classification, now=now)
        events_created += 1
        if event.application_id is not None and event.application_id not in known_applications:
            known_applications.add(event.application_id)
            applications_created += 1

    if not dry_run and result.next_cursor is not None:
        store.set_cursor(source.name, result.next_cursor, synced_at=now)

    if result.truncated:
        errors.append(f"batch truncated at limit={effective_limit}; more messages remain")

    return SyncReport(
        fetched=len(result.messages),
        new_messages=new_messages,
        events_created=events_created,
        applications_created=applications_created,
        needs_review=needs_review,
        unknown=unknown,
        errors=errors,
        started_at=now,
        finished_at=now,
    )


def _render_sync_report(report: SyncReport, *, dry_run: bool) -> None:
    """Print a sync summary table."""
    table = Table(title="dry run — nothing written" if dry_run else "sync complete")
    table.add_column("metric")
    table.add_column("count", justify="right")
    table.add_row("fetched", str(report.fetched))
    table.add_row("new messages", str(report.new_messages))
    table.add_row("events created", str(report.events_created))
    table.add_row("applications created", str(report.applications_created))
    table.add_row("needs review", str(report.needs_review))
    table.add_row("unknown", str(report.unknown))
    console.print(table)
    for problem in report.errors:
        console.print(f"[yellow]warning[/yellow] {problem}")
    if report.needs_review:
        console.print(f"run [bold]jobtrack review[/bold] to resolve {report.needs_review} item(s)")


# --- commands ---------------------------------------------------------------


@app.command()
def sync(
    since: Annotated[str | None, typer.Option(help="ISO date or day count to fetch from.")] = None,
    full: Annotated[bool, typer.Option(help="Ignore the stored cursor and re-scan.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Classify and report, write nothing.")
    ] = False,
    limit: Annotated[int | None, typer.Option(help="Cap the number of messages fetched.")] = None,
) -> None:
    """Fetch new mail, classify it, and record the resulting events."""
    config = _load_config()
    lower_bound = _parse_since(since)
    with _open_store(config) as store:
        report = run_sync(
            _build_source(config),
            _build_classifier(config),
            store,
            config,
            now=datetime.now(UTC),
            dry_run=dry_run,
            limit=limit,
            since=lower_bound,
            full=full,
        )
    _render_sync_report(report, dry_run=dry_run)


@app.command()
def review(
    limit: Annotated[int, typer.Option(help="How many queued messages to walk.")] = 20,
) -> None:
    """Walk the low-confidence queue, accepting or correcting each guess.

    Shows the subject, sender, and a body excerpt alongside the classifier's guess and
    the stable rule ids that produced it, then takes an accept / correct / skip / quit
    answer. Corrections become ``Override`` rows and survive a later reclassify (I6).
    """
    config = _load_config()
    with _open_store(config) as store:
        queue = store.pending_review(limit)
        if not queue:
            console.print("[green]review queue is empty[/green]")
            return

        for index, item in enumerate(queue, start=1):
            guess = item.classification
            console.rule(f"[{index}/{len(queue)}] {item.message.subject or '(no subject)'}")
            console.print(f"from: {item.message.from_email}")
            console.print(f"received: {_format_datetime(item.message.received_at)}")
            console.print(f"guess: [bold]{guess.event_type.value}[/bold] ({guess.confidence:.2f})")
            console.print(f"company: {guess.company or '—'}   role: {guess.role or '—'}")
            console.print(f"evidence: {', '.join(guess.evidence) or '—'}")
            excerpt = item.message.body_text[:REVIEW_EXCERPT_CHARS].strip()
            if excerpt:
                console.print(f"\n{excerpt}\n")

            answer = _prompt_action()
            if answer == "q":
                break
            if answer == "s":
                continue
            if answer == "a":
                store.accept_classification(item.message.message_id, now=datetime.now(UTC))
                console.print("[green]accepted[/green]")
                continue

            event_type: str = typer.prompt("event type", default="")
            company: str = typer.prompt("company", default="")
            role: str = typer.prompt("role", default="")
            note: str = typer.prompt("note", default="")
            store.set_override(
                Override(
                    message_id=item.message.message_id,
                    event_type=_parse_event_type(event_type),
                    company=company.strip() or None,
                    role=role.strip() or None,
                    corrected_at=datetime.now(UTC),
                    note=note.strip() or None,
                )
            )
            console.print("[green]correction saved[/green]")


@app.command()
def reclassify(
    # `all` shadows a builtin, but CONTRACTS.md §9 freezes the name and it is the --all flag.
    all: Annotated[  # noqa: A002
        bool, typer.Option("--all", help="Also re-run messages a human reviewed.")
    ] = False,
) -> None:
    """Re-run the classifier over stored messages after a rules change.

    Human corrections are preserved by default (I6); ``--all`` discards them too.
    """
    config = _load_config()
    classifier = _build_classifier(config)
    now = datetime.now(UTC)
    with _open_store(config) as store:
        cleared = store.clear_classifications(only_unreviewed=not all)
        messages = store.list_messages()
        for message in messages:
            store.reapply_classification(message, classifier.classify(message), now=now)
        flagged = sum(1 for row in store.list_applications(now=now) if row.needs_review)
    console.print(f"cleared {cleared} classification(s), reclassified {len(messages)} message(s)")
    if flagged:
        console.print(f"run [bold]jobtrack review[/bold] to resolve {flagged} flagged item(s)")


@app.command("list")
def list_applications(
    status: Annotated[str | None, typer.Option(help="Filter by derived status.")] = None,
    company: Annotated[
        str | None, typer.Option(help="Filter by company (substring, normalized).")
    ] = None,
) -> None:
    """Print a table of applications with their derived status."""
    config = _load_config()
    wanted = _parse_status(status)
    with _open_store(config) as store:
        rows = store.list_applications(now=datetime.now(UTC), status=wanted, company=company)

    if not rows:
        console.print("[yellow]no applications match[/yellow]")
        return

    table = Table(title=f"{len(rows)} application(s)")
    for column in ("company", "role", "status", "applied", "last event", "events"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.company,
            row.role or "—",
            row.status.value,
            _format_datetime(row.applied_at),
            f"{_format_datetime(row.last_event_at)} ({row.last_event_type.value})",
            str(row.event_count),
        )
    console.print(table)


@app.command()
def stats() -> None:
    """Print counts, response rate, and median time-to-response."""
    config = _load_config()
    with _open_store(config) as store:
        rows = store.list_applications(now=datetime.now(UTC))

    if not rows:
        console.print("[yellow]no applications yet — run jobtrack sync[/yellow]")
        return

    responses = [
        row.days_to_first_response for row in rows if row.days_to_first_response is not None
    ]
    by_status = Counter(row.status.value for row in rows)

    table = Table(title=f"{len(rows)} application(s)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("response rate", f"{len(responses) / len(rows):.0%}")
    table.add_row(
        "median days to response",
        f"{statistics.median(responses):.1f}" if responses else "—",
    )
    table.add_row("needs review", str(sum(1 for row in rows if row.needs_review)))
    for name, count in sorted(by_status.items()):
        table.add_row(name, str(count))
    console.print(table)


@app.command()
def export(
    # `format` shadows a builtin, but CONTRACTS.md §9 freezes the name and it is --format.
    format: Annotated[  # noqa: A002
        str, typer.Option(help="csv or xlsx; defaults to the configured format.")
    ] = "",
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Destination file.")] = None,
) -> None:
    """Write a spreadsheet snapshot of every application."""
    config = _load_config()
    chosen = (format or config.export.default_format).strip().lower()
    if chosen not in _VALID_EXPORT_FORMATS:
        raise typer.BadParameter(f"{chosen!r} is not csv or xlsx", param_hint="--format")

    destination = output or config.home / f"{DEFAULT_EXPORT_STEM}.{chosen}"
    with _open_store(config) as store:
        now = datetime.now(UTC)
        applications = build_dataframe(store.list_applications(now=now))
        events = build_events_dataframe(store.list_events())

    written = (
        write_csv(applications, destination)
        if chosen == "csv"
        else write_xlsx(applications, destination, events=events)
    )
    console.print(f"wrote [bold]{written}[/bold]")


@app.command()
def dashboard(
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Destination .html file.")
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the file when it is written.")
    ] = False,
) -> None:
    """Write the self-contained Plotly dashboard."""
    config = _load_config()
    destination = output or config.home / DEFAULT_DASHBOARD_NAME
    with _open_store(config) as store:
        now = datetime.now(UTC)
        applications = build_dataframe(store.list_applications(now=now))
        events = build_events_dataframe(store.list_events())

    written = build_dashboard(applications, events, destination)
    console.print(f"wrote [bold]{written}[/bold]")
    if open_browser:
        webbrowser.open(written.as_uri())


auth_app = typer.Typer(help="Manage Gmail credentials.")
app.add_typer(auth_app, name="auth")


@auth_app.command("login")
def auth_login() -> None:
    """Run the OAuth consent flow and store a read-only Gmail token."""
    config = _load_config()
    run_oauth_flow(config)
    console.print(f"[green]authorized[/green]; token stored at {config.token_path}")


@auth_app.command("status")
def auth_status() -> None:
    """Report token presence, expiry, and granted scopes."""
    config = _load_config()
    status = credential_status(config)

    table = Table(title="gmail credentials")
    table.add_column("field")
    table.add_column("value")
    for key in (
        "credentials_path",
        "token_path",
        "has_client_secrets",
        "has_token",
        "valid",
        "expired",
        "expiry",
        "has_refresh_token",
        "scopes_ok",
        "error",
    ):
        table.add_row(key, str(status[key]))
    table.add_row("scopes", ", ".join(status["scopes"]) or "—")
    console.print(table)

    if not status["valid"]:
        console.print("[yellow]run [bold]jobtrack auth login[/bold] to authorize[/yellow]")


db_app = typer.Typer(help="Database maintenance.")
app.add_typer(db_app, name="db")


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply any pending schema migrations."""
    config = _load_config()
    with _open_store(config) as store:
        console.print(f"schema at version [bold]{store.schema_version()}[/bold]")


# --- entry point ------------------------------------------------------------


def main() -> int:
    """Entry point. The ONLY place that catches JobTrackError and maps it to an exit code.

    Returns:
        A shell exit code: 0 ok, 1 unexpected, 2 usage or configuration, 3 auth,
        4 transient network or quota.
    """
    _configure_logging()
    try:
        app()
    except SystemExit as exc:
        # Typer already rendered its own message and picked the code — a usage error is
        # its UsageError.exit_code, which is 2, the same EXIT_USAGE this module documents.
        if exc.code is None:
            return EXIT_OK
        return exc.code if isinstance(exc.code, int) else EXIT_ERROR
    except AuthError as exc:
        console.print(f"[red]auth error[/red] {exc}")
        console.print("run [bold]jobtrack auth login[/bold]")
        return EXIT_AUTH
    except TransientIngestError as exc:
        console.print(f"[red]transient failure[/red] {exc}; retry later")
        return EXIT_TRANSIENT
    except ConfigError as exc:
        console.print(f"[red]configuration error[/red] {exc}")
        return EXIT_USAGE
    except JobTrackError as exc:
        console.print(f"[red]error[/red] {exc}")
        return EXIT_ERROR
    return EXIT_OK


__all__ = [
    "EXIT_AUTH",
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_TRANSIENT",
    "EXIT_USAGE",
    "app",
    "main",
    "run_sync",
]
