# job-tracker — interfaces and type contracts

Frozen. Implement byte-identically. `...` bodies are yours to fill; signatures are not yours to
change. If a contract blocks you, report it — do not edit it.

## 1. Shared types — `src/jobtrack/models.py` (M0 owns)

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """A single observed step in an application's lifecycle.

    Declaration order is NOT precedence order; see constants.EVENT_PRECEDENCE.

    StrEnum, NOT (str, Enum): str(EventType.REJECTION) must yield "rejection", not
    "EventType.REJECTION". Store and export interpolate these directly, so the
    difference is a silent data-corruption bug rather than a style preference.
    """
    APPLICATION_RECEIVED = "application_received"
    ASSESSMENT           = "assessment"           # OA, take-home, coding challenge
    INTERVIEW            = "interview"            # screen, onsite, scheduling
    OFFER                = "offer"
    REJECTION            = "rejection"
    WITHDRAWN            = "withdrawn"            # candidate-initiated
    RECRUITER_OUTREACH   = "recruiter_outreach"   # inbound, no application yet
    UNKNOWN              = "unknown"              # not job-related, or unparseable


class ApplicationStatus(StrEnum):
    """DERIVED from an application's event history. Never persisted as a column."""
    APPLIED     = "applied"
    ASSESSMENT  = "assessment"
    INTERVIEWING = "interviewing"
    OFFER       = "offer"
    REJECTED    = "rejected"
    WITHDRAWN   = "withdrawn"
    GHOSTED     = "ghosted"      # non-terminal, no event for config.store.ghost_after_days


class RawMessage(BaseModel):
    """A normalized email. The only thing M1 produces and the only thing M2 consumes.

    body_text is plain text with HTML already stripped and whitespace collapsed.
    headers keys are lowercased.
    """
    model_config = ConfigDict(frozen=True)

    message_id: str                      # Gmail message id — the global dedupe key
    thread_id: str
    received_at: datetime                # tz-aware UTC
    from_email: str                      # lowercased
    from_name: str | None = None
    to_email: str | None = None
    subject: str = ""
    body_text: str = ""
    snippet: str = ""
    labels: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)


class Classification(BaseModel):
    """M2's output. Fully determined by the RawMessage — no I/O, no clock, no randomness."""
    model_config = ConfigDict(frozen=True)

    message_id: str
    event_type: EventType
    company: str | None = None           # display form, verbatim from the email
    company_key: str | None = None       # normalize_company(company); the matching key
    role: str | None = None
    location: str | None = None
    ats: str | None = None               # "greenhouse" | "lever" | ... | None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False
    evidence: list[str] = Field(default_factory=list)   # stable rule ids that fired
    classifier_name: str                 # "rules" | "ollama" | "composite"
    classifier_version: str              # semver for rules; prompt SHA-256 for LLM


class Override(BaseModel):
    """A human correction. Wins over any classifier output, survives reclassify."""
    message_id: str
    event_type: EventType | None = None
    company: str | None = None
    role: str | None = None
    corrected_at: datetime
    note: str | None = None


class EventRow(BaseModel):
    """One stored event, after overrides are applied."""
    event_id: int
    application_id: str | None           # None only for UNKNOWN / unlinked
    message_id: str
    event_type: EventType
    occurred_at: datetime
    confidence: float
    needs_review: bool
    is_overridden: bool
    subject: str
    from_email: str


class ApplicationRow(BaseModel):
    """One application with its derived status. The unit of the spreadsheet and the charts."""
    application_id: str
    company: str
    company_key: str
    role: str | None
    location: str | None
    ats: str | None
    status: ApplicationStatus            # derived
    applied_at: datetime
    last_event_at: datetime
    last_event_type: EventType
    event_count: int
    days_to_first_response: int | None   # applied → first non-APPLICATION_RECEIVED event
    days_since_last_event: int
    needs_review: bool
    source_thread_ids: list[str]


class ReviewItem(BaseModel):
    """One entry in the low-confidence queue."""
    message: RawMessage
    classification: Classification
    suggested_application_id: str | None


class ApplicationMatchCandidate(BaseModel):
    """Pre-fetched by the store, handed to the pure linker. Keeps matching testable."""
    application_id: str
    company_key: str
    role: str | None
    thread_ids: list[str]
    applied_at: datetime
    last_event_at: datetime


class StageFlow(BaseModel):
    """One Sankey link: `count` applications moved from `source` to `target`."""
    source: str
    target: str
    count: int


class SyncReport(BaseModel):
    """Returned by the sync orchestration so the CLI can print a summary."""
    fetched: int
    new_messages: int
    events_created: int
    applications_created: int
    needs_review: int
    unknown: int
    errors: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
```

### Frozen constants — `src/jobtrack/constants.py` (M0 owns)

```python
EVENT_PRECEDENCE: tuple[EventType, ...] = (
    EventType.WITHDRAWN,
    EventType.REJECTION,
    EventType.OFFER,
    EventType.INTERVIEW,
    EventType.ASSESSMENT,
    EventType.APPLICATION_RECEIVED,
    EventType.RECRUITER_OUTREACH,
    EventType.UNKNOWN,
)
"""Highest-precedence matched type wins. MUST cover every EventType — resolve_event_type
indexes into this, so a missing member is a crash. WITHDRAWN leads: it is explicit and
terminal. REJECTION outranks APPLICATION_RECEIVED because rejection emails restate the
application language ('Thanks for applying to X ... unfortunately')."""

TERMINAL_EVENTS: frozenset[EventType] = frozenset(
    {EventType.REJECTION, EventType.OFFER, EventType.WITHDRAWN}
)

EXPORT_COLUMNS: tuple[str, ...] = (
    "application_id", "company", "role", "location", "ats", "status",
    "applied_at", "last_event_at", "last_event_type", "event_count",
    "days_to_first_response", "days_since_last_event", "needs_review",
)
"""FROZEN. M4 emits exactly these, in this order. M5 reads exactly these. Do not reorder."""

DEFAULT_GMAIL_QUERY: str = (
    '-in:chats ('
    '"thank you for applying" OR "thanks for applying" OR "application received" OR '
    '"we received your application" OR "your application" OR "application status" OR '
    '"not moving forward" OR "other candidates" OR "unfortunately" OR '
    '"interview" OR "next steps" OR "coding challenge" OR "take-home" OR "assessment" OR '
    'from:greenhouse.io OR from:lever.co OR from:hire.lever.co OR from:myworkday.com OR '
    'from:ashbyhq.com OR from:smartrecruiters.com OR from:icims.com OR from:taleo.net OR '
    'from:jobvite.com OR from:workable.com OR from:breezy.hr OR from:bamboohr.com'
    ')'
)
"""Tuned for RECALL, not precision — the classifier is the real filter. Overridable in config."""
```

### Config — `src/jobtrack/config.py` (M0 owns)

```python
DEFAULT_JOBTRACK_HOME: Path = Path.home() / ".local" / "share" / "jobtrack"
"""Overridable by the JOBTRACK_HOME environment variable. Holds config.toml,
credentials.json, token.json, and jobtrack.db. NEVER inside the repo."""


class GmailConfig(BaseModel):
    query: str = DEFAULT_GMAIL_QUERY
    lookback_days: int = 400
    max_per_sync: int = 500


class ClassifyConfig(BaseModel):
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    backend: str = "rules"                  # Phase 3: "rules+ollama"
    ollama_model: str | None = None         # pinned tag; resolved to a digest at runtime
    ollama_host: str = "http://localhost:11434"


class StoreConfig(BaseModel):
    ghost_after_days: int = 30


class ExportConfig(BaseModel):
    default_format: str = "xlsx"            # "xlsx" | "csv"


class Config(BaseModel):
    """Fully-resolved runtime configuration. Constructed once in cli.py and passed
    explicitly to every collaborator — never read from a global or re-read from disk
    inside a library module.
    """
    home: Path = DEFAULT_JOBTRACK_HOME
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @property
    def db_path(self) -> Path: ...
    @property
    def credentials_path(self) -> Path: ...
    @property
    def token_path(self) -> Path: ...


def load_config(home: Path | None = None) -> Config:
    """Resolve JOBTRACK_HOME (argument > $JOBTRACK_HOME > default), read config.toml if
    present, and merge it over the defaults. A missing config.toml is NOT an error —
    every field has a usable default.

    Raises:
        ConfigError: config.toml is malformed, or home exists but is not writable.
    """
    ...
```

## 2. Errors — `src/jobtrack/errors.py` (M0 owns)

```python
class JobTrackError(Exception):
    """Base. Every module raises from this tree; cli.py is the only place that catches."""

class ConfigError(JobTrackError):        """Missing/malformed config or JOBTRACK_HOME."""
class AuthError(JobTrackError):          """No credentials, expired/revoked token. → exit 3."""
class IngestError(JobTrackError):        """Base for M1."""
class TransientIngestError(IngestError):  """429/5xx/timeout. Retry with backoff. → exit 4."""
class PermanentIngestError(IngestError):  """4xx that retrying will not fix."""
class ClassificationError(JobTrackError): """M2 could not produce a Classification at all."""
class StoreError(JobTrackError):         """Base for M3."""
class MigrationError(StoreError):        """Schema migration failed."""
class ExportError(JobTrackError):        """M4/M5 could not write output."""
```

**Contract.** Wrap third-party exceptions at your module boundary: no `HttpError` escapes
`ingest/`, no `sqlite3.Error` escapes `store/`. Never `sys.exit()` in a library module.

## 3. Cross-module invariants

These bind every module. Violating one is a bug even if your tests pass.

- **I1 — `message_id` is the universal dedupe key.** Processing the same message twice must
  produce no second event and no second application. `sync` is idempotent by construction.
- **I2 — the classifier is pure.** Same `RawMessage` in ⇒ byte-identical `Classification` out.
  No network, no DB, no `datetime.now()`, no randomness, no dict-ordering dependence.
- **I3 — precedence, not first-match.** Event typing scores all types and picks by
  `EVENT_PRECEDENCE`.
- **I4 — status is derived.** `ApplicationStatus` is computed from the event history on every
  read. It is never a stored column.
- **I5 — events are append-only.** Correcting a mistake writes an `Override`; it never mutates
  or deletes an event row.
- **I6 — overrides win, always.** Applied at read time, after classification. `reclassify` must
  never clobber one.
- **I7 — all datetimes are tz-aware UTC** in memory and ISO-8601 UTC on disk. A naive datetime
  crossing a module boundary is a bug.
- **I8 — matching uses `company_key`, display uses `company`.** `normalize_company()` is the sole
  producer of `company_key`, and it is deterministic and idempotent:
  `normalize_company(normalize_company(x)) == normalize_company(x)`.
- **I9 — the cursor advances only after the batch commits.** A crash mid-sync re-fetches; I1
  makes the replay harmless.
- **I10 — `EXPORT_COLUMNS` is the M4↔M5 wire format.** Neither side reorders, renames, or drops.
- **I11 — read-only Gmail.** No code path may call a mutating Gmail method.

## 4. M1 ingest — `src/jobtrack/ingest/`

```python
# ingest/source.py
class FetchResult(BaseModel):
    messages: list[RawMessage]
    next_cursor: str | None       # opaque; Gmail historyId for GmailSource
    fetched_at: datetime
    truncated: bool = False       # True if capped by limit — more remains

class EmailSource(Protocol):
    """Any mailbox that can yield RawMessages. Implement to add a provider."""
    name: str

    def fetch(
        self,
        *,
        query: str,
        since: datetime | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> FetchResult:
        """Fetch messages matching `query`.

        Prefers an incremental delta from `cursor` when the provider supports it and the
        cursor is still valid; otherwise falls back to a dated query from `since`.

        Raises:
            TransientIngestError: rate limited, 5xx, or timed out — retry with backoff.
            PermanentIngestError: malformed query or unrecoverable 4xx.
            AuthError: credentials missing, expired, or revoked.
        """
        ...


# ingest/auth.py
GMAIL_SCOPES: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]
"""Read-only. Never widen this list. (I11)"""

def load_credentials(config: Config) -> Credentials:
    """Load and silently refresh the stored OAuth token.

    Raises:
        AuthError: no token.json, or refresh failed (revoked/expired).
    """
    ...

def run_oauth_flow(config: Config) -> Credentials:
    """Run the interactive desktop OAuth consent flow and persist token.json (mode 0600).

    Raises:
        AuthError: credentials.json missing or the user declined consent.
    """
    ...

def credential_status(config: Config) -> dict[str, Any]:
    """Report token presence, expiry, and granted scopes for `jobtrack auth status`."""
    ...


# ingest/gmail.py
class GmailSource:
    """EmailSource backed by the Gmail API.

    Owns: pagination, exponential backoff on 429/5xx, historyId delta sync with a documented
    fallback to a dated query when history.list 404s (deltas expire after ~1 week), and the
    Gmail-payload → RawMessage transform.
    """
    name = "gmail"

    def __init__(self, credentials: Credentials, *, service: Any | None = None) -> None:
        """Args:
            credentials: from auth.load_credentials.
            service: injected googleapiclient resource. Tests pass a fake here — this
                parameter is the reason ingest is testable without a network.
        """
        ...

    def fetch(self, *, query: str, since: datetime | None = None,
              cursor: str | None = None, limit: int | None = None) -> FetchResult: ...

def parse_gmail_message(payload: dict[str, Any]) -> RawMessage:
    """Convert a raw `users.messages.get(format='full')` payload into a RawMessage.

    Walks the MIME tree preferring text/plain, falling back to html_to_text(text/html).
    Lowercases header keys and the sender address; normalizes `internalDate` to UTC.

    Raises:
        PermanentIngestError: payload is missing id, threadId, or internalDate.
    """
    ...


# ingest/html.py
def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text.

    Drops script/style/head, converts <br> and block ends to newlines, unescapes entities,
    collapses runs of whitespace. Deterministic — M2's purity (I2) depends on it.
    """
    ...
```

## 5. M2 classify — `src/jobtrack/classify/`

```python
# classify/base.py
class Classifier(Protocol):
    """RawMessage → Classification. THE plug point for a future Ollama backend.

    Implementations must be deterministic (I2) and must never raise for ordinary input —
    an unparseable message is EventType.UNKNOWN with confidence 0.0, not an exception.
    """
    name: str
    version: str

    def classify(self, message: RawMessage) -> Classification: ...

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Order-preserving. Default implementation may map classify() over the input."""
        ...


class CompositeClassifier:
    """Primary classifier with a fallback for low-confidence results.

    Phase 1 ships this with fallback=None (a pass-through). Phase 3 constructs it as
    CompositeClassifier(RulesClassifier(), OllamaClassifier(), min_confidence=0.60) —
    and no caller changes.
    """
    def __init__(self, primary: Classifier, fallback: Classifier | None = None,
                 *, min_confidence: float = 0.60) -> None: ...
    def classify(self, message: RawMessage) -> Classification: ...
    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]: ...


# classify/rules.py
class RulesClassifier:
    """Deterministic pattern-based classifier. No I/O, no clock, no randomness.

    Pipeline: detect_ats → score every EventType → resolve by EVENT_PRECEDENCE →
    extract company/role/location → score_confidence.
    """
    name = "rules"
    version = "1.0.0"          # bump on any pattern change

    def classify(self, message: RawMessage) -> Classification: ...
    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]: ...


def detect_ats(message: RawMessage) -> tuple[str | None, list[str]]:
    """Identify the applicant-tracking system from sender, Reply-To, and List-Unsubscribe.

    Returns:
        (ats_slug or None, rule_ids that fired) e.g. ("greenhouse", ["ats.sender.greenhouse"]).
    """
    ...

def score_event_types(message: RawMessage) -> dict[EventType, list[str]]:
    """Score the message against EVERY event type — never stop at the first hit (I3).

    Returns:
        Mapping of each matched EventType to the rule ids that fired for it.
        Empty dict means nothing matched.
    """
    ...

def resolve_event_type(scores: dict[EventType, list[str]]) -> tuple[EventType, list[str]]:
    """Pick the winner by EVENT_PRECEDENCE.

    Returns:
        (winning type, its rule ids). (UNKNOWN, []) when scores is empty.
    """
    ...

def extract_company(message: RawMessage, ats: str | None) -> tuple[str | None, list[str]]:
    """Ordered extraction chain: ATS-specific header/sender → subject capture group →
    body signature → sender display name. Returns (display name, rule ids)."""
    ...

def extract_role(message: RawMessage, ats: str | None) -> tuple[str | None, list[str]]: ...
def extract_location(message: RawMessage) -> str | None: ...


# classify/normalize.py
def normalize_company(name: str | None) -> str | None:
    """Produce the stable matching key for a company (I8).

    Casefolds, strips legal suffixes (Inc, LLC, Ltd, Corp, GmbH, PBC, Co), drops punctuation,
    collapses whitespace. Deterministic and idempotent.

    >>> normalize_company("Acme Robotics, Inc.") == normalize_company("acme robotics")
    True
    """
    ...

def normalize_role(title: str | None) -> str | None:
    """Canonicalize a job title for fuzzy comparison: casefold, strip seniority noise and
    req ids, expand common abbreviations (SWE → software engineer)."""
    ...

def role_similarity(a: str | None, b: str | None) -> float:
    """Similarity in [0,1] between two normalized titles. Used by the linker's fuzzy match.
    Deterministic; no external NLP dependency."""
    ...


# classify/confidence.py
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "ats_detected":        0.35,
    "subject_pattern":     0.40,
    "body_pattern":        0.20,
    "company_extracted":   0.05,
    "ambiguous_penalty":  -0.20,
}
"""The rubric lives HERE, as one table. No magic numbers scattered through rules.py."""

def score_confidence(
    *,
    ats: str | None,
    winning_type: EventType,
    evidence: list[str],
    company: str | None,
    all_scores: dict[EventType, list[str]],
) -> float:
    """Additive rubric from CONFIDENCE_WEIGHTS, clamped to [0.0, 1.0].

    Applies ambiguous_penalty when two adjacent-precedence types both matched.
    """
    ...

def needs_review(confidence: float, company: str | None, *, threshold: float) -> bool:
    """True when confidence < threshold or company is None."""
    ...
```

## 6. M3 store — `src/jobtrack/store/`

```python
# store/db.py
SCHEMA_VERSION: int = 1

class Store:
    """The only writer of SQLite. The only module that knows SQL.

    Tables: messages · classifications · applications · events · overrides · sync_state ·
    schema_version. Parameterized queries only; every column named explicitly.
    """

    @classmethod
    def open(cls, path: Path) -> Store:
        """Open (creating parent dirs as needed) with foreign_keys=ON and WAL enabled.

        Raises:
            StoreError: the file exists but is not a readable jobtrack database.
        """
        ...

    def migrate(self) -> None:
        """Apply pending migrations in order, in a transaction.

        Raises:
            MigrationError: a migration failed; the DB is left at its prior version.
        """
        ...

    def close(self) -> None: ...
    def __enter__(self) -> Store: ...
    def __exit__(self, *exc: object) -> None: ...

    # --- ingest side -------------------------------------------------------
    def has_message(self, message_id: str) -> bool:
        """Dedupe check (I1). Cheap — indexed primary key lookup."""
        ...

    def record_message(self, message: RawMessage) -> None:
        """Persist raw metadata. Idempotent: a second call for the same id is a no-op."""
        ...

    def record_classification(self, classification: Classification) -> None:
        """Upsert on message_id. Replaces a prior classification (reclassify); never touches
        overrides (I6)."""
        ...

    def link_and_record_event(
        self, message: RawMessage, classification: Classification, *, now: datetime
    ) -> EventRow:
        """Link the message to an application (creating one if needed) and append its event.

        Fetches candidates, delegates the decision to linker.match_application, and creates a
        new application when there is no match. UNKNOWN classifications are recorded with
        application_id=None. Idempotent on message_id (I1). Append-only (I5).

        Raises:
            StoreError: the write failed; the transaction is rolled back.
        """
        ...

    # --- read side ---------------------------------------------------------
    def get_application(self, application_id: str, *, now: datetime) -> ApplicationRow | None: ...

    def list_applications(
        self, *, now: datetime, status: ApplicationStatus | None = None,
        company: str | None = None, needs_review: bool | None = None,
    ) -> list[ApplicationRow]:
        """All applications with derived status (I4), overrides applied (I6).
        `company` matches against company_key, so it is normalization-insensitive."""
        ...

    def list_events(self, application_id: str | None = None) -> list[EventRow]:
        """Events, oldest first. None returns all, including unlinked UNKNOWNs."""
        ...

    def match_candidates(self, company_key: str, thread_id: str, *, within_days: int
                         ) -> list[ApplicationMatchCandidate]:
        """Fetch linking candidates: same thread_id, or same company_key within the window."""
        ...

    # --- review / overrides -------------------------------------------------
    def pending_review(self, limit: int | None = None) -> list[ReviewItem]:
        """Messages flagged needs_review that have no override yet, oldest first."""
        ...

    def set_override(self, override: Override) -> None:
        """Record a human correction and re-derive affected applications.
        Upsert on message_id. Survives reclassify (I6)."""
        ...

    def accept_classification(self, message_id: str, *, now: datetime) -> None:
        """Confirm the classifier was right: clears needs_review without altering fields.
        Still recorded as labeled data for future eval."""
        ...

    def clear_classifications(self, *, only_unreviewed: bool = True) -> int:
        """Drop stored classifications ahead of a reclassify. Returns rows cleared.
        Never deletes messages, applications, or overrides."""
        ...

    # --- sync state ---------------------------------------------------------
    def get_cursor(self, source: str) -> str | None: ...
    def set_cursor(self, source: str, cursor: str | None, *, synced_at: datetime) -> None:
        """Persist the cursor. Callers must only call this AFTER the batch commits (I9)."""
        ...


# store/linker.py
LINK_WINDOW_DAYS: int = 180

def match_application(
    classification: Classification,
    candidates: Sequence[ApplicationMatchCandidate],
    message_thread_id: str,
    *,
    now: datetime,
    window_days: int = LINK_WINDOW_DAYS,
    role_threshold: float = 0.75,
) -> str | None:
    """Decide which existing application a message belongs to. PURE — candidates are
    pre-fetched so this is unit-testable with no DB.

    Ordered rules:
        1. thread_id already belongs to an application  → that application.
        2. same company_key, role_similarity >= role_threshold, within window_days → that one
           (most recent last_event_at wins on ties).
        3. same company_key and either role is None, within window_days → that one.
        4. otherwise → None (caller creates a new application).

    Returns:
        application_id, or None to create a new one.
    """
    ...

def derive_status(events: Sequence[EventRow], *, now: datetime, ghost_after_days: int
                  ) -> ApplicationStatus:
    """Compute status from event history (I4).

    Terminal events (REJECTION/OFFER/WITHDRAWN) win regardless of recency — the most recent
    terminal event decides. Otherwise the furthest stage reached decides, downgraded to GHOSTED
    when the last event is older than ghost_after_days.
    """
    ...
```

### Schema sketch (M3 owns the real DDL)

```sql
messages(message_id PK, thread_id, received_at, from_email, from_name, to_email,
         subject, body_text, snippet, labels_json, headers_json, ingested_at)
classifications(message_id PK REFERENCES messages, event_type, company, company_key, role,
         location, ats, confidence, needs_review, evidence_json,
         classifier_name, classifier_version, classified_at)
applications(application_id PK, company, company_key, role, location, ats,
         applied_at, created_at)                      -- NO status column (I4)
events(event_id PK AUTOINCREMENT, application_id NULL REFERENCES applications,
       message_id UNIQUE REFERENCES messages, event_type, occurred_at, created_at)
overrides(message_id PK REFERENCES messages, event_type, company, role, corrected_at, note)
sync_state(source PK, cursor, last_synced_at)
schema_version(version PK, applied_at)

CREATE INDEX idx_apps_company_key ON applications(company_key);
CREATE INDEX idx_events_app       ON events(application_id, occurred_at);
CREATE INDEX idx_cls_review       ON classifications(needs_review);
```

## 7. M4 export — `src/jobtrack/export/tabular.py`

```python
def build_dataframe(applications: Sequence[ApplicationRow]) -> pd.DataFrame:
    """Applications → DataFrame with exactly EXPORT_COLUMNS, in order (I10).

    Datetimes are tz-aware UTC. Empty input yields an empty frame with the correct columns
    and dtypes — downstream code must never special-case emptiness.
    """
    ...

def build_events_dataframe(events: Sequence[EventRow]) -> pd.DataFrame:
    """Events → long-format DataFrame. Columns:
    application_id, message_id, event_type, occurred_at, confidence, needs_review, subject."""
    ...

def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write UTF-8 CSV (ISO-8601 dates, no index). Returns the resolved path.

    Raises:
        ExportError: path is not writable.
    """
    ...

def write_xlsx(df: pd.DataFrame, path: Path, *, events: pd.DataFrame | None = None) -> Path:
    """Write an .xlsx via openpyxl: sheet 'Applications', plus 'Events' when provided.

    Freezes the header row, autosizes columns, and adds an autofilter.

    Raises:
        ExportError: path not writable or openpyxl failed.
    """
    ...
```

## 8. M5 viz — `src/jobtrack/viz/`

```python
# viz/charts.py — each returns a configured Figure; none writes a file, none touches SQLite.
def status_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Application count by ApplicationStatus, ordered by pipeline stage (not alphabetically)."""
    ...

def applications_over_time(df: pd.DataFrame, *, freq: str = "W") -> go.Figure:
    """Applications submitted per period, from applied_at."""
    ...

def top_companies_bar(df: pd.DataFrame, *, top_n: int = 20) -> go.Figure:
    """Companies by application count, descending, stacked by status."""
    ...

def response_time_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of days_to_first_response, excluding nulls."""
    ...

def compute_stage_flows(events_df: pd.DataFrame, applications_df: pd.DataFrame) -> list[StageFlow]:
    """Turn per-application event sequences into Sankey links. PURE — unit-test this directly.

    Each application contributes consecutive transitions through the stages it actually reached
    (Applied → Assessment → Interview → Offer/Rejected). Applications that stalled contribute a
    terminal link into 'Ghosted'. Zero-count links are dropped.
    """
    ...

def funnel_sankey(flows: Sequence[StageFlow]) -> go.Figure:
    """Render stage flows as a plotly.graph_objects.Sankey.

    Node order follows pipeline stage; terminal nodes (Rejected/Ghosted/Offer) sit rightmost.
    """
    ...


# viz/dashboard.py
def build_dashboard(
    applications_df: pd.DataFrame,
    events_df: pd.DataFrame,
    path: Path,
    *,
    title: str = "Job Application Tracker",
) -> Path:
    """Compose every figure into ONE self-contained HTML file (plotly.js inlined — the file
    must render with no network). Includes a summary header (totals, response rate, median
    time-to-response). Returns the resolved path.

    An empty DataFrame renders an explanatory placeholder, not a traceback.

    Raises:
        ExportError: path is not writable.
    """
    ...
```

## 9. M6 cli — `src/jobtrack/cli.py`

```python
app = typer.Typer(name="jobtrack", help="Track job applications from your Gmail inbox.")

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_AUTH, EXIT_TRANSIENT = 0, 1, 2, 3, 4

def run_sync(
    source: EmailSource, classifier: Classifier, store: Store, config: Config,
    *, now: datetime, dry_run: bool = False, limit: int | None = None,
) -> SyncReport:
    """Orchestrate one sync: fetch → dedupe → classify → link → record → advance cursor.

    Takes every collaborator as a parameter so the e2e test can drive it with a fake source,
    a tmp_path store, and a frozen clock. dry_run classifies and reports but writes nothing,
    including the cursor.

    Raises:
        AuthError, TransientIngestError, StoreError: propagated to main() for exit-code mapping.
    """
    ...

@app.command()
def sync(since: str | None = None, full: bool = False,
         dry_run: bool = False, limit: int | None = None) -> None: ...

@app.command()
def review(limit: int = 20) -> None:
    """Walk the low-confidence queue: show subject/sender/body excerpt and the classifier's
    guess with its evidence rule ids; accept, correct, or skip. Corrections become Overrides."""
    ...

@app.command()
def reclassify(all: bool = False) -> None: ...
@app.command("list")
def list_applications(status: str | None = None, company: str | None = None) -> None: ...
@app.command()
def stats() -> None: ...
@app.command()
def export(format: str = "xlsx", output: Path | None = None) -> None: ...
@app.command()
def dashboard(output: Path | None = None, open_browser: bool = False) -> None: ...

auth_app = typer.Typer(help="Manage Gmail credentials.")
@auth_app.command("login")
def auth_login() -> None: ...
@auth_app.command("status")
def auth_status() -> None: ...

def main() -> int:
    """Entry point. The ONLY place that catches JobTrackError and maps it to an exit code."""
    ...
```

## 10. Future — M7 Ollama classifier (Phase 3, not now)

Specified so Phase 1 leaves the right seams. Implementing it must require **zero changes** to
M1, M3, M4, M5, or M6 — only constructing a different `Classifier` in `cli.py`.

```python
class OllamaClassifier:
    """Local-LLM classifier for messages the rules were unsure about.

    Reproducibility contract — all mandatory:
      * model pinned by DIGEST + quantization, not tag — the default is chosen by running
        the eval harness over the shortlist in PLAN.md §8, not assumed here
      * temperature=0, top_p=1, fixed seed, bounded num_predict, thinking/reasoning DISABLED
        (Qwen3-family models emit reasoning traces by default; they break determinism)
      * Ollama structured output: the Classification JSON schema passed as `format`.
        No free-text parsing, no regex over model prose.
      * prompt lives in a versioned template file; its SHA-256 IS `classifier_version`,
        so editing the prompt is a version bump and old rows stay attributable
      * responses cached by (prompt_sha, model_digest, message_id) — reclassify is free
        and byte-identical
      * a malformed or schema-invalid response degrades to the rules result; it never raises
      * ships with an eval harness scoring against accepted/corrected review items before
        the backend may be enabled

    Wiring: CompositeClassifier(RulesClassifier(), OllamaClassifier(), min_confidence=0.60).
    """
    name = "ollama"

    def __init__(self, model: str, *, host: str = "http://localhost:11434",
                 seed: int = 42, think: bool = False, cache: Path | None = None) -> None:
        """Args:
            model: Ollama tag, resolved to a digest at construction and recorded in `version`.
                No default — the eval harness picks it (PLAN.md §8) and config.toml names it.
            think: must stay False; exposed only so the eval harness can measure the cost of
                reasoning traces before rejecting them.
        """
        ...
    @property
    def version(self) -> str:
        """SHA-256 of the prompt template + the resolved model digest + quantization."""
        ...
    def classify(self, message: RawMessage) -> Classification: ...


def evaluate(classifier: Classifier, labeled: Sequence[tuple[RawMessage, Classification]]
             ) -> dict[str, float]:
    """Score a classifier against review-queue labels. Reports schema-compliance rate, overall
    accuracy, accuracy on the confirmation-vs-rejection pair specifically, company/role exact
    match after normalization, and median latency. This is how the model gets chosen."""
    ...
```
