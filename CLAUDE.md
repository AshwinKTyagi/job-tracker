# job-tracker — conventions for implementation agents

Read this and `CONTRACTS.md` in full before writing code. Implement only the module assigned to
you in `PLAN.md` §3.

## Golden rules

1. **Signatures in `CONTRACTS.md` are frozen.** Implement them byte-identically — names, order,
   type hints, defaults. If a contract is wrong or insufficient, **stop and report it**; do not
   unilaterally change it. Other agents are compiling against it right now.
2. **Stay in your directory.** Do not create, edit, or delete files outside your module. In
   particular: never edit `models.py`, `errors.py`, `config.py`, or `constants.py` — those are
   M0's, and they are already final.
3. **No network in tests. Ever.** Not even "just this one integration test."
4. **Gmail access is read-only.** Scope is `gmail.readonly`. Never call a mutating Gmail method.

## Layout

```
job-tracker/
├── pyproject.toml
├── PLAN.md · CLAUDE.md · CONTRACTS.md · README.md
├── src/jobtrack/
│   ├── __init__.py · __main__.py · cli.py          M6
│   ├── models.py · errors.py · config.py · constants.py   M0
│   ├── ingest/   __init__ auth source gmail html    M1
│   ├── classify/ __init__ base rules patterns normalize confidence   M2
│   ├── store/    __init__ db repo linker schema.sql migrations/      M3
│   ├── export/   __init__ tabular                   M4
│   └── viz/      __init__ charts dashboard          M5
└── tests/
    ├── conftest.py
    ├── fixtures/emails/*.json      # RawMessage-shaped, redacted
    ├── fixtures/expected.jsonl     # golden classifications
    ├── unit/{ingest,classify,store,export,viz}/
    └── e2e/
```

Runtime state lives in `$JOBTRACK_HOME` (default `~/.local/share/jobtrack`), never in the repo:
`config.toml`, `credentials.json`, `token.json`, `jobtrack.db`.

## Style

- Python 3.11+. `from __future__ import annotations` at the top of every module.
- Full type hints on every function, including tests. `mypy --strict` on `src/` must pass.
- `ruff` + `black`, line length **100**.
- `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants.
  Private helpers get a leading underscore and are not imported across modules.
- **Pydantic v2** (`BaseModel`) for anything crossing a module boundary. `@dataclass(frozen=True)`
  for module-internal value objects. Do not mix.
- Interfaces are `typing.Protocol`, not ABCs. Depend on the Protocol, never on a concrete class.
- Google-style docstrings on every public function: one-line summary, `Args`, `Returns`, `Raises`.
  Private helpers need only the summary line.
- Constants over literals. Regexes are named, `re.compile`d at module level, `re.VERBOSE` with a
  comment when non-trivial.
- Prefer pure functions. If something needs the clock, take `now: datetime` as a parameter — do
  not reach for `datetime.now()` inside logic.

## Forbidden

| Don't | Instead |
|---|---|
| `print()` outside `cli.py` / `viz/` | `logger = logging.getLogger(__name__)` |
| `sys.exit()` in a library module | Raise a `JobTrackError` subclass; `cli.py` maps it to an exit code |
| `except:` / `except Exception:` swallowing | Catch the narrowest type; re-raise wrapped in a `JobTrackError` |
| f-string or `%` interpolation into SQL | `?` placeholders, always |
| `SELECT *` | Name every column; map by name, never by tuple index |
| Module-level DB connection or other global mutable state | Pass `Store` / `Config` explicitly |
| `datetime.now()` inside classify/store logic | Inject `now: datetime`; UTC-aware everywhere |
| Naive datetimes anywhere | `datetime.now(timezone.utc)`; persist ISO-8601 UTC |
| Adding a runtime dependency not in `PLAN.md` §4 | Report it; don't just `pip install` |
| Writing files outside `$JOBTRACK_HOME` or an explicit `-o` path | — |
| Committing `credentials.json`, `token.json`, `*.db`, `.venv/` | `.gitignore` covers these; keep it that way |
| Broadening the Gmail OAuth scope | `gmail.readonly` is the whole budget |
| Storing `ApplicationStatus` as a column | Derive it from events on read |
| `assert` for runtime validation | `assert` is for tests; raise a real error in `src/` |
| Bare `Any`, `# type: ignore` without a reason comment | Type it properly |

## Errors

Every module raises from the `JobTrackError` hierarchy in `errors.py` (see `CONTRACTS.md` §2).
Wrap third-party exceptions at your module's boundary — a `googleapiclient.errors.HttpError`
must never escape `ingest/`, and a `sqlite3.Error` must never escape `store/`.

Transient vs. permanent matters: `TransientIngestError` (429, 5xx, socket timeout) is retried
with exponential backoff and surfaces as exit code 4; `PermanentIngestError` is not retried.

## Testing

- `pytest`. Tests mirror the source tree: `tests/unit/<module>/test_<file>.py`.
- **No network, no real Gmail, no real Ollama.** `ingest/` is tested against recorded JSON
  fixtures of Gmail API responses; the transport is injected.
- **Classifier golden tests.** `tests/fixtures/emails/*.json` are redacted real emails;
  `expected.jsonl` holds the expected `event_type` / `company` / `role` for each. Every fixture
  must pass. **Adding a pattern to `patterns.py` requires adding a fixture that exercises it**,
  including at least one negative — a rejection that must not be read as a confirmation.
- Store tests use a real SQLite DB in `tmp_path`, never a mock. Migrations are tested by running
  them against an empty file and asserting the resulting schema.
- Determinism: classifying the same fixture twice must produce byte-identical output. Assert it.
- Coverage floor: **90% on `classify/` and `store/`**, 75% elsewhere. `pytest -q` must pass with
  the network unplugged.

## Definition of done, per module

Signatures match `CONTRACTS.md` · `mypy --strict src/jobtrack/<module>` clean · `ruff check` and
`black --check` clean · your unit tests pass · coverage floor met · no file outside your
directory modified · every public function has a docstring.

## Environment — Phase 0 already landed this

**Always use `.venv/bin/python`.** Never `/opt/anaconda3/bin/python3`: Anaconda base ships
pydantic 1.10 and an incompatible typer/click pair, so code written against it imports fine
and fails at runtime.

```
$ .venv/bin/python -m pytest      # 33 passing
$ .venv/bin/ruff check src tests
$ .venv/bin/black --check src tests
$ .venv/bin/mypy src/jobtrack
```

Resolved versions in the venv:

```
pydantic                   2.13.4
typer                      0.27.1
click                      8.4.2
pandas                     3.0.5
plotly                     6.9.0
openpyxl                   3.1.5
beautifulsoup4             4.15.0
google-api-python-client   2.198.0
mypy                       2.3.1
ruff                       0.16.3
black                      26.5.1
pytest                     9.1.1
```

**pandas is 3.x, not 2.x.** M4 and M5: default string dtype is `str` rather than `object`, and
copy-on-write is always on. Write pandas-3 idioms — no `inplace=`, no chained assignment, and
do not assume `object` dtype when asserting on columns.

### Files M0 owns — already written, do not edit

`src/jobtrack/{models,errors,config,constants}.py`, `pyproject.toml`, `.gitignore`,
`tests/conftest.py`. They implement CONTRACTS.md §1–§3 exactly. If one blocks you, report it.

`tests/conftest.py` gives you `make_message` (RawMessage factory), `tmp_config`, `frozen_now`,
`email_fixtures`, and `expected` — use them rather than rolling your own.

### Two contract corrections Phase 0 made

Both were found by running the tests, and both are already reflected in CONTRACTS.md:

1. **`EVENT_PRECEDENCE` was missing `WITHDRAWN`** — it listed 7 of 8 `EventType` members, so
   `resolve_event_type` would have dropped or crashed on it. `WITHDRAWN` now leads the tuple.
   A test asserts the tuple covers every enum member; keep it passing.
2. **Enums are `StrEnum`, not `(str, Enum)`** — under `(str, Enum)`,
   `str(EventType.REJECTION)` yields `"EventType.REJECTION"`, which would silently write enum
   *names* into SQLite and DataFrames. `StrEnum` yields `"rejection"`.
