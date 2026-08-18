# job-tracker — Architecture

A terminal tool that reads Gmail once a day, extracts job-application events, stores them in
SQLite, and renders a spreadsheet + an interactive chart dashboard.

## 1. Problem shape

Application-confirmation and rejection emails are lexically near-identical — both routinely
open with "Thank you for applying to ___". The rejection is distinguished only by a later
clause ("unfortunately", "not moving forward", "decided to proceed with other candidates").

Two consequences drive the whole design:

1. **Event-type precedence is ordered, not first-match.** A message is scored against *every*
   event type and the highest-precedence match wins, with `REJECTION` ranked above
   `APPLICATION_RECEIVED`. First-match-wins would label every rejection as a confirmation.
2. **Status is derived, never stored.** An application's status is a pure function of its event
   history. This makes re-classification safe: wipe classifications, re-run, statuses re-derive.

## 2. High-level data flow

```
                    ┌──────────────────────────────────────────────┐
                    │  Gmail API (read-only)                       │
                    └───────────────────┬──────────────────────────┘
                                        │ users.messages.list / history.list
                                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ M1 ingest        OAuth · query build · incremental cursor          │
   │                  MIME walk · HTML→text · header normalize          │
   └───────────────────┬────────────────────────────────────────────────┘
                       │  FetchResult{ messages: list[RawMessage], next_cursor }
                       ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ M2 classify      PURE. no network, no db, no clock.                │
   │                  ATS detect → ordered event-type rules →           │
   │                  company/role extract → deterministic confidence   │
   └───────────────────┬────────────────────────────────────────────────┘
                       │  Classification{ event_type, company, role, confidence, evidence[] }
                       ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ M3 store         link message→application (thread, then            │
   │                  company_key+role+window) · append-only events ·   │
   │                  overrides · sync cursor · derive status           │
   │                  SQLite @ $JOBTRACK_HOME/jobtrack.db               │
   └──────┬────────────────────────┬───────────────────────┬────────────┘
          │ ApplicationRow[]       │ ReviewItem[]          │ EventRow[]
          ▼                        ▼                       ▼
   ┌─────────────┐        ┌─────────────────┐      ┌──────────────────┐
   │ M4 export   │        │ M6 review (CLI) │      │ M5 viz           │
   │ CSV / XLSX  │        │ accept/correct  │      │ Plotly → 1 HTML  │
   └─────────────┘        └────────┬────────┘      └──────────────────┘
                                   │ Override → M3 (wins at read time, survives reclassify)
                                   └──────────────────────────────────►
```

### Daily run (`jobtrack sync`)

1. Load config, open store, run migrations.
2. Read sync cursor. Fetch new messages (history-delta if the cursor is fresh, else a dated
   query). Fetch is capped and paginated.
3. For each message not already in `messages`: persist raw metadata, classify, link to an
   application (creating one if needed), append an event.
4. `event_type == UNKNOWN` → recorded but linked to no application. Low confidence or missing
   company → flagged `needs_review`.
5. Advance the cursor **only after** the batch commits. Crash mid-run ⇒ re-fetch, and
   `message_id` dedupe makes the replay a no-op.

## 3. Module boundaries

Each module is one agent's work. A module owns its directory exclusively; no agent edits
another module's files. All cross-module traffic goes through the types and signatures frozen
in `CONTRACTS.md`.

### M0 — foundation `src/jobtrack/{models,errors,config,constants}.py`
Package skeleton, `pyproject.toml`, `.venv`, `.gitignore`, all shared Pydantic models, enums,
the error hierarchy, config loading, the frozen `EXPORT_COLUMNS` schema, and `tests/conftest.py`
with the email-fixture loader. **Every other module imports this.**

### M1 — ingest `src/jobtrack/ingest/`
Owns: OAuth desktop flow + token refresh (`auth.py`), the `EmailSource` Protocol (`source.py`),
the Gmail implementation (`gmail.py`), MIME traversal and HTML→text (`html.py`).
Produces `RawMessage`. Knows nothing about jobs, applications, or classification.

### M2 — classify `src/jobtrack/classify/`
Owns: `Classifier` Protocol (`base.py`), pattern tables (`patterns.py`), the rules engine
(`rules.py`), company/role normalization (`normalize.py`), the confidence rubric
(`confidence.py`).
`RawMessage → Classification`. **Pure**: no I/O, no wall clock, no randomness. This is the
highest-value module to test and the one the future Ollama backend slots into.

### M3 — store `src/jobtrack/store/`
Owns: schema + migrations (`schema.sql`, `migrations/`), connection handling (`db.py`), the
repository API (`repo.py`), and message→application matching (`linker.py`).
Sole writer of SQLite. Only module that knows SQL. Derives `ApplicationStatus` from events.

### M4 — export `src/jobtrack/export/tabular.py`
`ApplicationRow[] → pandas.DataFrame → .csv / .xlsx`. Column order and dtypes are frozen by
`EXPORT_COLUMNS` in M0 so M5 can be built in parallel against the same shape.

### M5 — viz `src/jobtrack/viz/`
`charts.py` (one function per figure, each returns a `plotly.graph_objects.Figure`) and
`dashboard.py` (compose figures into one self-contained HTML file). Includes
`compute_stage_flows()`, the pure transform from event history to Sankey links.
Consumes the same DataFrame shape as M4; never touches SQLite.

### M6 — cli `src/jobtrack/cli.py`, `__main__.py`
Typer app. Wires the modules, owns the interactive `review` command, is the **only** layer that
catches exceptions and maps them to exit codes, and the only layer that prints.

## 4. External dependencies and constraints

### Runtime
| Package | Why |
|---|---|
| `google-api-python-client`, `google-auth`, `google-auth-oauthlib` | Gmail API + OAuth |
| `pydantic>=2.7` | All boundary models |
| `typer>=0.12`, `rich` | CLI + review UI |
| `pandas>=2.2`, `openpyxl>=3.1` | DataFrame + XLSX |
| `plotly>=6.0` | Bar, funnel, timeline, **Sankey** |
| `beautifulsoup4`, `lxml` | HTML → text |
| `python-dateutil`, `platformdirs` | Date parsing, XDG paths |

Dev: `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`.

**Do not use Anaconda base.** It has `pydantic 1.10` and a broken `typer 0.9` / `click 8.2`
pairing. Phase 0 creates `.venv` via `python3 -m venv` (`uv` is not installed on this machine).

### Google Cloud setup (one-time, manual — the user does this)
1. New GCP project → enable the Gmail API.
2. OAuth consent screen: External, testing mode, add the user's own address as a test user.
3. Credentials → OAuth client ID → **Desktop app** → download JSON.
4. Save to `$JOBTRACK_HOME/credentials.json`.
5. `jobtrack auth login` opens a browser once and writes `token.json`.

### Hard constraints
- **Scope is `https://www.googleapis.com/auth/gmail.readonly` and nothing else.** The tool must
  never modify, label, archive, or delete mail. Requesting a broader scope is a spec violation.
- Secrets and data live in `$JOBTRACK_HOME` (default `~/.local/share/jobtrack`, override by env),
  **never** in the repo. `credentials.json`, `token.json`, `*.db` are gitignored.
- Gmail API quota: 250 quota-units/user/second. `messages.get` is 5 units. Batch in pages of 100
  with exponential backoff on 429/5xx. A daily run touches well under any daily cap.
- `historyId` deltas expire after roughly a week. If `history.list` returns 404, fall back to a
  dated query and log the downgrade.
- Offline-first: everything except `sync` and `auth` runs with no network.

### Config `$JOBTRACK_HOME/config.toml`
```toml
[gmail]
query          = "<default recall-oriented OR-query; see CONTRACTS §DEFAULT_GMAIL_QUERY>"
lookback_days  = 400
max_per_sync   = 500

[classify]
min_confidence   = 0.60      # below this ⇒ needs_review
backend          = "rules"   # future: "rules+ollama"
# ollama_model   = ""        # Phase 3; set from the eval harness result, pinned by digest
# ollama_host    = "http://localhost:11434"

[store]
ghost_after_days = 30     # no event for N days & not terminal ⇒ GHOSTED

[export]
default_format = "xlsx"
```

## 5. Sequential vs. parallelizable phases

```
Phase 0  ── M0 foundation ──┐   SEQUENTIAL. 1 agent. Blocks everything.
                            │
                            ├── M1 ingest   ─┐
Phase 1                     ├── M2 classify ─┤  PARALLEL. 5 agents, zero file overlap.
                            ├── M3 store    ─┤
                            ├── M4 export   ─┤
                            └── M5 viz      ─┘
                                             │
Phase 2  ────────────────────── M6 cli ──────┘   SEQUENTIAL. 1 agent. Integration + e2e.

Phase 3  (optional, later) ── M7 Ollama classifier
```

**Why Phase 1 truly parallelizes.** M4 and M5 would normally be coupled through the DataFrame,
but `EXPORT_COLUMNS` is frozen in M0, so both build against a fixed shape. M3 and M2 would
normally be coupled through classification storage, but `Classification` is frozen in M0 and the
linker takes pre-fetched candidates rather than querying inside the matching logic. M1 and M2 are
coupled only through `RawMessage`, also frozen in M0.

**Phase 1 agent brief (each agent).** Read `CLAUDE.md` + `CONTRACTS.md` in full, then implement
only the files listed under your module in PLAN.md §3. Fill in the stubs in your CONTRACTS
section, keeping every public signature byte-identical. Write tests under `tests/unit/<module>/`.
Do not create, edit, or delete files outside your directory — if you need a change to a shared
type, stop and report it rather than editing `models.py`.

**Phase 2 brief.** Wire the CLI, build the `review` command, add `tests/e2e/test_sync_flow.py`
running a fixture mailbox through ingest→classify→store→export→dashboard against a `tmp_path`
DB, write the README (including GCP setup and the launchd/cron snippets below). Phase 2 may fix
integration bugs in Phase 1 modules.

### Scheduling (documented, not coded)

launchd — `~/Library/LaunchAgents/com.jobtrack.sync.plist`, runs 08:15 daily:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.jobtrack.sync</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/ashwintyagi/Documents/Projects/job-tracker/.venv/bin/jobtrack</string>
    <string>sync</string>
  </array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>15</integer></dict>
  <key>StandardOutPath</key><string>/tmp/jobtrack.log</string>
  <key>StandardErrorPath</key><string>/tmp/jobtrack.err</string>
</dict></plist>
```
`launchctl load ~/Library/LaunchAgents/com.jobtrack.sync.plist`

cron equivalent: `15 8 * * * /path/to/.venv/bin/jobtrack sync >> /tmp/jobtrack.log 2>&1`

## 6. CLI surface

| Command | Purpose |
|---|---|
| `jobtrack auth login` / `auth status` | OAuth consent; show token validity |
| `jobtrack sync [--since D] [--full] [--dry-run] [--limit N]` | The daily job |
| `jobtrack reclassify [--all\|--unreviewed]` | Re-run the classifier over stored messages after a rules change; overrides are preserved |
| `jobtrack review [--limit N]` | Walk the low-confidence queue, accept or correct |
| `jobtrack list [--status S] [--company C]` | Table of applications |
| `jobtrack stats` | Counts, response rate, median time-to-response |
| `jobtrack export [--format csv\|xlsx] [-o PATH]` | Spreadsheet snapshot |
| `jobtrack dashboard [-o PATH] [--open]` | Plotly HTML (bar, Sankey, timeline, response-time) |
| `jobtrack db migrate` | Apply pending migrations |

Exit codes: `0` ok · `1` unexpected · `2` usage · `3` auth · `4` transient network/quota.

## 7. Classification design (M2 detail)

### Ordered event-type precedence — an invariant, not a heuristic
```
WITHDRAWN > REJECTION > OFFER > INTERVIEW > ASSESSMENT > APPLICATION_RECEIVED
          > RECRUITER_OUTREACH > UNKNOWN
```
Score against all types; highest-precedence non-empty match wins. `REJECTION` outranks
`APPLICATION_RECEIVED` because rejections restate the application language, and outranks
`INTERVIEW` because post-interview rejections restate the interview.

### Three-stage pipeline
1. **ATS / sender detection** — match sender domain, `Reply-To`, and `List-Unsubscribe` against
   a table of known ATS hosts (greenhouse, lever, workday, ashby, smartrecruiters, icims, taleo,
   jobvite, workable, breezy, bamboohr, recruitee, teamtailor, jazzhr, dover, rippling, wellfound,
   linkedin, indeed). Sets `ats` and contributes confidence.
2. **Event typing** — ordered pattern tables over normalized subject and body. Every pattern has
   a stable rule id (`rej.subject.not_moving_forward`, `ack.body.application_received`,
   `ats.sender.greenhouse`) recorded in `Classification.evidence`.
3. **Field extraction** — company via an ordered chain (ATS-specific sender/header → subject
   capture group → body signature → sender display name), role mostly via subject capture groups.
   `normalize_company()` produces the `company_key` used for matching; the display string is kept
   verbatim.

### Confidence rubric — fixed, additive, documented
| Signal | Δ |
|---|---|
| Sender/header identifies a known ATS | +0.35 |
| High-precision subject pattern for the winning type | +0.40 |
| Corroborating body pattern | +0.20 |
| Company extracted by a high-precision extractor | +0.05 |
| Two types matched with adjacent precedence (genuine ambiguity) | −0.20 |

Clamp to `[0.0, 1.0]`. `needs_review = confidence < min_confidence or company is None`.
The rubric is a table in `confidence.py`, not scattered magic numbers.

### Future LLM backend (M7) — reproducibility contract, specified now
Not implemented in Phase 1. When it lands it must satisfy all of:
- Model **pinned by digest and quantization**, not tag — see §8 for selection.
- `temperature=0`, `top_p=1`, fixed `seed`, bounded `num_predict`.
- Ollama structured outputs: pass the `Classification` JSON schema as `format`. No free-text
  parsing, no regex over model prose.
- Prompt lives in a versioned template file; its SHA-256 is `classifier_version`. **Editing the
  prompt is a version bump** — old rows stay attributable to the prompt that produced them.
- Responses cached by `(prompt_sha, model_digest, message_id)`, so a reclassify is free and
  byte-identical.
- Slots in as `CompositeClassifier(primary=RulesClassifier(), fallback=OllamaClassifier(), min_confidence=...)`
  — the fallback only sees messages the rules were unsure about. No caller changes.
- Every accepted/corrected review item is labeled eval data; M7 ships with an eval harness that
  reports accuracy against it before the backend may be enabled.

## 8. Local model selection (M7) — decide by eval, not by vibes

The job here is narrow: a short email in, a small fixed-schema JSON out (one enum + two or three
short strings). That is a **constrained extraction** task, not a reasoning task. It does not need
7B. A well-chosen 3–4B model is typically as accurate here, several times faster, and leaves the
laptop usable during a daily batch.

Because M7 ships an eval harness scoring against the review-queue labels anyway, model choice is a
measurement, not a guess. Pull the shortlist, run the harness, keep the smallest model that clears
the accuracy bar.

### Shortlist to benchmark

| Model | ~Size | Why it's on the list |
|---|---|---|
| **`qwen3:4b`** | 4B | The one to beat. Qwen3 is the strongest small instruction-follower for schema-constrained output, and 4B is right-sized for this task. **Must run with thinking disabled** (`/no_think` or `think: false`) — reasoning traces destroy both latency and reproducibility. |
| **`granite4:small-h`** / `granite3.3:2b` | 2–3B | IBM tunes Granite explicitly for enterprise extraction and tool/JSON output. Punches above its weight on exactly this shape of task. |
| **`phi4-mini`** | 3.8B | Microsoft; strong instruction-following per parameter, reliable JSON. |
| **`gemma3:4b`** | 4B | Google; excellent general quality at 4B, long context if a verbose ATS email needs it. |
| **NuExtract 2.0 (4B)** | 4B | The wildcard worth a look: purpose-built for **schema-driven information extraction** rather than chat. If it wins, it likely wins clearly. Availability in the Ollama library is patchy — may need a GGUF from Hugging Face via a `Modelfile`. |
| `qwen2.5:7b` | 7B | Already installed. Use as the baseline to beat, not the default. |
| `phi3:instruct` | 3.8B | Already installed, but superseded — `phi4-mini` should replace it. |

Verify tags with `ollama pull <tag>` at implementation time; the Ollama library moves and some of
the above may have newer point releases.

### Selection criteria, in order
1. **Schema compliance** — % of responses that are valid against the `Classification` JSON schema
   on the first try. A model that needs retries is disqualified; retries break reproducibility.
2. **Accuracy on the confusable pair** — the "Thanks for applying" confirmation vs. the "Thanks for
   applying … unfortunately" rejection. This is the whole reason the LLM exists. Weight it heavily.
3. **Company/role extraction exactness** after `normalize_company()`.
4. **Latency** — the fallback only sees low-confidence messages, so a daily run is likely tens of
   calls; anything under a few seconds each is fine. Don't over-optimize this.
5. **Size**, as the tiebreaker.

### Reproducibility caveats specific to local models
- A fixed `seed` gives determinism **on one machine with one build**. Output can shift across
  Ollama versions, quantizations, and GPU/CPU backends — so pin the **digest and the quant level**
  (prefer `q4_K_M` or better; avoid `q3` and below, which degrade JSON reliability), and treat an
  Ollama upgrade as a `classifier_version` bump.
- Prefer models with **native structured-output support** through Ollama's `format` parameter.
  Grammar-constrained decoding is what makes schema compliance a property of the sampler rather
  than a hope about the prompt.
- Never let the model invent an `EventType`. The schema's enum constrains it; the parser rejects
  anything outside it and degrades to the rules result.
