# job-tracker

Track job applications straight from your Gmail inbox. `jobtrack` reads your mail
(read-only), works out which messages are application acknowledgements, rejections,
interview invites, assessments, and offers, groups them into applications, and gives you a
spreadsheet and a dashboard.

Everything except `sync` and `auth login` runs offline.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

# one-time Google Cloud setup — see below — then:
.venv/bin/jobtrack auth login
.venv/bin/jobtrack sync
.venv/bin/jobtrack list
.venv/bin/jobtrack dashboard --open
```

## How it works

```
Gmail ──▶ ingest ──▶ classify ──▶ store ──┬──▶ export   (.csv / .xlsx)
         RawMessage  Classification  SQLite└──▶ viz      (self-contained .html)
```

- **ingest** pulls messages over the Gmail API and normalizes MIME/HTML into plain text.
- **classify** is a pure, deterministic rules engine: no network, no clock, no randomness.
  The same email always produces the same verdict, and every verdict records the rule ids
  that fired so you can see *why*.
- **store** is the only module that speaks SQL. Events are append-only; an application's
  status is derived from its event history on every read, never stored as a column.
- **export** and **viz** both read the same frozen column schema, so they can never drift.

Anything the classifier is unsure about is flagged rather than guessed at — walk the queue
with `jobtrack review`.

## Commands

| Command | What it does |
|---|---|
| `jobtrack auth login` | Run the OAuth consent flow once; writes `token.json`. |
| `jobtrack auth status` | Token presence, expiry, and granted scopes. |
| `jobtrack sync [--since D] [--full] [--dry-run] [--limit N]` | The daily job. Idempotent. |
| `jobtrack review [--limit N]` | Walk the low-confidence queue; accept, correct, or skip. |
| `jobtrack reclassify [--all]` | Re-run the classifier after a rules change. |
| `jobtrack list [--status S] [--company C]` | Table of applications. |
| `jobtrack stats` | Counts, response rate, median time-to-response. |
| `jobtrack export [--format csv\|xlsx] [-o PATH]` | Spreadsheet snapshot. |
| `jobtrack dashboard [-o PATH] [--open]` | Plotly HTML: bar, Sankey, timeline, response times. |
| `jobtrack db migrate` | Apply pending schema migrations. |

`--since` accepts either an ISO-8601 date (`2026-06-01`) or a bare day count (`30`, meaning
30 days ago).

### `sync` is safe to re-run

`message_id` is the dedupe key, so re-running `sync` — or re-running it after a crash —
creates no duplicate events and no duplicate applications. The resume cursor is only
advanced after the batch has committed, so the worst case is re-fetching mail you already
have.

### Classifier backends

Two backends, selected by `JOBTRACK_CLASSIFY_BACKEND`:

| Backend | What it does |
|---|---|
| `rules` (default) | Pure pattern engine. Deterministic, instant, fully offline. |
| `ollama` | A local LLM leads, with `rules` as the fallback. |

The split is not arbitrary. Pattern tables are good at sorting mail into event types and
bad at pulling a company or a job title out of prose — on a real 102-application mailbox
the rules engine extracted a usable role for **21%** of them, and produced companies like
`this time` and `our Software Engineer` by grabbing noun phrases out of rejection text. The
LLM is the reverse, so with `backend = ollama` it leads and the rules catch it when it is
unsure.

**Losing Ollama degrades quality, not uptime.** A stopped daemon, a timeout, or a malformed
response all score 0.0, and the composite falls through to the rules engine. `sync` keeps
working.

#### Setting it up

```bash
brew install ollama          # or however you like
ollama serve
ollama pull qwen2.5:7b

cp .env.example .env         # then edit
.venv/bin/jobtrack reclassify   # re-run the classifier over stored mail
```

Reckon on roughly **8 seconds per message** with a 7B model, so a first pass over a large
mailbox takes a while — `sync` and `reclassify` both draw a progress bar so you can see how
far along it is. The bar appears only on a terminal, so a cron redirect or a pipe stays
clean. Responses are cached on `(prompt_sha, model_digest, message_id)`, so
every later `reclassify` is instant and byte-identical. A 3–4B model is several times faster
at this task and worth benchmarking — see PLAN.md §8 for the shortlist and
`jobtrack.classify.evaluate` for the harness that scores them against your review labels.

#### Reproducibility

The classifier is contractually pure: the same email must always produce the same output.
That is held together by `temperature=0`, a fixed seed, grammar-constrained decoding via
Ollama's `format` parameter, a model pinned by **digest and quantization** rather than tag,
and the response cache. `classifier_version` is the prompt's SHA-256 plus the model digest
plus the quantization level, so editing `classify/prompts/classify_v1.txt`, upgrading the
model, or changing quantization all invalidate old attributions instead of silently changing
what stored rows mean.

A fixed seed only gives determinism on one machine with one build. Treat an Ollama upgrade
as a version bump.

### Corrections stick

`jobtrack review` writes an *override*, which is applied at read time and always wins.
`jobtrack reclassify` refreshes the classifier's own output but never touches an override,
so improving the rules can't silently undo work you did by hand. `reclassify --all`
discards your corrections too — that is the only way to lose them.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Unexpected failure |
| `2` | Usage or configuration error |
| `3` | Authentication failure — run `jobtrack auth login` |
| `4` | Transient network or quota failure — retry later |

Exit `4` is the one worth retrying automatically; a scheduled run that hits a rate limit
returns it rather than failing loudly.

## Google Cloud setup (one-time, manual)

1. Create a new GCP project and enable the **Gmail API**.
2. OAuth consent screen: **External**, testing mode. Add your own address as a test user.
3. Credentials → **Create OAuth client ID** → **Desktop app** → download the JSON.
4. Save it to `$JOBTRACK_HOME/credentials.json`.
5. Run `jobtrack auth login`. A browser opens once; the grant is written to `token.json`.

### Scope

The requested scope is `https://www.googleapis.com/auth/gmail.readonly` **and nothing
else**. `jobtrack` never modifies, labels, archives, or deletes mail. Broadening that scope
is a spec violation, not a feature.

## Files and configuration

Runtime state lives in `$JOBTRACK_HOME` (default `~/.local/share/jobtrack`) — never in the
repo. `credentials.json`, `token.json`, and `*.db` are gitignored.

```
$JOBTRACK_HOME/
├── config.toml       # optional; every field has a working default
├── credentials.json  # your OAuth client, from Google Cloud
├── token.json        # the grant, mode 0600
└── jobtrack.db       # SQLite
```

`config.toml` is optional. A missing file is not an error.

Classifier settings can also come from a `.env` file in the working directory — see
`.env.example`. Precedence, lowest first: **defaults → config.toml → .env → real environment
variables**. `.env` is gitignored.

```bash
JOBTRACK_CLASSIFY_BACKEND=ollama
JOBTRACK_OLLAMA_MODEL=qwen2.5:7b
JOBTRACK_OLLAMA_HOST=http://localhost:11434
JOBTRACK_MIN_CONFIDENCE=0.60
```

```toml
[gmail]
query          = "<recall-oriented OR-query; the classifier is the real filter>"
lookback_days  = 400
max_per_sync   = 500

[classify]
min_confidence = 0.60      # below this ⇒ flagged for review
backend        = "rules"

[store]
ghost_after_days = 30      # silent for N days and not terminal ⇒ GHOSTED

[export]
default_format = "xlsx"
```

## Scheduling a daily sync

### launchd (macOS)

Save as `~/Library/LaunchAgents/com.jobtrack.sync.plist`, then
`launchctl load ~/Library/LaunchAgents/com.jobtrack.sync.plist`.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.jobtrack.sync</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/job-tracker/.venv/bin/jobtrack</string>
    <string>sync</string>
  </array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>15</integer></dict>
  <key>StandardOutPath</key><string>/tmp/jobtrack.log</string>
  <key>StandardErrorPath</key><string>/tmp/jobtrack.err</string>
</dict></plist>
```

### cron

```
15 8 * * * /path/to/job-tracker/.venv/bin/jobtrack sync >> /tmp/jobtrack.log 2>&1
```

## Development

Always use `.venv/bin/python`. Anaconda base ships pydantic 1.10 and an incompatible
typer/click pair — code written against it imports fine and fails at runtime.

```bash
.venv/bin/python -m pytest        # no network: the suite runs with sockets disabled
.venv/bin/ruff check src tests
.venv/bin/black --check src tests
.venv/bin/mypy src/jobtrack
```

Tests never touch the network, a real Gmail account, or a real Ollama. `ingest/` is driven
by recorded JSON fixtures with the transport injected; `classify/` is pinned by a golden
corpus in `tests/fixtures/`; `store/` runs against a real SQLite file in `tmp_path`.

See `CLAUDE.md` for conventions, `CONTRACTS.md` for the frozen cross-module interfaces, and
`PLAN.md` for the module breakdown.
