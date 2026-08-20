-- Migration 0001: initial schema (SCHEMA_VERSION = 1).
--
-- This file is a human-readable reference for the schema after all migrations have been
-- applied. `Store.migrate()` does NOT read this file directly -- it applies the numbered
-- files under `migrations/` in order and records each one in `schema_version`. Keep this
-- file in sync with `migrations/0001_initial.sql` (and any migration added later).
--
-- Design notes (see the final report for the full rationale):
--   * No `status` column anywhere (I4): ApplicationStatus is derived from `events` on read.
--   * `events` is append-only (I5): event_id is an AUTOINCREMENT surrogate key, message_id
--     is UNIQUE so a message produces at most one event row, and no UPDATE/DELETE is ever
--     issued against this table by repo.py.
--   * `overrides` is a separate table so a human correction never mutates a `classifications`
--     or `events` row (I5, I6); it is joined in at read time and always wins.
--   * `classifications.reviewed_at` is an addition beyond the CONTRACTS.md sketch: it
--     distinguishes classifications a human has confirmed (via `accept_classification`)
--     from ones nobody has looked at yet, which is what `clear_classifications` uses to
--     decide what a reclassify is allowed to discard.
--   * All timestamps are stored as ISO-8601 UTC TEXT (I7).

CREATE TABLE IF NOT EXISTS messages (
    message_id   TEXT PRIMARY KEY,
    thread_id    TEXT NOT NULL,
    received_at  TEXT NOT NULL,
    from_email   TEXT NOT NULL,
    from_name    TEXT,
    to_email     TEXT,
    subject      TEXT NOT NULL DEFAULT '',
    body_text    TEXT NOT NULL DEFAULT '',
    snippet      TEXT NOT NULL DEFAULT '',
    labels_json  TEXT NOT NULL DEFAULT '[]',
    headers_json TEXT NOT NULL DEFAULT '{}',
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classifications (
    message_id         TEXT PRIMARY KEY REFERENCES messages(message_id),
    event_type         TEXT NOT NULL,
    company             TEXT,
    company_key         TEXT,
    role                 TEXT,
    location             TEXT,
    ats                   TEXT,
    confidence           REAL NOT NULL,
    needs_review         INTEGER NOT NULL,
    evidence_json        TEXT NOT NULL DEFAULT '[]',
    classifier_name      TEXT NOT NULL,
    classifier_version   TEXT NOT NULL,
    classified_at        TEXT NOT NULL,
    reviewed_at           TEXT
);

CREATE TABLE IF NOT EXISTS applications (
    application_id TEXT PRIMARY KEY,
    company         TEXT NOT NULL,
    company_key     TEXT NOT NULL,
    role             TEXT,
    location         TEXT,
    ats               TEXT,
    applied_at       TEXT NOT NULL,
    created_at       TEXT NOT NULL
    -- NO status column (I4): derived from events on every read.
);

CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT REFERENCES applications(application_id),
    message_id     TEXT NOT NULL UNIQUE REFERENCES messages(message_id),
    event_type     TEXT NOT NULL,
    occurred_at    TEXT NOT NULL,
    created_at     TEXT NOT NULL
    -- Append-only (I5): repo.py never UPDATEs or DELETEs a row here.
);

CREATE TABLE IF NOT EXISTS overrides (
    message_id    TEXT PRIMARY KEY REFERENCES messages(message_id),
    event_type    TEXT,
    company        TEXT,
    role            TEXT,
    corrected_at   TEXT NOT NULL,
    note            TEXT
);

CREATE TABLE IF NOT EXISTS sync_state (
    source          TEXT PRIMARY KEY,
    cursor           TEXT,
    last_synced_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_apps_company_key ON applications(company_key);
CREATE INDEX IF NOT EXISTS idx_events_app       ON events(application_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_cls_review       ON classifications(needs_review);
