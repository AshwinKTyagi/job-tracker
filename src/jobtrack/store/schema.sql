-- jobtrack schema, version 1.
--
-- This file is the canonical description of the CURRENT schema. It is kept
-- byte-identical to the union of everything under store/migrations/ and a unit test
-- (tests/unit/store/test_db.py::test_migrations_match_schema_sql) fails if the two
-- ever drift apart.
--
-- Conventions:
--   * every timestamp column is ISO-8601 UTC text (invariant I7);
--   * `applications` has NO status column — ApplicationStatus is derived from the event
--     history on every read (invariant I4);
--   * `events` is append-only (invariant I5): a correction writes an `overrides` row,
--     it never mutates or deletes an event;
--   * `messages.message_id` is the universal dedupe key (invariant I1);
--   * columns whose value has no injected clock (`ingested_at`, `classified_at`) default
--     to the database clock. Nothing derives from them — they are provenance only.

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT    NOT NULL
);

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
    ingested_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS classifications (
    message_id         TEXT PRIMARY KEY REFERENCES messages (message_id) ON DELETE CASCADE,
    event_type         TEXT    NOT NULL,
    company            TEXT,
    company_key        TEXT,
    role               TEXT,
    location           TEXT,
    ats                TEXT,
    confidence         REAL    NOT NULL DEFAULT 0.0,
    needs_review       INTEGER NOT NULL DEFAULT 0,
    evidence_json      TEXT    NOT NULL DEFAULT '[]',
    classifier_name    TEXT    NOT NULL,
    classifier_version TEXT    NOT NULL,
    classified_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- NO status column here: status is derived from `events` on read (I4).
CREATE TABLE IF NOT EXISTS applications (
    application_id TEXT PRIMARY KEY,
    company        TEXT NOT NULL,
    company_key    TEXT NOT NULL,
    role           TEXT,
    location       TEXT,
    ats            TEXT,
    applied_at     TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

-- Append-only (I5). application_id is NULL for UNKNOWN or otherwise unlinked messages.
CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT REFERENCES applications (application_id) ON DELETE SET NULL,
    message_id     TEXT NOT NULL UNIQUE REFERENCES messages (message_id) ON DELETE CASCADE,
    event_type     TEXT NOT NULL,
    occurred_at    TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

-- A human correction. Wins over the classifier at read time and survives reclassify (I6).
-- All three correction columns NULL means "the classifier was accepted as-is": the row is
-- retained as labeled data but changes nothing on read.
CREATE TABLE IF NOT EXISTS overrides (
    message_id   TEXT PRIMARY KEY REFERENCES messages (message_id) ON DELETE CASCADE,
    event_type   TEXT,
    company      TEXT,
    role         TEXT,
    corrected_at TEXT NOT NULL,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS sync_state (
    source         TEXT PRIMARY KEY,
    cursor         TEXT,
    last_synced_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_apps_company_key ON applications (company_key);
CREATE INDEX IF NOT EXISTS idx_events_app       ON events (application_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_message   ON events (message_id);
CREATE INDEX IF NOT EXISTS idx_cls_review       ON classifications (needs_review);
CREATE INDEX IF NOT EXISTS idx_messages_thread  ON messages (thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_recv    ON messages (received_at);
