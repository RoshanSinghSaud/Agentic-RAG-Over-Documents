"""Per-session token ceiling (Layer 8 Day 21).

A "session" is a thread_id: the same key the checkpointer uses for a
conversation. Totals live in their own table in the checkpoint SQLite file, so
they last exactly as long as the conversation they belong to (including
across restarts), and need no new infrastructure.

thread_id is chosen by the client, so this bounds what one conversation can
spend, not what one person can: a new thread_id starts at zero. The per-IP
rate limit in app.py is what bounds a person.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class SessionLedger:
    def __init__(self, path: Path | str) -> None:
        # Own connection: SqliteSaver serialises its writes on its connection
        # with its own lock. Two connections to one file is what SQLite's file
        # locking is for; timeout makes a writer wait instead of failing.
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS session_usage (
                       thread_id  TEXT PRIMARY KEY,
                       tokens     INTEGER NOT NULL DEFAULT 0,
                       cost_usd   REAL    NOT NULL DEFAULT 0,
                       requests   INTEGER NOT NULL DEFAULT 0,
                       updated_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
                   )"""
            )

    def tokens_used(self, thread_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT tokens FROM session_usage WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return row[0] if row else 0

    def add(self, thread_id: str, tokens: int, cost_usd: float) -> int:
        """Record one request's usage; return the session's new token total."""
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO session_usage (thread_id, tokens, cost_usd, requests)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(thread_id) DO UPDATE SET
                       tokens     = tokens + excluded.tokens,
                       cost_usd   = cost_usd + excluded.cost_usd,
                       requests   = requests + 1,
                       updated_at = CURRENT_TIMESTAMP""",
                (thread_id, tokens, cost_usd),
            )
            return self._conn.execute(
                "SELECT tokens FROM session_usage WHERE thread_id = ?", (thread_id,)
            ).fetchone()[0]
