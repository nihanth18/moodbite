"""SQLite logging of feedback events.

Only numeric mood vectors and restaurant features are stored. Images never reach this module.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

ACTIONS = {"going": 1, "not_feeling_it": 0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,              -- ISO-8601 local timestamp
    hour INTEGER NOT NULL,         -- 0-23, used for time_of_day context
    mood TEXT NOT NULL,            -- JSON: emotion probs + valence + arousal
    restaurant TEXT NOT NULL,      -- JSON: restaurant feature dict (no images, no user identity)
    action TEXT NOT NULL,          -- 'going' | 'not_feeling_it'
    reward INTEGER NOT NULL,       -- 1 | 0
    policy TEXT NOT NULL           -- 'rules' | 'linucb' (which ranker produced the list)
);
"""


class Storage:
    def __init__(self, path: str | Path = "moodbite.db"):
        self.path = str(path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_feedback(
        self,
        mood: dict[str, float],
        restaurant: dict[str, Any],
        action: str,
        policy: str = "rules",
        ts: datetime | None = None,
    ) -> int:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; expected one of {sorted(ACTIONS)}")
        bad = [k for k, v in mood.items() if not isinstance(v, (int, float))]
        if bad:
            raise ValueError(f"mood must be numeric only; non-numeric keys: {bad}")
        ts = ts or datetime.now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO feedback (ts, hour, mood, restaurant, action, reward, policy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts.isoformat(timespec="seconds"),
                    ts.hour,
                    json.dumps({k: float(v) for k, v in mood.items()}),
                    json.dumps(restaurant),
                    action,
                    ACTIONS[action],
                    policy,
                ),
            )
            return int(cur.lastrowid)

    def load_events(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "hour": r["hour"],
                "mood": json.loads(r["mood"]),
                "restaurant": json.loads(r["restaurant"]),
                "action": r["action"],
                "reward": r["reward"],
                "policy": r["policy"],
            }
            for r in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])

    def clear(self) -> None:
        """Delete all logged feedback (user right-to-erasure)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM feedback")
