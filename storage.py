"""Feedback log: SQLite locally, or Postgres when given a postgres:// URL (persistent hosting).

Only numeric mood vectors and restaurant features are stored. Images never reach this module.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

ACTIONS = {"going": 1, "not_feeling_it": 0}

SQLITE_SCHEMA = """
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
POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    ts TEXT NOT NULL,
    hour INTEGER NOT NULL,
    mood TEXT NOT NULL,
    restaurant TEXT NOT NULL,
    action TEXT NOT NULL,
    reward INTEGER NOT NULL,
    policy TEXT NOT NULL
);
"""


def is_postgres_url(target: str | Path) -> bool:
    return str(target).startswith(("postgres://", "postgresql://"))


class Storage:
    def __init__(self, target: str | Path = "moodbite.db"):
        """`target` is a SQLite file path or a postgres:// connection URL."""
        self.target = str(target)
        self.kind = "postgres" if is_postgres_url(self.target) else "sqlite"
        self.path = self.target if self.kind == "sqlite" else None
        if self.kind == "postgres":
            self._exec(POSTGRES_SCHEMA)
        else:
            conn = sqlite3.connect(self.target)
            try:
                conn.executescript(SQLITE_SCHEMA)
            finally:
                conn.close()

    @property
    def persistent(self) -> bool:
        """True when feedback survives app restarts on hosting (i.e. an external database)."""
        return self.kind == "postgres"

    @property
    def label(self) -> str:
        return "Hosted database" if self.persistent else "Local file (resets on hosted restarts)"

    # --- low-level -----------------------------------------------------------------------------
    def _exec(self, sql: str, params: tuple = (), fetch: str | None = None) -> Any:
        """Run one statement; `?` placeholders work for both backends."""
        if self.kind == "postgres":
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(self.target, row_factory=dict_row, connect_timeout=10) as conn:
                cur = conn.execute(sql.replace("?", "%s"), params)
                return self._fetch(cur, fetch)
        conn = sqlite3.connect(self.target)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                cur = conn.execute(sql, params)
                return self._fetch(cur, fetch)
        finally:
            conn.close()

    @staticmethod
    def _fetch(cur: Any, fetch: str | None) -> Any:
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        return None

    # --- API -----------------------------------------------------------------------------------
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
        row = self._exec(
            "INSERT INTO feedback (ts, hour, mood, restaurant, action, reward, policy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (
                ts.isoformat(timespec="seconds"),
                ts.hour,
                json.dumps({k: float(v) for k, v in mood.items()}),
                json.dumps(restaurant),
                action,
                ACTIONS[action],
                policy,
            ),
            fetch="one",
        )
        return int(row["id"])

    def load_events(self) -> list[dict[str, Any]]:
        rows = self._exec("SELECT * FROM feedback ORDER BY id", fetch="all")
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
        return int(self._exec("SELECT COUNT(*) AS n FROM feedback", fetch="one")["n"])

    def metrics(self) -> dict[str, Any]:
        """'Going' rate overall and per ranking policy: the app's accuracy measure."""
        rows = self._exec("SELECT policy, COUNT(*) AS n, SUM(reward) AS going FROM feedback "
                          "GROUP BY policy", fetch="all")
        per = {r["policy"]: {"n": int(r["n"]), "going": int(r["going"] or 0)} for r in rows}
        for v in per.values():
            v["rate"] = v["going"] / v["n"] if v["n"] else None
        n = sum(v["n"] for v in per.values())
        going = sum(v["going"] for v in per.values())
        return {"n": n, "going": going, "rate": going / n if n else None, "by_policy": per}

    def clear(self) -> None:
        """Delete all logged feedback (user right-to-erasure)."""
        self._exec("DELETE FROM feedback")
