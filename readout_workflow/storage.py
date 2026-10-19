"""One authoritative SQLite checkpoint, plus a small best-effort status file.

Historical event IDs stay unchanged after consolidation. A separate signature
protects the active equations/settings from accidental mixing on later resumes.
Only nominal-map phases are accepted; robustness jobs cannot be recorded here.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

MAP_PHASES = frozenset(
    {
        "feasibility_normal",
        "feasibility_deep",
        "diagnostic",
        "strength_1",
        "strength_2",
        "continuation",
    }
)


def utc():
    return datetime.now(UTC).isoformat()


def key(pair):
    return f"{int(pair[0])}:{int(pair[1])}"


def atomic_json(path, value, *, best_effort=False, attempts=12):
    """Retry Windows sharing violations; status locks must not stop physics."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2)
    last_error = None
    try:
        for index in range(attempts):
            try:
                temporary.write_text(serialized, encoding="utf-8")
                temporary.replace(path)
                return True
            except PermissionError as error:
                last_error = error
                if index + 1 < attempts:
                    time.sleep(min(0.1 * (index + 1), 0.5))
        if not best_effort:
            raise last_error
        print(
            f"Status file locked; committed SQLite events remain safe: {path}",
            file=sys.stderr,
            flush=True,
        )
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


class Ledger:
    """Append-only map events; only the parent process writes to SQLite."""

    def __init__(self, output, policy):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "samples.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events(sample_id TEXT PRIMARY KEY,phase TEXT,pair_key TEXT,payload TEXT)"
        )
        signature = sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
        previous = self.db.execute(
            "SELECT value FROM metadata WHERE key='calculation_signature'"
        ).fetchone()
        if previous and previous[0] != signature:
            self.db.close()
            raise RuntimeError(
                "Equations, seeds or settings changed; use a new output directory"
            )
        original = self.db.execute(
            "SELECT value FROM metadata WHERE key='policy_hash'"
        ).fetchone()
        # Preserve the original namespace so all 1,923 existing events remain
        # recognized. The new signature records the cleaned implementation.
        self.policy_hash = original[0] if original else signature
        self.db.execute(
            "INSERT OR IGNORE INTO metadata VALUES('policy_hash',?)",
            (self.policy_hash,),
        )
        self.db.execute(
            "INSERT OR IGNORE INTO metadata VALUES('calculation_signature',?)",
            (signature,),
        )
        self.db.commit()
        self.ids = {r[0] for r in self.db.execute("SELECT sample_id FROM events")}
        self.phase = "initializing"

    def sample_id(self, phase, pair):
        return f"{self.policy_hash[:16]}:{phase}:{key(pair)}"

    def done(self, phase, pair):
        return self.sample_id(phase, pair) in self.ids

    def rows(self, phase):
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT payload FROM events WHERE phase=? ORDER BY rowid", (phase,)
            )
        ]

    def save(self, phase, pair, result):
        if phase not in MAP_PHASES:
            raise ValueError(f"Phase is outside the authorized map-only scope: {phase}")
        identifier = self.sample_id(phase, pair)
        if identifier in self.ids:
            raise RuntimeError("Refusing to overwrite a committed event")
        record = result | dict(
            phase=phase, sample_id=identifier, pair_key=key(pair), completed_utc=utc()
        )
        self.db.execute(
            "INSERT INTO events VALUES(?,?,?,?)",
            (identifier, phase, key(pair), json.dumps(record)),
        )
        self.db.commit()
        self.ids.add(identifier)
        self.progress()
        return record

    def progress(self, **extra):
        counts = dict(
            self.db.execute("SELECT phase,count(*) FROM events GROUP BY phase")
        )
        atomic_json(
            self.output / "progress.json",
            dict(
                phase=self.phase,
                updated_utc=utc(),
                pid=os.getpid(),
                event_counts=counts,
                total_events=len(self.ids),
                scope="map_only",
                robustness_authorized=False,
                **extra,
            ),
            best_effort=True,
        )

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(FULL)")
        self.db.close()


def status(output):
    """Read committed counts without starting a model or optimization."""
    output = Path(output)
    database = output / "samples.sqlite"
    if not database.exists():
        return dict(phase="not_started", events=0, scope="map_only")
    connection = sqlite3.connect(database)
    try:
        counts = dict(
            connection.execute("SELECT phase,count(*) FROM events GROUP BY phase")
        )
    finally:
        connection.close()
    snapshot = output / "progress.json"
    state = (
        json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.exists() else {}
    )
    return state | dict(
        event_counts=counts,
        total_events=sum(counts.values()),
        scope="map_only",
        robustness_authorized=False,
    )
