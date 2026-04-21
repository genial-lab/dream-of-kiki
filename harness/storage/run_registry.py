"""Run registry — SQLite-backed, reproducibility contract R1."""
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class RunRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    c_version TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    commit_sha TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_output_hashes (
                    run_id TEXT PRIMARY KEY,
                    output_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

    def _compute_run_id(
        self, c_version: str, profile: str, seed: int, commit_sha: str
    ) -> str:
        """Compute deterministic run_id (128-bit SHA-256 prefix).

        Contract R1: same (c_version, profile, seed, commit_sha)
        tuple always produces the same run_id bit-for-bit.

        History: initially truncated to 16 hex chars (64 bits),
        bumped to 32 hex chars (128 bits) in commit df731b0 after
        code-review finding MED2 identified 50%-collision risk at
        ~2^32 runs. No migration was required because the DB was
        empty at bump time. Any future change to this slice width
        requires a migration script to recompute existing row ids.
        """
        key = f"{c_version}|{profile}|{seed}|{commit_sha}".encode()
        return hashlib.sha256(key).hexdigest()[:32]

    def register(
        self, c_version: str, profile: str, seed: int, commit_sha: str
    ) -> str:
        run_id = self._compute_run_id(c_version, profile, seed, commit_sha)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, c_version, profile, seed, commit_sha) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, c_version, profile, seed, commit_sha),
            )
        return run_id

    def get(self, run_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"run_id not found: {run_id}")
            return dict(row)

    def register_output_hash(self, run_id: str, output_hash: str) -> None:
        """Record the SHA-256 hash of the canonical op output for ``run_id``.

        Idempotent on exact match — calling twice with the same
        ``(run_id, output_hash)`` is a silent no-op. Recording a
        *different* hash for the same ``run_id`` raises :
        the registry is the source of truth for R1 and a conflict is
        an empirical-axis violation signal that must surface to the
        caller (so the pilot / gate script can register the failure).

        Raises:
            KeyError: ``run_id`` is not registered in the ``runs`` table.
            ValueError: a different hash is already recorded for
                ``run_id`` — message includes the R1 tag and the
                offending ``run_id`` so the caller can log both.
        """
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            exists = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"run_id not found: {run_id}")
            existing = conn.execute(
                "SELECT output_hash FROM run_output_hashes WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != output_hash:
                    raise ValueError(
                        f"R1 violation: output_hash conflict for run_id "
                        f"{run_id!r}: recorded={existing[0]!r} "
                        f"attempted={output_hash!r}"
                    )
                return
            conn.execute(
                "INSERT INTO run_output_hashes (run_id, output_hash) "
                "VALUES (?, ?)",
                (run_id, output_hash),
            )

    def get_output_hash(self, run_id: str) -> str:
        """Return the recorded output hash for ``run_id``.

        Raises:
            KeyError: no hash has been recorded for ``run_id`` (either
                the run itself is unknown, or no
                ``register_output_hash`` call has landed for it yet).
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT output_hash FROM run_output_hashes WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"output_hash not found for run_id: {run_id}")
            hash_value: str = row[0]
            return hash_value
