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
        """Create or migrate the schema.

        Three idempotent states are handled :

        1. Fresh DB (no ``run_output_hashes`` table) → create both
           ``runs`` and the new multi-artifact ``run_output_hashes``.
        2. Legacy DB (single-hash-per-run schema from commit
           ``ebf1410``, no ``artifact_name`` column) → run the
           rename-copy-drop migration to the new schema.
        3. DB already on the new schema → no-op.
        """
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
            self._migrate_run_output_hashes_schema(conn)

    @staticmethod
    def _migrate_run_output_hashes_schema(conn: sqlite3.Connection) -> None:
        """Ensure ``run_output_hashes`` is on the multi-artifact schema.

        Detects the current state via ``PRAGMA table_info`` and applies
        the minimal action : create-fresh, migrate-from-v1, or no-op.
        The migration runs in a single transaction (the caller's
        ``with conn:`` block) so a crash midway leaves the legacy
        table intact.
        """
        columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(run_output_hashes)"
            ).fetchall()
        ]
        if not columns:
            # State 1 — fresh DB.
            conn.execute("""
                CREATE TABLE run_output_hashes (
                    run_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    hash_type TEXT NOT NULL DEFAULT 'sha256',
                    output_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (run_id, artifact_name),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)
            return
        if "artifact_name" in columns:
            # State 3 — already migrated.
            return
        # State 2 — legacy single-hash schema, migrate.
        conn.execute(
            "ALTER TABLE run_output_hashes RENAME TO run_output_hashes_v1"
        )
        conn.execute("""
            CREATE TABLE run_output_hashes (
                run_id TEXT NOT NULL,
                artifact_name TEXT NOT NULL,
                hash_type TEXT NOT NULL DEFAULT 'sha256',
                output_hash TEXT NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (run_id, artifact_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        conn.execute("""
            INSERT INTO run_output_hashes
                (run_id, artifact_name, hash_type, output_hash, recorded_at)
            SELECT run_id, 'canonical', 'sha256', output_hash, recorded_at
            FROM run_output_hashes_v1
        """)
        conn.execute("DROP TABLE run_output_hashes_v1")

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

    def register_output_hash(
        self,
        run_id: str,
        output_hash: str,
        *,
        artifact_name: str = "canonical",
        hash_type: str = "sha256",
    ) -> None:
        """Record a hash for ``(run_id, artifact_name)`` (R1).

        Idempotent on the exact tuple
        ``(run_id, artifact_name, output_hash, hash_type)`` : a second
        call with identical values is a silent no-op so retry / resume
        loops do not require extra bookkeeping.

        A conflict on either ``output_hash`` or ``hash_type`` for the
        same ``(run_id, artifact_name)`` raises :
        the registry is the source of truth for R1 and a conflict is
        an empirical-axis violation signal that must surface to the
        caller (so the pilot / gate script can register the failure).

        Backward-compat : calling without kwargs targets
        ``artifact_name='canonical'``, ``hash_type='sha256'``.

        Raises:
            KeyError: ``run_id`` is not registered in the ``runs`` table.
            ValueError: a different hash or hash_type is already
                recorded for ``(run_id, artifact_name)`` — message
                includes the R1 tag, ``run_id``, and ``artifact_name``
                so the caller can log all three.
        """
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            exists = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"run_id not found: {run_id}")
            existing = conn.execute(
                "SELECT output_hash, hash_type FROM run_output_hashes "
                "WHERE run_id = ? AND artifact_name = ?",
                (run_id, artifact_name),
            ).fetchone()
            if existing is not None:
                if existing[0] != output_hash or existing[1] != hash_type:
                    raise ValueError(
                        f"R1 violation: output_hash conflict for run_id "
                        f"{run_id!r} artifact_name {artifact_name!r}: "
                        f"recorded=({existing[0]!r}, {existing[1]!r}) "
                        f"attempted=({output_hash!r}, {hash_type!r})"
                    )
                return
            conn.execute(
                "INSERT INTO run_output_hashes "
                "(run_id, artifact_name, hash_type, output_hash) "
                "VALUES (?, ?, ?, ?)",
                (run_id, artifact_name, hash_type, output_hash),
            )

    def get_output_hash(
        self,
        run_id: str,
        *,
        artifact_name: str = "canonical",
    ) -> str:
        """Return the recorded hash for ``(run_id, artifact_name)``.

        Backward-compat : calling without kwargs returns the hash
        registered under ``artifact_name='canonical'``.

        Raises:
            KeyError: no hash has been recorded for
                ``(run_id, artifact_name)``.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT output_hash FROM run_output_hashes "
                "WHERE run_id = ? AND artifact_name = ?",
                (run_id, artifact_name),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"output_hash not found for run_id={run_id!r} "
                    f"artifact_name={artifact_name!r}"
                )
            hash_value: str = row[0]
            return hash_value

    def list_output_hashes(self, run_id: str) -> dict[str, str]:
        """Return ``{artifact_name: output_hash}`` for ``run_id``.

        Empty dict if ``run_id`` has no registered artifacts (or is
        unknown entirely — documented asymmetry with
        ``get_output_hash``, which raises). Hashes are returned in
        ``artifact_name`` order.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT artifact_name, output_hash FROM run_output_hashes "
                "WHERE run_id = ? ORDER BY artifact_name",
                (run_id,),
            ).fetchall()
            return {name: output_hash for name, output_hash in rows}
