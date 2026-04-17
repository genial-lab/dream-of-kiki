"""Unit tests for run registry (SQLite-backed)."""
from datetime import datetime
from pathlib import Path

import pytest

from harness.storage.run_registry import RunRegistry


@pytest.fixture
def tmp_registry(tmp_path: Path) -> RunRegistry:
    db_path = tmp_path / "runs.db"
    return RunRegistry(db_path=db_path)


def test_register_run_creates_entry(tmp_registry: RunRegistry) -> None:
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    assert run_id is not None
    assert tmp_registry.get(run_id)["profile"] == "P_min"


def test_register_run_is_idempotent_for_same_inputs(tmp_registry: RunRegistry) -> None:
    args = dict(c_version="C-v0.1.0+STABLE", profile="P_equ", seed=1, commit_sha="def")
    run_id_1 = tmp_registry.register(**args)
    run_id_2 = tmp_registry.register(**args)
    assert run_id_1 == run_id_2  # Deterministic run_id for repro contract R1
