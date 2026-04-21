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


def test_run_id_has_128_bit_entropy(tmp_registry: RunRegistry) -> None:
    # 128 bits = 32 hex chars — collision negligible at any scale
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_equ",
        seed=1,
        commit_sha="xyz",
    )
    assert len(run_id) == 32
    assert all(c in "0123456789abcdef" for c in run_id)


# --------------------------------------------------------------------------
# Output-hash API — second half of R1 (recorded output is stable)
# --------------------------------------------------------------------------


def test_register_output_hash_stores_and_roundtrips(
    tmp_registry: RunRegistry,
) -> None:
    """``register_output_hash`` + ``get_output_hash`` round-trip."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    output_hash = "a" * 64
    tmp_registry.register_output_hash(run_id, output_hash)
    assert tmp_registry.get_output_hash(run_id) == output_hash


def test_register_output_hash_idempotent_on_exact_match(
    tmp_registry: RunRegistry,
) -> None:
    """Registering the same (run_id, hash) twice is a silent no-op."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    output_hash = "b" * 64
    tmp_registry.register_output_hash(run_id, output_hash)
    # Second call must not raise and must not mutate the stored value.
    tmp_registry.register_output_hash(run_id, output_hash)
    assert tmp_registry.get_output_hash(run_id) == output_hash


def test_register_output_hash_raises_on_conflict(
    tmp_registry: RunRegistry,
) -> None:
    """Conflicting hash for an existing run_id → R1-labelled ValueError."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    tmp_registry.register_output_hash(run_id, "c" * 64)
    with pytest.raises(ValueError) as excinfo:
        tmp_registry.register_output_hash(run_id, "d" * 64)
    message = str(excinfo.value)
    assert "R1" in message
    assert run_id in message


def test_register_output_hash_raises_on_unknown_run_id(
    tmp_registry: RunRegistry,
) -> None:
    """Unknown run_id must not silently acquire an output hash."""
    with pytest.raises(KeyError):
        tmp_registry.register_output_hash("deadbeef" * 4, "e" * 64)


def test_get_output_hash_raises_when_missing(
    tmp_registry: RunRegistry,
) -> None:
    """Registered run without a recorded hash → KeyError."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    with pytest.raises(KeyError):
        tmp_registry.get_output_hash(run_id)
