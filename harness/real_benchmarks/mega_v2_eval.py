"""mega-v2 80/20 self-eval loader (cycle-3 C3.1).

mega-v2 is the FineFab training mega-corpus (498K examples across
25 domains, cf. user memory ``mega dataset v2``). Cycle-3 uses it
as the ``retained-benchmark`` replacement : at cycle start we carve
the full corpus into an 80/20 train / eval split (seeded, per R1)
and never touch the eval shard during dream-episodes. This loader
materialises that split deterministically.

Unlike MMLU + HellaSwag, mega-v2 is an internal artefact (no HF
pin) — the ``DatasetPin`` registry therefore has no entry ; the
loader is anchored instead on a local JSONL path + SHA-256.

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.1
  docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md §5
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from harness.real_benchmarks import MissingLocalDatasetError


@dataclass(frozen=True)
class MegaV2EvalRecord:
    """Frozen mega-v2 record used by dream-ops + self-eval.

    Fields
    ------
    id
        Stable record identifier (``mv2-{index:04d}`` convention).
    context
        Input prompt / context string.
    expected
        Expected continuation / answer string.
    domain
        One of the 25 mega-v2 domains (cf. ``SYNTHETIC_DOMAINS`` in
        :mod:`harness.benchmarks.mega_v2.adapter` for the cycle-1/2
        placeholder taxonomy).
    """

    id: str
    context: str
    expected: str
    domain: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class MegaV2EvalLoader:
    """Load mega-v2 + carve 80/20 splits deterministically.

    Parameters
    ----------
    local_path
        Filesystem path to the mega-v2 JSONL.
    expected_sha256
        Optional SHA-256 check (same semantics as MMLU loader).
    """

    def __init__(
        self,
        *,
        local_path: Path,
        expected_sha256: str | None = None,
    ) -> None:
        if not local_path.exists():
            raise MissingLocalDatasetError(
                f"mega-v2 fixture not found at {local_path!s} ; "
                "pass a pre-materialised JSONL export (internal "
                "artefact — no HF pin)."
            )
        self._path = local_path
        self._actual_sha256 = _hash_file(local_path)
        if (
            expected_sha256 is not None
            and expected_sha256 != self._actual_sha256
        ):
            raise ValueError(
                f"sha256 mismatch on {local_path!s}: expected "
                f"{expected_sha256!r}, got {self._actual_sha256!r}"
            )
        self._hash_verified = expected_sha256 is not None

    @property
    def local_path(self) -> Path:
        return self._path

    @property
    def hash_verified(self) -> bool:
        return self._hash_verified

    def local_file_sha256(self) -> str:
        return self._actual_sha256

    def _iter_raw(self) -> Iterator[dict]:
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _record_from_raw(self, row: dict) -> MegaV2EvalRecord:
        return MegaV2EvalRecord(
            id=str(row["id"]),
            context=str(row["context"]),
            expected=str(row["expected"]),
            domain=str(row.get("domain", "unknown")),
        )

    def iter_records(self) -> Iterator[MegaV2EvalRecord]:
        """Yield records in fixture order."""
        for row in self._iter_raw():
            yield self._record_from_raw(row)

    def train_eval_split(
        self,
        *,
        eval_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple[list[MegaV2EvalRecord], list[MegaV2EvalRecord]]:
        """Carve an 80/20 (configurable) train/eval split.

        Returns ``(train, eval_)`` disjoint lists.
        ``eval_fraction`` must lie strictly in ``(0, 1)``.
        Deterministic under ``(path_content, seed)`` — same file
        digest and same seed always give byte-identical partitions.
        """
        if not 0.0 < eval_fraction < 1.0:
            raise ValueError(
                f"eval_fraction must be in (0, 1), got "
                f"{eval_fraction}"
            )
        records = [self._record_from_raw(r) for r in self._iter_raw()]
        n_total = len(records)
        if n_total < 2:
            raise ValueError(
                f"mega-v2 split needs ≥2 records, got {n_total}"
            )
        n_eval = int(round(n_total * eval_fraction))
        # Guard against edge-cases where rounding collapses a split.
        n_eval = max(1, min(n_eval, n_total - 1))
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        eval_idx = sorted(indices[:n_eval])
        train_idx = sorted(indices[n_eval:])
        train = [records[i] for i in train_idx]
        eval_ = [records[i] for i in eval_idx]
        return train, eval_

    def get_seeded_sample(
        self, seed: int, n: int
    ) -> list[MegaV2EvalRecord]:
        """Return ``n`` records drawn via a seeded permutation."""
        records = [self._record_from_raw(r) for r in self._iter_raw()]
        if n > len(records):
            raise ValueError(
                f"requested {n} records but fixture has "
                f"{len(records)}"
            )
        rng = random.Random(seed)
        return rng.sample(records, n)


# --------------------------------------------------------------------------
# Evaluator (cycle-3 C3.8 Phase A) — mega-v2 retained self-eval.
# Measures negative-log-likelihood of ``expected`` given ``context``
# averaged across records. Lower NLL = better retention. Accuracy is
# reported as ``1 - nll/nll_0`` for a coarse ``[0,1]``-normalised score.
# --------------------------------------------------------------------------


_DEFAULT_MEGA_V2_FALLBACK = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "mega_v2_sanity.jsonl"
)


def _mega_v2_default_fallback_records() -> list[dict]:
    """8-row hand-authored mega-v2-schema fallback.

    Sized small deliberately — the full mega-v2 at 498 K rows must
    come from the FineFab artefact path. The fallback exists so
    the pilot pipeline runs end-to-end on a host that has no
    internal export available yet.
    """
    return [
        {
            "id": "mv2-fb-0001",
            "context": "Python type hints are introduced via the",
            "expected": " typing module.",
            "domain": "software",
        },
        {
            "id": "mv2-fb-0002",
            "context": "The capital city of Japan is",
            "expected": " Tokyo.",
            "domain": "world_facts",
        },
        {
            "id": "mv2-fb-0003",
            "context": "Force equals mass times",
            "expected": " acceleration.",
            "domain": "physics",
        },
        {
            "id": "mv2-fb-0004",
            "context": "HTTP 404 means",
            "expected": " not found.",
            "domain": "networking",
        },
        {
            "id": "mv2-fb-0005",
            "context": "The chemical symbol for iron is",
            "expected": " Fe.",
            "domain": "chemistry",
        },
        {
            "id": "mv2-fb-0006",
            "context": "In binary 1010 equals decimal",
            "expected": " 10.",
            "domain": "mathematics",
        },
        {
            "id": "mv2-fb-0007",
            "context": "SQL SELECT is used to",
            "expected": " retrieve data.",
            "domain": "databases",
        },
        {
            "id": "mv2-fb-0008",
            "context": "The Eiffel Tower is located in",
            "expected": " Paris.",
            "domain": "world_facts",
        },
    ]


def _load_mega_v2_records(
    n_samples: int,
    seed: int,
    *,
    fixture_path: Path | None = None,
) -> list[MegaV2EvalRecord]:
    """Materialise ``n_samples`` mega-v2 records.

    Priority : caller path → committed fallback fixture → in-module
    hand-authored fallback. No HF path because mega-v2 is an
    internal artefact, not a public dataset.
    """
    def _materialise(raws: list[dict]) -> list[MegaV2EvalRecord]:
        rng = random.Random(seed)
        rng.shuffle(raws)
        if len(raws) >= n_samples:
            selected = raws[:n_samples]
        else:
            selected = [
                raws[i % len(raws)] for i in range(n_samples)
            ]
        return [
            MegaV2EvalRecord(
                id=str(row["id"]),
                context=str(row["context"]),
                expected=str(row["expected"]),
                domain=str(row.get("domain", "unknown")),
            )
            for row in selected
        ]

    target = fixture_path or _DEFAULT_MEGA_V2_FALLBACK
    if target.exists():
        raws = []
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raws.append(json.loads(line))
        return _materialise(raws)

    return _materialise(_mega_v2_default_fallback_records())


def _per_token_nll(
    model_callable,
    tokenizer,
    context_ids: list[int],
    expected_ids: list[int],
) -> float:
    """Return the per-token NLL of ``expected_ids`` given ``context_ids``."""
    import mlx.core as mx
    import numpy as np

    if not expected_ids:
        return 0.0
    full = context_ids + expected_ids
    tokens = mx.array([full])
    mx.random.seed(0)
    logits = model_callable(tokens)
    # Widen bf16 → fp32 on the MLX side : numpy has no bf16 dtype.
    logits_fp32 = logits[0].astype(mx.float32)
    mx.eval(logits_fp32)
    logits_np = np.asarray(logits_fp32).astype(np.float32)
    start = len(context_ids) - 1
    total = 0.0
    n = 0
    for i, tok_id in enumerate(expected_ids):
        pos = start + i
        if pos < 0 or pos >= logits_np.shape[0]:
            total += 20.0  # saturating penalty for degenerate ctx
            n += 1
            continue
        row = logits_np[pos]
        m = float(np.max(row))
        logsumexp = m + float(np.log(np.sum(np.exp(row - m)) + 1e-30))
        total += logsumexp - float(row[int(tok_id)])
        n += 1
    return total / max(n, 1)


def evaluate_mega_v2(
    model,
    tokenizer,
    *,
    n_samples: int = 100,
    seed: int = 0,
    fixture_path: Path | None = None,
) -> dict[str, float]:
    """Run the mega-v2 retained-accuracy eval against ``model``.

    Computes per-token NLL of each record's ``expected`` continuation
    given its ``context`` ; averages across records ; reports both
    the raw ``nll`` and a coarse ``accuracy = exp(-nll)`` so the
    pilot's composite score combines MMLU + HellaSwag + mega-v2 on a
    comparable [0, 1] scale. Deterministic under
    ``(model_weights, tokenizer, fixture, seed)``.

    Returns ``{"accuracy": float, "nll": float, "n": int}``.
    """
    import math

    records = _load_mega_v2_records(
        n_samples, seed, fixture_path=fixture_path
    )
    forward = model.model if hasattr(model, "model") else model

    total_nll = 0.0
    n = 0
    for record in records:
        try:
            ctx_ids = tokenizer.encode(record.context)
        except TypeError:
            ctx_ids = tokenizer.encode(
                record.context, add_special_tokens=True
            )
        try:
            exp_ids = tokenizer.encode(record.expected)
        except TypeError:
            exp_ids = tokenizer.encode(
                record.expected, add_special_tokens=False
            )
        if not exp_ids:
            exp_ids = [0]
        nll = _per_token_nll(forward, tokenizer, ctx_ids, exp_ids)
        total_nll += nll
        n += 1
    mean_nll = total_nll / max(n, 1)
    # exp(-nll) in [0, 1] — a coarse accuracy proxy so the pilot's
    # composite score combines cleanly with MMLU/HellaSwag.
    accuracy = math.exp(-mean_nll) if mean_nll < 50 else 0.0
    return {"accuracy": accuracy, "nll": mean_nll, "n": n}


__all__ = [
    "MegaV2EvalLoader",
    "MegaV2EvalRecord",
    "evaluate_mega_v2",
]
