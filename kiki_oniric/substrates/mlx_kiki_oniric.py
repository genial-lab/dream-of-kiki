"""MLX kiki-oniric substrate (cycle-1 canonical implementation).

This module is a **namespace marker** for cycle 2. Existing cycle-1
code is not moved — it stays at its current locations under
`kiki_oniric/{core, dream, profiles, eval}/`. This module
documents which components belong to the MLX substrate and
provides convenient re-exports.

Cycle 2 adds `kiki_oniric.substrates.esnn_thalamocortical` as a
sibling, also implementing the framework C primitives and
profiles per the DR-3 Conformance Criterion.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§6.2 (DR-3 Conformance Criterion : signature typing + axiom
property tests passing + BLOCKING invariants enforceable).
"""
from __future__ import annotations

# Substrate identity
MLX_SUBSTRATE_NAME = "mlx_kiki_oniric"
# DualVer empirical axis `+PARTIAL` means : the substrate itself is
# green against the current framework-C (DR-3 conformance all three
# conditions pass on MLX, all BLOCKING invariants S1/S2/S3/I1 enforced,
# 180 tests passing ≥90% coverage). What is incomplete is the
# publication-track part of the cycle-2 evaluation matrix — Phase 3
# (cross-substrate ablation MLX vs E-SNN, C2.9-C2.12) and Phase 4
# (Paper 2 narrative C2.13-C2.16) are **scoped-deferred**. No contract
# test is failing or skipped on MLX. See framework-C spec §12.3 for
# formal PARTIAL semantics, CHANGELOG.md [C-v0.6.0+PARTIAL] for bump
# rationale, and STATUS.md for gate G9 CONDITIONAL-GO/PARTIAL status.
MLX_SUBSTRATE_VERSION = "C-v0.6.0+PARTIAL"


def mlx_substrate_components() -> dict[str, str]:
    """Return the canonical map of MLX substrate components.

    Each value is the dotted path to the cycle-1 implementation
    of the named primitive / operation / profile / guard. Cycle-2
    E-SNN substrate provides the same API at sibling paths.
    """
    return {
        # 8 typed Protocols (substrate-agnostic, defined in core)
        "primitives": "kiki_oniric.core.primitives",
        # 4 operations (skeleton + MLX backend per file)
        "replay": "kiki_oniric.dream.operations.replay",
        "downscale": "kiki_oniric.dream.operations.downscale",
        "restructure": "kiki_oniric.dream.operations.restructure",
        "recombine": "kiki_oniric.dream.operations.recombine",
        # 3 invariant guards
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        # Runtime + swap
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        # 3 profiles
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
        # Evaluation harness
        "eval_retained": "kiki_oniric.dream.eval_retained",
        "ablation": "kiki_oniric.eval.ablation",
        "statistics": "kiki_oniric.eval.statistics",
    }
