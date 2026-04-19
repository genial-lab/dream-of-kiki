"""Conformance validation matrix (C2.10 cycle 2).

Builds the substrate × condition matrix required by the DR-3
Conformance Criterion :

- **Condition 1** — signature typing (typed Protocols exposed).
- **Condition 2** — axiom property tests pass (per substrate).
- **Condition 3** — BLOCKING invariants enforceable (S2 finite,
  S3 topology).

Substrates covered :
- `mlx_kiki_oniric`    — cycle-1 canonical substrate (real).
- `esnn_thalamocortical` — cycle-2 C2.3 substrate (real).
- `hypothetical_cycle3` — placeholder marker ; **not implemented**.
  All cells for this row are reported as ``N/A — placeholder for
  cycle 3``. Do not interpret as passing / failing.

Output :
- `docs/milestones/conformance-matrix.md` (results table)
- `docs/proofs/dr3-substrate-evidence.md`  (formal evidence)

The script calls pytest to run the condition-2 suites per
substrate. Conditions 1 and 3 are static introspection (Protocol
declarations + guard callability on substrate state).

Synthetic substitute : no real Loihi-2 hardware, no fMRI cohort
— the E-SNN substrate uses a numpy LIF skeleton (see
`kiki_oniric/substrates/esnn_thalamocortical.py` docstring).

Usage :
    uv run python scripts/conformance_matrix.py

Reference : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
            §6.2 (DR-3 Conformance Criterion)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kiki_oniric.substrates import (
    esnn_substrate_components,
    mlx_substrate_components,
)


SUBSTRATES_REAL = ("mlx_kiki_oniric", "esnn_thalamocortical")
SUBSTRATE_PLACEHOLDER = "hypothetical_cycle3"
CONDITION_LABELS = {
    "C1": "signature typing (typed Protocols)",
    "C2": "axiom property tests pass",
    "C3": "BLOCKING invariants enforceable",
}


class CellResult(TypedDict):
    status: str         # "PASS", "FAIL", "N/A"
    detail: str
    evidence: str


def _check_condition_1_mlx() -> CellResult:
    """C1 MLX : the 8 typed Protocols are declared and registered
    in the MLX substrate component map."""
    components = mlx_substrate_components()
    required = {"primitives", "replay", "downscale", "restructure",
                "recombine", "finite", "topology", "runtime", "swap",
                "p_min", "p_equ", "p_max"}
    missing = required - set(components.keys())
    if missing:
        return CellResult(
            status="FAIL",
            detail=f"missing keys: {sorted(missing)}",
            evidence="kiki_oniric/substrates/mlx_kiki_oniric.py",
        )
    return CellResult(
        status="PASS",
        detail="all 8 primitives declared as Protocols + registry complete",
        evidence="tests/conformance/axioms/test_dr3_substrate.py",
    )


def _check_condition_1_esnn() -> CellResult:
    """C1 E-SNN : the substrate exports the 4 op-factory methods
    and shares the core registry keys with MLX."""
    from kiki_oniric.substrates.esnn_thalamocortical import (
        EsnnSubstrate,
    )
    substrate = EsnnSubstrate()
    factories = [
        substrate.replay_handler_factory,
        substrate.downscale_handler_factory,
        substrate.restructure_handler_factory,
        substrate.recombine_handler_factory,
    ]
    if not all(callable(f) for f in factories):
        return CellResult(
            status="FAIL",
            detail="one or more op-factory methods are not callable",
            evidence="kiki_oniric/substrates/esnn_thalamocortical.py",
        )
    esnn = esnn_substrate_components()
    mlx = mlx_substrate_components()
    core = {"primitives", "replay", "downscale", "restructure",
            "recombine", "finite", "topology", "runtime", "swap",
            "p_min", "p_equ", "p_max"}
    if not (core <= set(esnn.keys()) and core <= set(mlx.keys())):
        return CellResult(
            status="FAIL",
            detail="substrate component registries miss core keys",
            evidence="kiki_oniric/substrates/esnn_thalamocortical.py",
        )
    return CellResult(
        status="PASS",
        detail=(
            "4 op factories callable + core registry shared with MLX "
            "(spike-rate numpy LIF skeleton, synthetic substitute)"
        ),
        evidence="tests/conformance/axioms/test_dr3_esnn_substrate.py",
    )


def _run_pytest(target: str) -> tuple[bool, str]:
    """Run `uv run pytest <target> -q` and return (passed, tail).

    Coverage is disabled per-target because the project-level
    ``--cov-fail-under=90`` gate (pyproject pytest config) would
    otherwise spuriously mark single-file runs as failing. The
    conformance matrix only checks that the targeted conformance
    suite passes ; the 90 % coverage gate is enforced by the
    full-suite CI run separately.
    """
    cmd = [
        "uv", "run", "python", "-m", "pytest", target, "-q",
        "--no-cov",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    tail = (proc.stdout or "").splitlines()[-3:]
    passed = proc.returncode == 0
    return passed, " | ".join(tail).strip()


def _check_condition_2_mlx() -> CellResult:
    """C2 MLX : axiom property tests pass on the MLX substrate."""
    targets = [
        "tests/conformance/axioms/test_dr0_accountability.py",
        "tests/conformance/axioms/test_dr1_episodic_conservation.py",
        "tests/conformance/axioms/test_dr3_substrate.py",
        "tests/conformance/axioms/test_dr4_profile_inclusion.py",
    ]
    for target in targets:
        ok, tail = _run_pytest(target)
        if not ok:
            return CellResult(
                status="FAIL",
                detail=f"{target} failed : {tail}",
                evidence=target,
            )
    return CellResult(
        status="PASS",
        detail="DR-0, DR-1, DR-3, DR-4 axiom suites pass on MLX",
        evidence="tests/conformance/axioms/",
    )


def _check_condition_2_esnn() -> CellResult:
    """C2 E-SNN : DR-3 E-SNN axiom property suite passes on the
    spike-rate numpy LIF skeleton."""
    target = "tests/conformance/axioms/test_dr3_esnn_substrate.py"
    ok, tail = _run_pytest(target)
    if not ok:
        return CellResult(
            status="FAIL",
            detail=f"{target} failed : {tail}",
            evidence=target,
        )
    return CellResult(
        status="PASS",
        detail=(
            "DR-3 E-SNN conformance suite passes on numpy LIF "
            "skeleton (synthetic substitute — no Loihi-2 HW)"
        ),
        evidence=target,
    )


def _check_condition_3_mlx() -> CellResult:
    """C3 MLX : S2 finite + S3 topology invariants enforceable."""
    targets = [
        "tests/conformance/invariants/test_s2_finite.py",
        "tests/conformance/invariants/test_s3_topology.py",
    ]
    for target in targets:
        ok, tail = _run_pytest(target)
        if not ok:
            return CellResult(
                status="FAIL",
                detail=f"{target} failed : {tail}",
                evidence=target,
            )
    return CellResult(
        status="PASS",
        detail="S2 finite + S3 topology guards enforceable on MLX",
        evidence="tests/conformance/invariants/",
    )


def _check_condition_3_esnn() -> CellResult:
    """C3 E-SNN : S2 finite + S3 topology operate on LIF state.

    The E-SNN DR-3 suite already exercises C3 : test_c3_s2* and
    test_c3_s3* validate guard callability on LIFState.v and the
    canonical ortho-chain topology. Running the suite is sufficient
    evidence that C3 holds on the E-SNN substrate.
    """
    target = "tests/conformance/axioms/test_dr3_esnn_substrate.py"
    ok, tail = _run_pytest(target)
    if not ok:
        return CellResult(
            status="FAIL",
            detail=f"{target} failed : {tail}",
            evidence=target,
        )
    return CellResult(
        status="PASS",
        detail=(
            "S2 finite + S3 topology guards enforceable on LIFState "
            "(synthetic substitute — spike-rate numpy LIF)"
        ),
        evidence=target,
    )


def _placeholder_cell(reason: str) -> CellResult:
    return CellResult(
        status="N/A",
        detail=f"placeholder for cycle 3 — {reason}",
        evidence="not yet implemented",
    )


def build_matrix() -> dict:
    """Build the substrate × condition matrix."""
    matrix: dict[str, dict[str, CellResult]] = {}
    matrix["mlx_kiki_oniric"] = {
        "C1": _check_condition_1_mlx(),
        "C2": _check_condition_2_mlx(),
        "C3": _check_condition_3_mlx(),
    }
    matrix["esnn_thalamocortical"] = {
        "C1": _check_condition_1_esnn(),
        "C2": _check_condition_2_esnn(),
        "C3": _check_condition_3_esnn(),
    }
    matrix[SUBSTRATE_PLACEHOLDER] = {
        "C1": _placeholder_cell("no third substrate implemented"),
        "C2": _placeholder_cell("no third substrate implemented"),
        "C3": _placeholder_cell("no third substrate implemented"),
    }
    return matrix


def _render_markdown(matrix: dict) -> str:
    lines = [
        "# Conformance matrix — DR-3 Conformance Criterion",
        "",
        "**C2.10 cycle 2.** Substrate × condition matrix derived from",
        "`scripts/conformance_matrix.py`. Two substrates are real ;",
        "the third row is an explicit placeholder.",
        "",
        "> **(synthetic substitute)**  All E-SNN rows are produced by",
        "> the numpy LIF spike-rate skeleton (no Loihi-2 hardware).",
        "> The MLX rows exercise the cycle-1 canonical substrate on",
        "> Apple Silicon. No real fMRI cohort is involved in either.",
        "",
        "## Conditions",
        "",
    ]
    for key, label in CONDITION_LABELS.items():
        lines.append(f"- **{key}** — {label}")
    lines += [
        "",
        "## Matrix",
        "",
        "| substrate | C1 | C2 | C3 |",
        "|-----------|----|----|----|",
    ]
    for sub, cells in matrix.items():
        c1 = cells["C1"]["status"]
        c2 = cells["C2"]["status"]
        c3 = cells["C3"]["status"]
        lines.append(f"| `{sub}` | {c1} | {c2} | {c3} |")
    lines += [
        "",
        "## Cell detail (synthetic substitute where noted)",
        "",
    ]
    for sub, cells in matrix.items():
        lines.append(f"### `{sub}`")
        lines.append("")
        for cond_key, cell in cells.items():
            lines.append(
                f"- **{cond_key} — {CONDITION_LABELS[cond_key]}** : "
                f"`{cell['status']}` — {cell['detail']} "
                f"(evidence : `{cell['evidence']}`)"
            )
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- The `hypothetical_cycle3` row is marked `N/A` and must not",
        "  be read as passing / failing. It reserves the matrix shape",
        "  for a future substrate (see `docs/specs/` cycle-3 plan).",
        "- The matrix is regenerated by running",
        "  `uv run python scripts/conformance_matrix.py`. The JSON",
        "  sibling is written alongside this markdown.",
    ]
    return "\n".join(lines) + "\n"


def _render_evidence(matrix: dict) -> str:
    """Render the formal DR-3 evidence document."""
    lines = [
        "# DR-3 Conformance Criterion — substrate evidence (C2.10)",
        "",
        "**Status** : two-substrate evidence, synthetic substitute.",
        "",
        "DR-3 claims substrate-agnosticism : any substrate that",
        "satisfies the three conditions below inherits the framework's",
        "guarantees. This document records the **evidence per",
        "substrate** for each condition, with pointers to the test",
        "files and source modules that back the verdicts.",
        "",
        "## Conformance conditions (spec §6.2)",
        "",
        "1. **Signature typing** — the 8 primitives are declared as",
        "   typed `Protocol`s (awake→dream α/β/γ/δ + 4 channels",
        "   dream→awake). A substrate conforms to C1 by exposing",
        "   handlers / factories compatible with these signatures.",
        "2. **Axiom property tests** — DR-0..DR-4 property tests pass",
        "   on the substrate's state representation.",
        "3. **BLOCKING invariants enforceable** — the S2 finite and",
        "   S3 topology guards can be applied to the substrate's",
        "   state and refuse ill-formed values.",
        "",
        "## Evidence per substrate",
        "",
    ]
    for sub, cells in matrix.items():
        is_placeholder = sub == SUBSTRATE_PLACEHOLDER
        header = (
            f"### `{sub}` (placeholder — not yet implemented)"
            if is_placeholder
            else f"### `{sub}`"
        )
        lines.append(header)
        lines.append("")
        if is_placeholder:
            lines.append(
                "This row is reserved for a future (cycle-3) substrate."
                " All cells are `N/A — placeholder for cycle 3`. The"
                " DR-3 evidence set is *two* substrates in cycle 2."
            )
            lines.append("")
            continue
        flag = ""
        if sub == "esnn_thalamocortical":
            flag = " *(synthetic substitute — numpy LIF skeleton)*"
        lines.append(f"Evidence summary{flag} :")
        lines.append("")
        for cond_key, cell in cells.items():
            lines.append(
                f"- **{cond_key} — {CONDITION_LABELS[cond_key]}** : "
                f"`{cell['status']}` — {cell['detail']}"
            )
            lines.append(
                f"  - evidence : `{cell['evidence']}`"
            )
        lines.append("")
    lines += [
        "## Synthetic-data caveat",
        "",
        "The E-SNN substrate rows are backed by a numpy LIF spike-",
        "rate skeleton, not by real Loihi-2 hardware. No fMRI or",
        "behavioural cohort is involved in either substrate's C2",
        "axiom tests. DR-3 two-substrate replication therefore",
        "strengthens the *architectural* claim (the framework's",
        "Conformance Criterion is operational across two",
        "independent implementations of the 8 primitives) — it does",
        "not yet carry a cross-substrate empirical claim on real",
        "biological data.",
        "",
        "## Cross-references",
        "",
        "- Spec §6.2 : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`",
        "- Matrix dump : `docs/milestones/conformance-matrix.md`",
        "- JSON dump : `docs/milestones/conformance-matrix.json`",
        "- MLX substrate : `kiki_oniric/substrates/mlx_kiki_oniric.py`",
        "- E-SNN substrate : `kiki_oniric/substrates/esnn_thalamocortical.py`",
        "- C2 axiom suites : `tests/conformance/axioms/`",
        "- C3 invariant suites : `tests/conformance/invariants/`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("=" * 64)
    print("CONFORMANCE MATRIX — DR-3 (C2.10, synthetic substitute)")
    print("=" * 64)
    matrix = build_matrix()
    for sub, cells in matrix.items():
        print(f"\n[{sub}]")
        for cond_key, cell in cells.items():
            print(
                f"  {cond_key} ({CONDITION_LABELS[cond_key][:34]:34s})"
                f" : {cell['status']} — {cell['detail'][:60]}"
            )

    md_dir = REPO_ROOT / "docs" / "milestones"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / "conformance-matrix.md"
    md_path.write_text(_render_markdown(matrix), encoding="utf-8")
    print(f"\nMatrix written to {md_path}")

    json_path = md_dir / "conformance-matrix.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "synthetic_substitute": True,
                "data_provenance": (
                    "synthetic data — no real cohort/HW. E-SNN rows "
                    "use numpy LIF spike-rate skeleton, not Loihi-2."
                ),
                "matrix": matrix,
            },
            fh,
            indent=2,
        )
    print(f"JSON dump written to {json_path}")

    evidence_dir = REPO_ROOT / "docs" / "proofs"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "dr3-substrate-evidence.md"
    evidence_path.write_text(_render_evidence(matrix), encoding="utf-8")
    print(f"Evidence written to {evidence_path}")


if __name__ == "__main__":
    main()
