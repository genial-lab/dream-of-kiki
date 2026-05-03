"""G4-quinto aggregator — load 3 step milestones and emit verdicts.

Per pre-reg sec 2-3 :

- **H5-A** : Jonckheere on step1 retention by arm, alpha = 0.0167.
- **H5-B** : Jonckheere on step2 retention by arm, alpha = 0.0167.
- **H5-C** : Welch two-sided between step3 (P_max with mog) and
  (P_max with none) at alpha = 0.0167. **Failure to reject** ->
  H5-C confirmed (RECOMBINE empty universalises across substrates).

Outputs :
    docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}

Option B handling : if ``step3_path`` is ``None`` (or missing on
disk), the aggregator emits a deferred H5-C block ; the
universality verdict is False by default and Step 3 is flagged
for a G4-sexto follow-up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STEP1 = (
    REPO_ROOT / "docs" / "milestones" / "g4-quinto-step1-2026-05-03.json"
)
DEFAULT_STEP2 = (
    REPO_ROOT / "docs" / "milestones" / "g4-quinto-step2-2026-05-03.json"
)
DEFAULT_STEP3 = (
    REPO_ROOT / "docs" / "milestones" / "g4-quinto-step3-2026-05-03.json"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-quinto-aggregate-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-quinto-aggregate-2026-05-03.md"
)


def aggregate_g4_quinto_verdict(
    step1_path: Path,
    step2_path: Path,
    step3_path: Path | None,
) -> dict[str, Any]:
    """Load the three step milestones and return aggregate verdicts.

    ``step3_path`` may be ``None`` to support compute Option B
    (Steps 1 + 2 only) ; the H5-C block is then deferred and the
    H5-C confirmed / universality flags resolve to False.
    """
    s1 = json.loads(step1_path.read_text())
    s2 = json.loads(step2_path.read_text())
    s3: dict[str, Any] | None
    if step3_path is not None and step3_path.exists():
        s3 = json.loads(step3_path.read_text())
    else:
        s3 = None

    h5a = s1["verdict"]["h5a_mlp_cifar"]
    h5b = s2["verdict"]["h5b_cnn_cifar"]
    h5a_confirmed = (
        not h5a.get("insufficient_samples", False)
        and bool(h5a.get("reject_h0"))
        and bool(h5a.get("monotonic_observed"))
    )
    h5b_confirmed = (
        not h5b.get("insufficient_samples", False)
        and bool(h5b.get("reject_h0"))
        and bool(h5b.get("monotonic_observed"))
    )

    h5c_block: dict[str, Any]
    ae_block: dict[str, Any]
    if s3 is None:
        h5c_block = {"deferred": True, "confirmed": False}
        ae_block = {"deferred": True}
        h5c_confirmed = False
    else:
        h5c = s3["verdict"]["h5c_recombine_strategy"]
        h5c_confirmed = (
            not h5c.get("insufficient_samples", False)
            and bool(h5c.get("h5c_recombine_empty_confirmed"))
        )
        h5c_block = {**h5c, "deferred": False, "confirmed": h5c_confirmed}
        ae_block = {**s3["verdict"].get("ae_observation", {}), "deferred": False}

    return {
        "h5a_benchmark_scale": {**h5a, "confirmed": h5a_confirmed},
        "h5b_architecture_scale": {**h5b, "confirmed": h5b_confirmed},
        "h5c_universality_recombine_empty": h5c_block,
        "ae_secondary": ae_block,
        "summary": {
            "h5a_confirmed": h5a_confirmed,
            "h5b_confirmed": h5b_confirmed,
            "h5c_confirmed": h5c_confirmed,
            "h5c_deferred": h5c_block.get("deferred", False),
            "h4c_to_h5c_universality": h5c_confirmed,
            "any_confirmed": (
                h5a_confirmed or h5b_confirmed or h5c_confirmed
            ),
            "all_three_confirmed": (
                h5a_confirmed and h5b_confirmed and h5c_confirmed
            ),
        },
    }


def _render_md(verdict: dict[str, Any]) -> str:
    h5a = verdict["h5a_benchmark_scale"]
    h5b = verdict["h5b_architecture_scale"]
    h5c = verdict["h5c_universality_recombine_empty"]
    ae = verdict["ae_secondary"]
    s = verdict["summary"]

    lines: list[str] = [
        "# G4-quinto aggregate verdict",
        "",
        "**Date** : 2026-05-03",
        "**Pre-registration** : "
        "[docs/osf-prereg-g4-quinto-pilot.md]"
        "(../osf-prereg-g4-quinto-pilot.md)",
        "",
        "## Summary",
        "",
        f"- H5-A (benchmark-scale) confirmed : **{s['h5a_confirmed']}**",
        f"- H5-B (architecture-scale) confirmed : **{s['h5b_confirmed']}**",
        (
            "- H5-C (universality of RECOMBINE-empty) confirmed : "
            f"**{s['h5c_confirmed']}**"
            + (" (deferred)" if s["h5c_deferred"] else "")
        ),
        (
            "- H4-C -> H5-C universality "
            "(FMNIST + CIFAR-CNN both empty) : "
            f"**{s['h4c_to_h5c_universality']}**"
        ),
        f"- All three confirmed : {s['all_three_confirmed']}",
        "",
        "## H5-A — benchmark-scale (MLP-on-CIFAR)",
        "",
    ]
    if h5a.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_min : {h5a['mean_p_min']:.4f}",
            f"- mean P_equ : {h5a['mean_p_equ']:.4f}",
            f"- mean P_max : {h5a['mean_p_max']:.4f}",
            f"- monotonic_observed : {h5a['monotonic_observed']}",
            f"- Jonckheere J : {h5a['j_statistic']:.4f}",
            (
                f"- one-sided p (alpha = "
                f"{h5a['alpha_per_test']:.4f}) : "
                f"{h5a['p_value']:.4f}"
            ),
            f"- reject_h0 : {h5a['reject_h0']}",
            f"- **H5-A confirmed** : {h5a['confirmed']}",
        ]
    lines += [
        "",
        "## H5-B — architecture-scale (CNN substrate)",
        "",
    ]
    if h5b.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_min : {h5b['mean_p_min']:.4f}",
            f"- mean P_equ : {h5b['mean_p_equ']:.4f}",
            f"- mean P_max : {h5b['mean_p_max']:.4f}",
            f"- monotonic_observed : {h5b['monotonic_observed']}",
            f"- Jonckheere J : {h5b['j_statistic']:.4f}",
            (
                f"- one-sided p (alpha = "
                f"{h5b['alpha_per_test']:.4f}) : "
                f"{h5b['p_value']:.4f}"
            ),
            f"- reject_h0 : {h5b['reject_h0']}",
            f"- **H5-B confirmed** : {h5b['confirmed']}",
        ]
    lines += [
        "",
        "## H5-C — universality of RECOMBINE-empty (CNN substrate)",
        "",
    ]
    if h5c.get("deferred"):
        lines.append(
            "DEFERRED (compute Option B chosen ; Step 3 will run "
            "in a G4-sexto follow-up)."
        )
    elif h5c.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_max (mog) : {h5c['mean_p_max_mog']:.4f}",
            f"- mean P_max (none) : {h5c['mean_p_max_none']:.4f}",
            (
                f"- Hedges' g (mog vs none) : "
                f"{h5c['hedges_g_mog_vs_none']:.4f}"
            ),
            f"- Welch t : {h5c['welch_t']:.4f}",
            (
                f"- Welch p two-sided (alpha = "
                f"{h5c['alpha_per_test']:.4f}) : "
                f"{h5c['welch_p_two_sided']:.4f}"
            ),
            (
                f"- fail_to_reject_h0 : "
                f"{h5c['fail_to_reject_h0']} -> "
                f"H5-C confirmed = {h5c['confirmed']}"
            ),
            "",
            (
                "*Honest reading* : Welch fail-to-reject = absence "
                "of evidence at this N for a difference between "
                "mog and none — under H5-C specifically, this "
                "**is** the predicted positive empirical claim "
                "that RECOMBINE adds nothing measurable beyond "
                "REPLAY+DOWNSCALE on the CNN substrate at "
                "CIFAR-10 scale."
            ),
        ]
    lines += [
        "",
        "### Secondary observation — AE strategy",
        "",
    ]
    if ae.get("deferred"):
        lines.append("DEFERRED")
    elif ae.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_max (ae) : {ae['mean_p_max_ae']:.4f}",
            f"- mean P_max (none) : {ae['mean_p_max_none']:.4f}",
            f"- Welch p two-sided : {ae['welch_p_two_sided']:.4f}",
        ]
    lines += [
        "",
        "## Verdict — DR-4 evidence",
        "",
        (
            "Per pre-reg §6 : EC stays PARTIAL across all outcomes ; "
            "FC stays at C-v0.12.0. If H5-C is confirmed, the "
            "partial refutation of DR-4 established by G4-ter and "
            "strengthened by G4-quater is **universalised** across "
            "FMNIST + CIFAR-CNN ; framework C claim 'richer ops "
            "yield richer consolidation' empirically refuted across "
            "two benchmarks. If H5-C is falsified, the refutation "
            "remains FMNIST-bound and CIFAR-CNN preserves the "
            "RECOMBINE contribution (scope-bound STABLE candidate)."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-quinto aggregator")
    parser.add_argument("--step1", type=Path, default=DEFAULT_STEP1)
    parser.add_argument("--step2", type=Path, default=DEFAULT_STEP2)
    parser.add_argument(
        "--step3",
        type=Path,
        default=DEFAULT_STEP3,
        help="Step 3 milestone path (Option A) or '/dev/null' / "
        "missing path (Option B, deferred).",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    step3: Path | None = args.step3
    # Treat /dev/null or any non-existing path as Option B deferral.
    if step3 is None or str(step3) == "/dev/null" or not step3.exists():
        step3 = None

    verdict = aggregate_g4_quinto_verdict(args.step1, args.step2, step3)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    args.out_md.write_text(_render_md(verdict))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    s = verdict["summary"]
    print(
        f"H5-A : {s['h5a_confirmed']}  "
        f"H5-B : {s['h5b_confirmed']}  "
        f"H5-C : {s['h5c_confirmed']}  "
        f"univ : {s['h4c_to_h5c_universality']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
