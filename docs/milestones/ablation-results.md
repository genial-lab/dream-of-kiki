# G4 Ablation Results — Synthetic mega-v2

**Date** : 2026-04-18 (S15.3)
**Benchmark** : mega-v2 synthetic placeholder
**Items** : 500 (25 domains × 20)
**Seeds** : 42, 123, 7

## Caveat

These results validate the **measurement + statistical pipeline**
end-to-end (synthetic predictors at 3 accuracy levels). NOT real
P_min/P_equ ablation evidence. Real ablation requires S16+ MLX
inference wiring + real mega-v2 dataset access.

## Accuracy table

| Profile  | Seed 42 | Seed 123 | Seed 7 |
|----------|---------|----------|--------|
| baseline | 0.500   | 0.500    | 0.500  |
| P_min    | 0.700   | 0.700    | 0.700  |
| P_equ    | 0.850   | 0.850    | 0.850  |

## Hypothesis tests (Bonferroni α = 0.0125)

| Hypothesis | Test | p-value | Result |
|-----------|------|---------|--------|
| H1 (forgetting) | Welch's t (one-sided) | 0.0000 | PASS |
| H2 (equivalence) | TOST | 0.0000 | PASS |
| H3 (monotonic) | Jonckheere-Terpstra | 0.0248 | fail |
| H4 (energy budget) | one-sample t | 0.0101 | PASS |

## Gate decision

**Criterion** : ≥2 hypotheses significant at α = 0.0125

**Result** : **PASS** — 3/4 significant

## Implications

PASS (synthetic) :

- Statistical pipeline validated end-to-end : Welch, TOST,
  Jonckheere, one-sample t all wire correctly through the
  AblationRunner / mega-v2 adapter / RetainedBenchmark stack.
- Path to real GO-FULL clear (S16+ real MLX inference + real
  mega-v2).
- G4 stays GO-CONDITIONAL pending real data.

Note on H3 : Jonckheere-Terpstra rejected at α = 0.05 but not at
the Bonferroni-corrected α = 0.0125. With identical accuracy
across seeds (zero within-group variance) and only two ordered
groups, the test is right at the corrected threshold. Real-data
seeds with genuine variance should resolve this cleanly.

## Raw data

JSON dump at `docs/milestones/ablation-results.json`.

## Next steps

- S16+ : wire `PMinProfile.swap_now` + `PEquProfile.swap_now`
  to feed real MLX-inferred predictions into AblationRunner.
- S16+ : integrate real mega-v2 dataset (replace synthetic
  fallback in `harness/benchmarks/mega_v2/adapter.py`).
- S16+ : re-run ablation with real predictors, verify gate
  decision against M1.b/M2.b/M4.a thresholds.
- S16+ : if gate passes 3 consecutive runs on real data,
  flip G4 from GO-CONDITIONAL to GO-FULL.
