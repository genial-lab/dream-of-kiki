# Results (Paper 1, draft S18.2)

**Target length** : ~1.5-2 pages markdown (≈ 1500-2000 words)

⚠️ **Caveat (synthetic placeholder, G2/G4 pilot).** Every
quantitative claim in this section comes from mock predictors at
scripted accuracy levels (50%/70%/85%) registered under run_id
`syn_s15_3_g4_synthetic_pipeline_v1` (dump
`docs/milestones/ablation-results.json`). Numbers validate the
*pipeline*, not P_equ efficacy on real linguistic data ; the
section is preserved here so reviewers can audit the reporting
template, but no headline empirical claim should be drawn from
it. Real mega-v2 + MLX-inferred predictors land cycle-1 closeout
(S20+) and will replace these placeholders.

---

## 7.1 P_min viability (G2)

We first verified that the P_min profile (replay + downscale only)
operates within architectural constraints (DR-0 accountability,
S1+S2 swap guards). On a 50-item synthetic retained benchmark, the
swap protocol committed in 100% of attempted cycles when the
predictor matched expected outputs and aborted with `S1 guard
failed` in 100% of cycles when accuracy degraded — establishing
the swap gating contract operationally.

**Table 7.1 — P_min pilot (G2, synthetic placeholder, G2 pilot)**

run_id : `syn_g2_pmin_pipeline_v1`
dump : `docs/milestones/g2-pmin-report.md`

| Seed | Baseline acc | P_min acc | Δ |
|------|--------------|-----------|---|
| 42   | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 123  | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 7    | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |

Gate verdict (synthetic pipeline validation only ; criterion
Δ ≥ −0.02) : **PASS**. See `docs/milestones/g2-pmin-report.md`
for raw results.

---

## 7.2 P_equ functional ablation (G4)

P_equ adds the `restructure` operation (Friston FEP source) and
the `recombine` operation (Hobson REM source) alongside `replay`
+ `downscale`, with channels β+δ → 1+3+4 wired. We ran the
ablation runner across 3 profiles (baseline, P_min, P_equ) × 3
seeds on a synthetic mega-v2-style 500-item benchmark stratified
across 25 domains.

**Table 7.2 — G4 ablation accuracy (synthetic placeholder, G4
pilot)**

run_id : `syn_s15_3_g4_synthetic_pipeline_v1`
dump : `docs/milestones/ablation-results.json`

| Profile  | Mean acc | Std | Range |
|----------|----------|-----|-------|
| baseline | [SYNTH 0.500] | [SYNTH 0.000] | 0.500-0.500 |
| P_min    | [SYNTH 0.700] | [SYNTH 0.000] | 0.700-0.700 |
| P_equ    | [SYNTH 0.850] | [SYNTH 0.000] | 0.850-0.850 |

(Replace with real ablation values post-S20+ ; new run_id will be
registered when real predictors are wired.)

---

## 7.3 H1 — Forgetting reduction (synthetic placeholder)

Welch's t-test (one-sided) on forgetting (1 − accuracy) of P_equ
versus baseline (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) :

- **Statistic** : t = [SYNTH −47.43]
- **p-value** : p < 0.001 (synthetic, will be tightened with real data)
- **Bonferroni α** : 0.0125
- **Synthetic-pipeline outcome** : H0 rejected on the mock
  predictors. **No empirical hypothesis decision** is announced
  here ; the genuine H1 verdict is deferred to S20+ when real
  mega-v2 predictors are wired and a fresh run_id is registered.

---

## 7.4 H3 — Monotonic representational alignment
       (synthetic placeholder)

Jonckheere-Terpstra trend test on accuracy across the ordered
profile chain (P_min < P_equ) (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) :

- **J-statistic** : [SYNTH 9.0]
- **p-value** : [SYNTH 0.0248]
- **Bonferroni α** : 0.0125
- **Synthetic-pipeline outcome** : fails to reject H0 at the
  Bonferroni-corrected threshold (would reject at conventional
  α = 0.05). **No empirical hypothesis decision** is announced
  here ; cycle 2 with P_max integrated should provide the third
  group needed to strengthen the trend signal on real data.

---

## 7.5 H4 — Energy budget compliance (synthetic placeholder)

One-sample t-test on the energy ratio
energy(dream) / energy(awake) versus the threshold 2.0 (master
spec §7.2 viability criterion) (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) :

- **Sample mean** : [SYNTH 1.6]
- **t-statistic** : [SYNTH −5.66]
- **p-value** : [SYNTH 0.0101]
- **Bonferroni α** : 0.0125
- **Synthetic-pipeline outcome** : H0 rejected on the mock
  energy-ratio sample ; the **empirical** H4 verdict is deferred
  to S20+ when real wall-clock energy traces are recorded under
  a freshly registered run_id.

---

## 7.6 H2 — P_max equivalence (cycle 2 deferred)

Per cycle-1 SCOPE-DOWN decision (master spec §7.3), P_max profile
remains skeleton-only. We executed a smoke-test TOST equivalence
of P_equ against itself (with a tiny deterministic perturbation)
to validate the statistical pipeline ; the test correctly accepted
equivalence (p ≈ 5e-08). Real H2 P_max equivalence test deferred
to cycle 2 alongside α-stream + ATTENTION_PRIOR canal-4 wiring.

---

## 7.7 Gate summary

Of the 4 pre-registered hypotheses :
- **H1 forgetting** : significant (PASS)
- **H2 equivalence** : smoke-only (cycle 2)
- **H3 monotonic** : borderline (PASS at α=0.05, fail at
  Bonferroni 0.0125)
- **H4 energy** : significant (PASS)

**G4 gate result (synthetic pipeline validation only)** :
**PASS** — see CAVEAT below (≥2 hypotheses significant at
Bonferroni-corrected α). See `docs/milestones/ablation-results.md`
for full data + JSON dump.

> **⚠️ CAVEAT — synthetic data only.** The PASS verdict above
> validates the *measurement and statistical pipeline*, not the
> efficacy of P_equ on real linguistic data. All numbers in this
> section derive from mock predictors at scripted accuracy levels
> (50% baseline, 70% P_min, 85% P_equ). Real mega-v2 + MLX
> inference results are pending cycle-1 closeout (S20+) per the
> G2/G4/G5 GO-CONDITIONAL decisions.

---

## Notes for revision

- Replace all [SYNTH ...] values with real ablation numbers post
  S20+ (real mega-v2 + real MLX inference)
- Add error bars / 95% CI tables for each metric across 3 seeds
- Add Figure 1 : accuracy distribution boxplot per profile
- Add Figure 2 : Jonckheere J-statistic with confidence band
- Cross-reference §6 Methodology for protocol details
- Discuss H3 borderline result in §8 Discussion
