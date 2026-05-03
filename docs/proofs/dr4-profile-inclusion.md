# DR-4 Profile Inclusion — Draft

**Axiom statement** (from framework spec §6.2) :

```
DR-4 (Profile chain inclusion)
ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)
∧ channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)
```

with **Lemma DR-4.L** : if P_min is valid (invariants satisfied) on
substrate S, then P_equ does not strictly worsen metrics monotone in
capacity.

---

## Profile definitions (recap from §3.1)

| Profile | Channels in | Channels out | Operations |
|---------|-------------|--------------|------------|
| P_min   | β           | 1            | {replay, downscale} |
| P_equ   | β + δ       | 1 + 3 + 4    | {replay, downscale, restructure, recombine_light} |
| P_max   | α + β + δ   | 1 + 2 + 3 + 4 | {replay, downscale, restructure, recombine_full} |

---

## Proof structure

### 1. Operations chain inclusion

We show `ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)` element-wise.

**P_min ⊆ P_equ** :
- replay ∈ ops(P_min) and replay ∈ ops(P_equ) ✓
- downscale ∈ ops(P_min) and downscale ∈ ops(P_equ) ✓
- {replay, downscale} ⊆ {replay, downscale, restructure, recombine}
  by set inclusion. ∎

**P_equ ⊆ P_max** :
- replay, downscale, restructure ∈ both ✓
- recombine_light ⊆ recombine_full (full recombine subsumes the
  light variant — light is a parameter restriction of full)
- Therefore ops(P_equ) ⊆ ops(P_max). ∎

### 2. Channel chain inclusion

We show `channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)`
element-wise on both directions (in-channels and out-channels).

**Channels in (awake → dream)** :
- {β} ⊆ {β, δ} ⊆ {α, β, δ} ✓ trivially by set inclusion.

**Channels out (dream → awake)** :
- {1} ⊆ {1, 3, 4} ⊆ {1, 2, 3, 4} ✓ trivially by set inclusion.

### 3. Combined

Both conditions hold ; DR-4 follows. ∎

---

## Lemma DR-4.L (monotonicity in capacity)

**Statement** : if P_min satisfies all invariants on substrate S,
then `best(P_equ, S) ≥ best(P_min, S)` for any metric M monotone in
capacity (where "monotone in capacity" means: more channels and ops
available → at least as much expressive power).

**Proof sketch** :

Let `R_min` be a P_min run on S that achieves metric value `M_min`.

Construct `R_equ` on S that mimics `R_min` :
- Use only the operations replay, downscale (subset of ops(P_equ))
- Use only channel β in, channel 1 out (subset of channels(P_equ))
- Ignore the additional channels δ, 3, 4 and operations restructure,
  recombine

By construction, `R_equ` produces the same trajectory as `R_min`
because the additional channels/ops are unused, and therefore
`M_equ(R_equ) = M_min(R_min) = M_min`.

Therefore `best(P_equ, S) ≥ M_equ(R_equ) = M_min = best(P_min, S)`. ∎

**Applies to metrics** : M1.b (avg accuracy), M2.b (RSA alignment) —
metrics that benefit from richer representations or more
consolidation operations.

**Does not apply to metrics** : M3.a (FLOPs ratio), M3.c (energy per
episode) — these are *cost* metrics where more capacity costs more,
not less.

---

## Status of each inclusion (S12.1)

| Inclusion | Status | Notes |
|-----------|--------|-------|
| ops(P_min) ⊆ ops(P_equ) | **Verified** | Test in test_dr4_profile_inclusion.py |
| channels(P_min) ⊆ channels(P_equ) | **Verified** | Same test |
| ops(P_equ) ⊆ ops(P_max) | **Pending** | P_max not yet wired (S13+) |
| channels(P_equ) ⊆ channels(P_max) | **Pending** | P_max not yet wired (S13+) |

The S12.1 axiom test therefore covers the P_min ⊆ P_equ chain
empirically. The P_equ ⊆ P_max chain will be added in S13+ when
P_max is wired. The proof above is structural and applies once
P_max definition is locked.

---

## Circulation

This proof is auxiliary to DR-2 (compositionality) — DR-4 is
structurally simpler and does not require external formal review.
Sub-agent `critic` review at draft time is sufficient. Final
incorporation into Paper 1 §6 framework section.

---

## Empirical-evidence amendment — G4-quater (2026-05-03)

The structural inclusions `ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)`
and `channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)` remain
proven. The Lemma DR-4.L statement (capacity-monotone metrics
non-decrease) remains formally true under its construction
proof above — at this scale, retention differences across
profiles are within ±0.001, i.e. a tie not a regression.

What G4-quater positively establishes is that the DR-4
*prediction* "richer profile retains more on capacity-monotone
metrics" is **empirically vacuous at this scale** for the
RECOMBINE channel : at N = 95 per arm,
`retention(P_max with RECOMBINE = mog) ≈
retention(P_max with RECOMBINE = none)` (Welch p = 0.989,
Hedges' g = 0.002). Under H4-C confirmation
(`docs/osf-prereg-g4-quater-pilot.md` §6), the framework-C
claim "richer ops yield richer consolidation at this scale"
is therefore **partially refuted** for the RECOMBINE channel
at the Split-FMNIST / 3-layer MLP scale.

This amendment does **not** weaken the inclusion proof itself
nor invalidate Lemma DR-4.L. It scopes the empirical
prediction : the channel inclusion is formally true (every
operation P_max can run, P_equ can also run), but the
empirically observed *gain* from the richer channels is
within statistical noise for this benchmark. Future work
pre-registered in `docs/osf-prereg-g4-quater-pilot.md` §6 —
CIFAR-10 / ImageNet / hierarchical E-SNN — could in principle
restore the empirical prediction at higher capacity ; until
then, no STABLE promotion of the framework-C "richer ops
yield richer consolidation" claim can occur.

Provenance :
- Pre-registration : `docs/osf-prereg-g4-quater-pilot.md`
- Aggregate verdict : `docs/milestones/g4-quater-aggregate-2026-05-03.{json,md}`
- Step 3 H4-C milestone : `docs/milestones/g4-quater-step3-2026-05-03.{json,md}`
- Paper 2 §7.1.6 : `docs/papers/paper2/results.md` + `docs/papers/paper2-fr/results.md`

---

## Empirical-evidence amendment — G4-quinto (2026-05-03)

**v0.4 (2026-05-03 G4-quinto addendum)** — extends the
G4-quater amendment from a single-benchmark to a
**two-benchmark × two-substrate** scope.

The structural inclusions `ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)`
and `channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)` remain
proven. Lemma DR-4.L (capacity-monotone metrics non-decrease)
remains formally true under its construction proof above —
within-arm differences across all G4-quinto cells stay within
±0.001 (a tie, not a regression).

What G4-quinto positively establishes is that the DR-4
*prediction* "richer profile retains more on capacity-monotone
metrics" is **empirically vacuous** for the RECOMBINE channel
across **both** benchmarks tested in the escalation ladder :

- Split-FMNIST × 3-layer MLX MLP (G4-quater H4-C, N = 95) :
  Welch p = 0.989, Hedges' g = 0.002, mean P_max(mog) = 0.7007
  vs mean P_max(none) = 0.7006.
- Split-CIFAR-10 × small CNN (G4-quinto H5-C, N = 30) :
  Welch p = 0.9918, Hedges' g = -0.0026,
  mean P_max(mog) = 0.9842 vs mean P_max(none) = 0.9845.
- Step 1 (5-layer MLP-on-CIFAR) and Step 2 (small CNN
  baseline) Jonckheere tests are both monotonic_observed = True
  in mean retention but reject_h0 = False (p = 0.4646 and
  p = 0.4823) ; the predicted DR-4 ordering exists in mean but
  not at statistical significance at this N.

Under the H5-C confirmation
(`docs/osf-prereg-g4-quinto-pilot.md` §6 row 4), the
framework-C claim "richer ops yield richer consolidation at
this scale" is therefore **partially refuted across two
benchmarks**, not just one : the RECOMBINE channel is
empirically empty on both Split-FMNIST 3-layer MLP **and**
Split-CIFAR-10 small CNN. The H5-A and H5-B nulls strengthen
this : even when the substrate widens (256-128-64-32 MLP) or
gains hierarchical conv structure (G4SmallCNN), the predicted
ordering does not statistically recover at N = 30. The
universality flag (`h4c_to_h5c_universality = True` in
`docs/milestones/g4-quinto-aggregate-2026-05-03.json`) is now
the dominant DR-4 evidence — overriding any single-benchmark
escape clause.

This amendment does **not** weaken the inclusion proof itself
nor invalidate Lemma DR-4.L. It scopes the empirical
prediction more strictly than the G4-quater addendum : the
channel inclusion is formally true across all profiles, but
the empirically observed *gain* from the richer channels is
within statistical noise on **both benchmarks tested so far**.
Future work pre-registered in
`docs/osf-prereg-g4-quinto-pilot.md` §6 row 6 — testing
ImageNet, transformer heads, hierarchical E-SNN — could in
principle restore the empirical prediction at higher capacity ;
until then, no STABLE promotion of the framework-C "richer
ops yield richer consolidation" claim can occur, and the
empirical scope of any such promotion would have to explicitly
exclude the {Split-FMNIST, Split-CIFAR-10} × {3-layer MLP,
5-layer MLP, small CNN} cells of the escalation ladder.

Provenance :
- Pre-registration : `docs/osf-prereg-g4-quinto-pilot.md`
- §9.1 amendment (HF parquet fallback) : same file, §9.1.
- Aggregate verdict : `docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}`
- Step 1 H5-A milestone : `docs/milestones/g4-quinto-step1-2026-05-03.{json,md}`
- Step 2 H5-B milestone : `docs/milestones/g4-quinto-step2-2026-05-03.{json,md}`
- Step 3 H5-C milestone : `docs/milestones/g4-quinto-step3-2026-05-03.{json,md}`
- Paper 2 §7.1.7 : `docs/papers/paper2/results.md` + `docs/papers/paper2-fr/results.md`
