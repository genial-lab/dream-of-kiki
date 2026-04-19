# Cycle-3 Plan Adaptation — PLOS CB Pivot (2026-04-20)

**Milestone** : cycle-3 plan adaptation following Paper 1 v0.2
venue retarget (Nature HB → PLOS Computational Biology)
**Trigger commit** : `d6866f3` (paper1 v0.2 W-series revisions +
MIT LICENSE + venue retarget to PLOS CB)
**Status** : **PIVOT-CODIFIED** — atomic plan markers updated,
Phase 2 work scoped-deferred to Paper 2

---

## Why this milestone exists

The cycle-3 atomic plan
(`docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md`)
was authored on 2026-04-19 against a **Nature Human Behaviour**
target requiring a 6-week multi-scale empirical sprint
(C3.1–C3.22, 1080-config Studio ablation, Phase 2 neuromorph +
fMRI tracks).

On 2026-04-20 the cold-review analysis
(`Business OS/paper1-cold-review-2026-04-19.md`) recommended
retargeting Paper 1 to **PLOS Computational Biology**. The
`d6866f3` commit codified the venue switch :
- §1 abstract reframed around **DR-2 generated-semigroup theorem**
  (proved, this paper)
- §5.6 added : cross-substrate conformance walkthrough
  (covered by C3.16 + C3.17 infra already landed)
- §7 rewritten : pipeline-and-framework validation (no synthetic
  p-values reported as empirical claims)
- §8.3 limitations : H1–H4 empirical decisions **moved to Paper 2**

PLOS CB scope-fit means most of cycle-3 Phase 2 (multi-scale
empirical, neuromorph cross-substrate, fMRI BOLD alignment) is
**no longer required for Paper 1 submission**. Reframing those
tasks as Paper 2 deliverables avoids 4–5 weeks of wall-clock
GPU budget and clarifies the publication critical path.

---

## Adaptation matrix

| Task | Original status | New status (PLOS CB pivot 2026-04-20) |
|------|-----------------|----------------------------------------|
| C3.1–C3.7, C3.10 | Phase 1 sem 1–3 | ✅ DONE |
| C3.8 (full ablation 1.5B + 7B + 35B) | Phase 1 sem 2–3 | **PARTIAL** — 1.5B done as Phase B (commit `22c58c9`), 7B + 35B **DEFERRED to Paper 2** |
| C3.9 (Gate D decision) | Phase 1 sem 3 end | **DEFERRED to Paper 2** (gate is on multi-scale, here only 1.5B available) |
| C3.11 (Norse wrapper) | Phase 2 sem 4 | ✅ DONE |
| C3.12 (SNN ops) | Phase 2 sem 4 | ✅ DONE |
| C3.13 (Norse vs MLX pilot) | Phase 2 sem 4–5 | **DEFERRED to Paper 2** — orphan partial driver `scripts/pilot_phase2b_neuromorph.py` (624 LOC) preserved with deferred-note header for Paper 2 reactivation |
| C3.14 (G10a milestone) | Phase 2 sem 4–5 | **DEFERRED to Paper 2** |
| C3.15 (Studyforrest BOLD loader) | Phase 2 sem 4 | ✅ DONE |
| C3.16 (HMM dream-state alignment) | Phase 2 sem 5 | ✅ DONE |
| C3.17 (CCA Studyforrest alignment) | Phase 2 sem 5 | ✅ DONE |
| C3.18 (G10c fMRI report) | Phase 2 sem 5 | **DEFERRED to Paper 2** |
| C3.19–C3.22 (Paper 1 v2 narrative + DualVer) | Phase 2 sem 6 | **REPLACED** by Paper 1 v0.2 PLOS CB workflow — narrative authored in `docs/papers/paper1/` (single tree, not paper1-v2/), DualVer graduation deferred until Paper 2 closeout |

**Net** : 11 of 22 tasks DONE in-tree, 6 DEFERRED to Paper 2,
4 REPLACED by Paper 1 v0.2 workflow, 1 PARTIAL.

---

## Empirical highlight kept in-scope

The Phase B 1.5B FP16 sanity pilot (commit `22c58c9`) produced
the only multi-scale data point retained for Paper 1 v0.2. It
serves as the **pipeline-validation artifact** in §7 (per the
PLOS CB framing : we validate the framework's machinery, not
hypothesis significance).

Phase B 1.5B FP16 verdict :
- `p_min` : t very large, p = 9.99e-21
- `p_equ` : p = 3.97e-19
- `p_max` : p = 1.66e-17
- Verdict : **GO 3/3**
- Pattern : within-domain consolidation (held-out mega-v2)
  **without** catastrophic forgetting (MMLU / HellaSwag delta = 0)
- Wall-clock : 46.75 min on Studio M3 Ultra

This is **not** a hypothesis test against H1–H4 (those moved to
Paper 2). It is a pipeline conformance check : the dream loop
can mutate a 1.5B Qwen FP16 model under DR-2 canonical order +
S1/S2/S3 guards + R1 determinism contract without breaking the
held-out evaluation surface.

---

## Immediate next actions (Paper 1 v0.2 critical path)

1. **arXiv submission prep** — `docs/papers/paper1/build/full-draft.pdf`
   v0.2 (22 pages, 296 KB) re-rendered 2026-04-20 ; verify
   `references.bib` resolves all citeproc entries ; finalise
   PLOS CB cover letter.
2. **DR-2 proof external review trigger** — circulate
   `docs/proofs/dr2-compositionality.md` to T-Col formal reviewers
   (Q_CR.1.b candidate list).
3. **§5.6 cross-substrate walkthrough audit** — verify the §5.6
   narrative cites the correct C3.16 + C3.17 commits and run-registry
   IDs (no synthetic numbers as empirical claims).
4. **STATUS.md + DualVer reconciliation** — keep
   `C-v0.7.0+PARTIAL` (Phase 2 cells deferred under §12.3
   transition rule) ; promotion to `+STABLE` deferred until
   Paper 2 closeout.

---

## Paper 2 deferred scope (next research arc)

The DEFERRED tasks above form a coherent Paper 2 backlog focused
on **multi-scale empirical claims** :

- **Multi-scale ablation** (C3.8 7B + 35B) — Studio M3 Ultra
  sprint, ~10 days wall-clock, OSF pre-reg amendment for H5
  trivariant on 7B + 35B cells.
- **Gate D decision** (C3.9) — H1–H4 sig 3/3 scales + H5 family
  Bonferroni + H6 profile ordering on full 1080-config matrix.
- **Phase 2b cross-substrate** (C3.13 + C3.14) — Norse vs MLX
  pilot reusing the preserved orphan driver
  `scripts/pilot_phase2b_neuromorph.py` ; G10a milestone report.
- **Phase 2c fMRI** (C3.18) — Phase-2c real-data cell + G10c
  milestone report on Studyforrest BOLD alignment.
- **Paper 2 narrative** (C3.19–C3.22 spirit) — multi-scale
  ablation paper authored after Gate D verdict, target venue TBD
  (NeurIPS / TMLR / Cognitive Science depending on Gate D outcome).

Re-spec window for Paper 2 will be opened **after** Paper 1
v0.2 submission lands an arXiv ID. The cycle-3 atomic plan is
**not** rewritten in place — DEFERRED markers are appended as
inline blocks to preserve the original sequencing for Paper 2
reactivation.

---

## Provenance

- Trigger commit : `d6866f3` (2026-04-20 W-series revisions)
- Cycle-3 atomic plan : `docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md`
- Cycle-3 design spec : `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
- Paper 1 v0.2 PDF : `docs/papers/paper1/build/full-draft.pdf`
  (22 pages, 296 KB, rendered 2026-04-20)
- Phase B sanity pilot : commit `22c58c9` ; dump
  `docs/milestones/pilot-cycle3-sanity-1p5b.md`
- Orphan driver preserved : `scripts/pilot_phase2b_neuromorph.py`
  (624 LOC, deferred-note header) — Paper 2 reactivation entry point
- Cold-review source : `Business OS/paper1-cold-review-2026-04-19.md`
  (external, in workspace not repo)

---

## DualVer impact

No bump. The PLOS CB pivot is a **scope reframing** (§12.3
transition rule), not a formal-axis change : axioms DR-0..DR-4
unchanged, invariants I/S/K unchanged, Conformance Criterion
unchanged. EC remains `+PARTIAL` because Phase 2 multi-scale
cells are explicitly deferred (now to Paper 2 instead of cycle-3
sem 6). Promotion to `+STABLE` happens at Paper 2 closeout, not
at Paper 1 v0.2 submission.

`C-v0.7.0+PARTIAL` is the canonical version through both
Paper 1 v0.2 PLOS CB submission and Paper 2 multi-scale work.
