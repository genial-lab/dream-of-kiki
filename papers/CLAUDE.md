# papers — drafts for Paper 1 (framework) and Paper 2 (ablation)

Two venue targets, two subdirectories :

- `paper1-framework/` — Nature HB / PLoS Comp Bio. Formal framework
  C, axioms DR-0..DR-4, Conformance Criterion. Sources under
  `submitted/`.
- `paper2-ablation/` — NeurIPS / ICML / TMLR. Empirical ablation on
  kiki-oniric (`P_min` / `P_equ` / `P_max`). Sources under `draft/`.

## Writing discipline

- Every empirical number must resolve to a **registered `run_id`**
  from `harness/storage/run_registry.py`. No "we observed ~0.6" —
  cite the run and the dump file under `docs/milestones/`.
- Every axiom / invariant mentioned must use the canonical ID
  (DR-0..DR-4, I1..In, S1..Sn, K1..Kn) and match `docs/glossary.md`.
  Do not introduce local synonyms in prose.
- Every proof sketched in the paper must have a full version under
  `docs/proofs/` (or be explicitly marked as deferred).
- Synthetic-benchmark results are allowed only in methodology /
  pipeline-validation sections, never in headline claims. Mark
  clearly as "synthetic placeholder (G2 pilot)".

## Anti-patterns specific to papers

- **Don't** paste numbers from a scratch notebook — they must come
  from a registry dump. If the dump doesn't exist, the claim doesn't
  exist.
- **Don't** edit `submitted/` after a venue submission snapshot ;
  fork to `submitted-revN/` instead so the submitted state is
  reproducible.
- **Don't** mix Paper 1 and Paper 2 scope : the formal framework is
  substrate-agnostic and must not reference `kiki_oniric` internals
  by name. The ablation paper conversely must cite DR / I / S IDs
  from Paper 1 by version.
- **Don't** use AI attribution in the byline or acknowledgments ;
  the project policy is "dreamOfkiki project contributors".
- **Don't** break the DualVer contract in prose : when citing
  framework C always give the full `C-vX.Y.Z+{STABLE,UNSTABLE}`
  tag, as shipped in `CHANGELOG.md`.
- **Don't** regenerate figures without recording the seed and
  harness version ; figure scripts belong under
  `scripts/` and emit to `docs/milestones/`.
