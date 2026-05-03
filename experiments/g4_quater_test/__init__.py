"""G4-quater test package — sequential 3-step empirical resolution
of the G4-ter H_DR4 inversion.

Locked recipe (per
``docs/superpowers/plans/2026-05-03-g4-quater-restructure-recombine-test.md``
Task 0.5) :

- Step 1 — H4-A test : 5-layer deeper substrate
  (``deeper_classifier.G4HierarchicalDeeperClassifier``).
- Step 2 — H4-B test : RESTRUCTURE factor sweep on the existing
  3-layer head (``experiments.g4_ter_hp_sweep``).
- Step 3 — H4-C test : RECOMBINE strategy ∈ {mog, ae, none} with
  ``none`` as the placebo control isolating REPLAY+DOWNSCALE
  contribution.

Pre-registration : ``docs/osf-prereg-g4-quater-pilot.md``.
"""
