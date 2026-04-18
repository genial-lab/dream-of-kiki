# scripts — milestone drivers + pilots

One-shot executable scripts that drive **G-gates** (G1..G6) or
produce artifacts consumed by the papers / the OSF pre-registration.
Scripts here are not library code — they end with a JSON dump into
`docs/milestones/` or `docs/proofs/`.

## Conventions

- A script corresponds to exactly one milestone / gate
  (e.g. `pilot_g2.py` → G2 pilot). Name after the gate.
- Header docstring must state : gate ID, what it validates, whether
  it is **pipeline-validation** or **empirical claim**, expected
  output path.
- Always seed `mlx.random.seed(...)` (and any RNG used) and iterate
  over a fixed `seeds = [...]` list. Record per-seed results.
- Write a deterministic JSON dump ; the dump path is part of the
  paper's artifact list.
- Register every real run via `harness.storage.run_registry` before
  writing results. Scripts that skip registration cannot feed a
  paper.

## Anti-patterns specific to scripts

- **Don't** hardcode absolute paths ; resolve from
  `Path(__file__).resolve().parents[1]` as `pilot_g2.py` does.
- **Don't** silently fall back to synthetic data — if real data is
  unavailable the script must exit with a clear "synthetic mode"
  banner (see `pilot_g2.py` docstring for the template).
- **Don't** emit human-only stdout reports without the JSON dump ;
  CI and the run registry both consume the JSON.
- **Don't** edit a pilot after its gate has been declared passed.
  Freeze it and create `pilot_g2_v2.py` if the methodology changes
  — the original must still reproduce its dump byte-for-byte.
- **Don't** add `sys.path.insert(...)` beyond the REPO_ROOT pattern
  already in use ; the harness is installable via `uv`.
