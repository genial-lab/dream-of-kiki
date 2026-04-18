# kiki_oniric — Track A substrate

Fork of `kiki-flow-core` that implements framework C on an MLX /
Apple-Silicon substrate. Package import name is `kiki_oniric`
(underscore) — never `kiki-oniric` in Python.

## Layout

- `core/primitives.py` — typed `Protocol` signatures for the 8
  primitives (α/β/γ/δ awake→dream, 4 channels dream→awake).
  These are the contract that satisfies **DR-3 Conformance
  Criterion condition (1)**. Do not loosen the types.
- `dream/runtime.py`, `episode.py`, `swap.py` — dream-state runtime,
  episode lifecycle, awake↔dream swap.
- `dream/operations/` — `replay`, `downscale`, `restructure` (and
  MLX-backed variants). Each operation must preserve the invariants
  its docstring cites.
- `dream/guards/` — `finite.py` (S2), `topology.py` (S3). Guards
  raise with the invariant ID in the message.
- `profiles/` — `p_min`, `p_equ`, (future `p_max`). A profile picks
  the primitives/channels it activates ; see framework-C §3.

## Anti-patterns specific to this package

- **Don't** add an operation without an accompanying guard reference
  and a conformance test under `tests/conformance/`.
- **Don't** rename primitive methods — they are part of the DR-3
  conformance contract exposed via `@runtime_checkable` Protocols.
- **Don't** import `numpy` into an MLX-only hot path ; the MLX and
  numpy variants live in separate files by design.
- **Don't** edit files here outside the jalonné rebase windows
  (S1 / S8 / S18, see `docs/fork-decision.md`) except in planned
  Story tasks.
- **Don't** bake seeds into a profile — seeds belong in the harness
  run registry.

## When changing a primitive signature

Bump framework C **formal axis** (MAJOR if breaking), update
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §2.1,
update `docs/interfaces/primitives.md`, and regenerate conformance
tests.
