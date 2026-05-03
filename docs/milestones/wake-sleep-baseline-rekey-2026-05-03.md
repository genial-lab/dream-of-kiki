# Wake-Sleep CL baseline — re-key (supersede entry)

**Supersedes :** `wake-sleep-baseline-2026-05-03.{json,md}`
(frozen append-only ; numerical values, run_ids and seed grid in
that dump are unchanged ; this entry replaces them with verified
Alfarano 2024 values on a benchmark the paper actually scores).
**Verify trail :** `wake-sleep-baseline-verify-2026-05-03.md`
(2026-05-03 PDF parse identifying the two mismatches resolved
here).

**Source :** Alfarano et al. 2024, IEEE TNNLS, arXiv 2401.08623,
Tables 2-3, row "ER-ACE+WSCL (Ours)", Split CIFAR-10
(5 binary tasks, class-incremental), buffer = 500.
**Bibkey :** `alfarano2024wakesleep`.
**Task split :** `cifar10_5tasks_buffer500`.
**Commit :** `1090b92decf027e8b8034beb63710fe7b9fec8b8`.

**(variant c — published reference values, re-keyed 2026-05-03
to CIFAR-10 buffer-500 to match the Alfarano 2024 Tables 2-3
ER-ACE+WSCL row.)**

## Resolution path retained

Per `wake-sleep-baseline-verify-2026-05-03.md` §"Action retained"
the maintainer chose **Option 1 — re-key on an Alfarano benchmark**.
The previous `split_fmnist_5tasks` placeholder pair
`(forgetting_rate=0.082, avg_accuracy=0.847)` did not match any
cell of Tables 2-3 at any buffer size for any of the three
benchmarks Alfarano scores (CIFAR-10, Tiny-ImageNet1/2,
FG-ImageNet). The new pair below is taken directly from
Tables 2-3 ER-ACE+WSCL row at CIFAR-10 buffer-500 and rescaled
from percentages to unit-interval decimals
(`74.18 % -> 0.7418`, `10.69 % -> 0.1069`).

The three previous run_ids
(`60a86e83…`, `4b6b475e…`, `fcd2873d…`) are no longer cited as
empirical anchors ; they remain in the frozen parent dump for
historical traceability under the docs/CLAUDE.md append-only
discipline.

## Rows (re-keyed)

| seed | run_id | forgetting_rate | avg_accuracy |
|------|--------|-----------------|--------------|
| 42  | `38ad694fe99c2dfbb3f8ca4c312852b7` | 0.1069 | 0.7418 |
| 123 | `2cdef3880915543654c81205fe4edf9a` | 0.1069 | 0.7418 |
| 7   | `16f511205877c190074790f094309316` | 0.1069 | 0.7418 |

## Variant-c semantics

The seed-round-trip identity (same numbers across seeds) is
**expected** under variant c — the values are published reference
constants, not a re-run. Variants a/b would yield seed-dependent
rows under a real Avalanche-driven training loop. The seed
argument round-trips into the registry for R1 provenance only.

## Provenance

- Driver : `scripts/baseline_wake_sleep_cl.py` (DEFAULT_TASK_SPLIT
  re-pointed to `cifar10_5tasks_buffer500` ; DEFAULT_OUT moved to
  the `-rekey` filename).
- Adapter constants :
  `kiki_oniric/substrates/wake_sleep_cl_baseline.py`
  `_REFERENCE_METRICS_BY_TASKSPLIT`.
- Paper 2 §7.7 EN + §7.7 FR : caption updated, table values
  re-keyed.
- Eval matrix : `docs/interfaces/eval-matrix.yaml` `baselines.
  wake_sleep_cl` block unchanged (bibkey + variant + scores_on
  semantics did not change — only the underlying benchmark
  identifier did, which is internal to the adapter).

## DualVer impact

**No DualVer bump.** The re-key is a placeholder-resolution
correction, not an axiom statement, primitive signature, channel
set, or invariant ID change (FC unchanged at v0.12.0). It is
also not a gate result (EC stays PARTIAL). Per framework-C §12
this is below the FC-PATCH threshold — recorded as a CHANGELOG
documentation entry only.
