# Phase B 35B Relaunch (2026-04-20)

**Status** : DEFERRED — Studio not actually solo, relaunch held for user decision

## Context

Initial 35B launch (PID 20352 at 02:53 CEST) crashed after 1 cell at
~18 min wall-clock due to concurrent-run resource contention on Studio.
At the time, 7B Phase B (PID 19746 at 02:43 CEST) and a separate 35B
MLX LoRA SFT job for micro-kiki Qwen36 were both running. The crash
manifested as a `leaked semaphore` error, consistent with MLX /
multiprocessing resource pressure under Python 3.14.

The 7B sweep has since completed (2.4 h wall-clock, 90/90 cells, 0
failures) producing `pilot-cycle3-real-7b.json` with GO verdict
(p_max collapsed to p 5.51e-02 vs p_min 1.44e-24 and p_equ 6.18e-27 —
see `scaling-law-analysis-2026-04-20.md`).

The plan was to relaunch 35B **solo** to avoid the original
contention, but pre-flight on 2026-04-20 06:10 CEST found Studio is
**not solo** :

## Pre-flight 2026-04-20 06:10 CEST

- `pgrep pilot_cycle3_real` : clean (no leftover pilot process)
- `pgrep python` : PID 29508 is active
- PID 29508 identity : `mlx_lm lora -c
  output/micro-kiki/lora-qwen36-35b-v3/config-python.yaml`
  (35B MLX LoRA SFT for micro-kiki Qwen36 v3)
- PID 29508 memory : 143 GB RSS, peak 118.7 GB per its own log,
  `mx.set_memory_limit(460 * 1024**3)` (460 GB ceiling)
- PID 29508 progress : iter 500 at 06:10 CEST, started 45 min earlier
  at ~05:25, ~0.22 it/sec
- Studio global memory : 444 GB used / 66 GB free
  (`top -l 1` PhysMem 444G used, 66G unused)
- Another active Python process : PID 32144 at 79 GB RSS
- Load avg 3.83 / 3.49 / 3.42

The MLX LoRA SFT matches the "Studio SFT 35B Opus en cours (mlx_lm,
batch=1 grad_accum=8, ETA ~2h)" state recorded in CLAUDE.md on
2026-04-14. It is an expected, known workload — not a leftover.

## Decision point

Launching the 35B Phase B eval now would :

1. Push combined memory demand to ~460 + (35B fp16 model ~70 GB + MLX
   cache) well beyond the 512 GB physical ceiling
2. Re-create the exact contention pattern that crashed the first
   attempt
3. Risk killing both the Phase B eval AND the micro-kiki SFT
   mid-training

Per the user's explicit constraint ("solo, no concurrent") the
safer path is to defer the relaunch until the MLX LoRA SFT
completes. At ~0.22 it/sec, if the SFT is configured for ~1000-2000
iters (typical Qwen LoRA SFT), completion window is 1-3 h from now.

**Recommended sequence** :
1. Wait for PID 29508 to finish (tail `log-python.txt` for
   `"Iter <final>: Saved adapter weights"` marker)
2. Confirm `pgrep -fl python | grep -i mlx` is clean
3. Confirm free memory > 300 GB
4. Then launch 35B Phase B with the fire-and-forget command below

## Launch command (held, not executed)

```bash
ssh studio 'cd ~/Documents/Projets/dreamOfkiki \
  && mkdir -p ~/dreamOfkiki-runs \
  && export PATH="$HOME/.local/bin:$PATH" \
  && nohup uv run python scripts/pilot_cycle3_real.py \
     --scale=qwen3p5-35b-fp16 \
     > ~/dreamOfkiki-runs/phase-b-35b-relaunch-$(date +%Y%m%d-%H%M).log \
     2>&1 & echo "PID=$!"; sleep 3; \
     pgrep -fl pilot_cycle3_real | head -3'
```

- **PID** : not yet launched
- **Log** : `~/dreamOfkiki-runs/phase-b-35b-relaunch-YYYYMMDD-HHMM.log`
- **Launch time** : deferred
- **Expected ETA** : ~27 h wall-clock once launched (18 min/cell x
  90 cells extrapolated from prior crash log per-cell timing)

## Monitoring (once launched)

```bash
ssh studio 'pgrep -fl pilot_cycle3_real'
ssh studio 'tail -5 ~/dreamOfkiki-runs/phase-b-35b-relaunch-*.log'
```

## Post-completion

1. scp JSON result
2. Update `docs/milestones/scaling-law-analysis-2026-04-20.md` with
   35B row of the comparison table
3. Decide H5-III power-law fit viability (3 points)
4. If p_max at 35B also collapsed (p >> 0.0125, delta trending
   to 0 or below) then confirmed scale-dependent monotonic collapse
   direction for H7 Paper 2
5. If p_max at 35B recovered (p < 0.0125, delta > 0.05) then
   U-shaped complex — refine H7 hypothesis

## Cross-references

- Phase B 1.5B : `pilot-cycle3-real-1p5b.json` (commit 22c58c9)
- Phase B 7B : `pilot-cycle3-real-7b.json` (commit c42179c)
- Scaling-law analysis : `scaling-law-analysis-2026-04-20.md`
