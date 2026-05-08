# Experiment Efficiency Improvements

Tracking changes made to shorten the experiment turnaround. Started
2026-05-07 in branch `visak/anomaly_reward`.

## Motivation

Two observations after the 050–059 round:
1. The 350k-step training budget appears to be roughly 2× more than
   needed — reward curves plateau well before the end.
2. All training runs to date have been on CPU. The user has an Nvidia
   A5000 (24 GB) but a previous attempt at GPU training caused a
   system crash. The current torch + driver + CUDA stack is fresh and
   coherent, and a synthetic benchmark of the actual policy network
   shows a ~37× speedup on the PPO update step. Worth re-attempting.

## Findings

### 1. 350k steps is overkill — plateau lands at ~135–150k

Sampled `rollout/ep_rew_mean` from tensorboard for the three biggest
recent runs (050, 054, 059):

| Step | 050 | 054 | 059 |
|---:|---:|---:|---:|
| 50k | 4.4 | 5.2 | 4.7 |
| 100k | 9.8 | 8.9 | 9.7 |
| **135k** | **10.7** | **10.3** | **10.8** |
| 200k | 10.1 | 7.8 | 9.8 |
| 250k | 11.6 | 10.2 | 10.3 |
| 350k | 11.3 | 9.9 | 9.8 |

All three flatten by ~135–150k; later steps wander but don't trend up.
**Recommended new default: `total_timesteps: 200000`** (~40% buffer
above the knee).

This change alone is a ~1.75× wall-clock reduction on the same hardware.

### 2. GPU is ~37× faster on the policy update step (and the stack works)

Synthetic benchmark of the 054 architecture (`ViTCVAEExtractor`,
0.29M params: ViT 3-layer + per-slot GRU + cross-attention with risk
bias) — pure forward+backward, optimizer step:

| | CPU | A5000 | speedup |
|---|---:|---:|---:|
| 400 fwd+bwd @ batch 64 | 94.8 s | 2.6 s | **36.9×** |
| 100 fwd+bwd @ batch 256 | 77.5 s | 2.0 s | **39.0×** |

Real wall-clock improvement on a small smoke (1024 steps, n_envs=1,
adversarial trainer with 050 config) ran 2026-05-07:

| | CPU | GPU | speedup |
|---|---:|---:|---:|
| End-to-end smoke | 276 s | 231 s | 1.20× |

The smoke understates the real gain because n_envs=1 means env
stepping (CPU-bound, sequential) dominates. For a full training run
with n_envs=20 SubprocVecEnv, env stepping is ~20× cheaper per
timestep, so the update step becomes a much bigger fraction of total
time and the GPU speedup is felt more directly. Realistic estimate
for full training end-to-end: **~1.5–2× from GPU alone**.

Combined with the 200k vs 350k timestep cut, expected full-training
turnaround: ~**2.5–3.5×** faster.

### 3. Stack compatibility — no issues

- Torch 2.11.0+cu130 (built against CUDA 13.0)
- Driver 580.105.08 (CUDA 13.0 runtime)
- A5000 (compute 8.6, 24 GB)
- stable-baselines3 2.8.0 (requires torch ≥2.3 — satisfied)
- gymnasium 1.2.3, highway-env 1.10.2 (no CUDA constraints)

System `nvcc` is 12.8 — that mismatch is **harmless**. Torch ships its
own CUDA runtime libraries in the wheel and uses those. The system
nvcc is only consulted if you build CUDA extensions yourself.

The 0.29M-param model uses ~50 MB of VRAM, ~0.2% of the A5000's 24 GB.
Stress-tested with 200 large matmuls and 400 full forward+backward
passes — no driver crashes, no warnings, peak VRAM negligible.

The previous GPU crash was almost certainly a stale driver/torch
combo from before the system upgrade.

## Changes applied

### Code (committed)

1. `src/driving/train.py:106` — `device="cpu"` → `device=cfg.get("device", "auto")`
2. `src/driving/train_adversarial.py:227` — same pattern
3. Both `run()` functions accept `device_override: str | None = None`
4. Both CLIs accept `--device {cpu|cuda|auto}` to override at runtime

`device="auto"` resolves to cuda when available, cpu otherwise — so
the change is safe on machines without a GPU.

### Recommended for future configs (NOT yet applied to existing
configs, since those experiments are done)

- `total_timesteps: 350000` → `200000`
- Optional: `n_envs: 20` → `32` (more parallel envs amortize the
  per-launch GPU overhead during rollouts; only worth it if A5000
  utilization is low during training)

Don't backfill these into existing 050–059 configs — those configs
record the actual hyperparameters used and shouldn't be rewritten.

## Verification

Smoke test invocation pattern (use this to sanity-check the GPU path
before committing to a long run):

```bash
source .predictive_driving/bin/activate
WANDB_MODE=offline python -m driving.train_adversarial \
    --config experiments/050_h10_v3_highway/config.yaml \
    --run_name smoke_gpu_check --smoke --device cuda
```

Expected: completes in ~3–5 min, no CUDA errors, `time/fps` line in
the SB3 progress output should be similar to or higher than the CPU
baseline.

## Concurrent state when these changes were made

The `fair_eval_ood_cutin.py` evaluation was running in the background
on CPU (PID 21927) when these changes were applied. The eval doesn't
import `train.py` / `train_adversarial.py` and was unaffected. Eval
state at this point: 5 of 12 (model, env) combos done — see
`experiments/ood_cutin_eval/results/eval.log` and W&B run
`6fi0y1jy` for live status.

## Resume-from-here checklist (in case of system crash)

If the box reboots and we need to pick up:

1. **Eval re-launch**: re-run `python scripts/fair_eval_ood_cutin.py`
   (no automatic resume — it would restart from combo 1). Per-combo
   JSON files in `experiments/ood_cutin_eval/results/results_*.json`
   are durable; if you want to skip already-completed combos, add a
   skip-if-exists guard at the top of the per-combo loop.
2. **Trainer changes are committed**, so a fresh checkout of this
   branch already has the GPU + `--device` support.
3. **Next training experiment** should be configured with
   `total_timesteps: 200000` and run via the existing
   `train_adversarial` entry point. Default device will pick up cuda
   automatically.
4. **GPU sanity check before any long run**: run the smoke command
   above. If it crashes, fall back to `--device cpu` and investigate
   the driver/torch version separately.
