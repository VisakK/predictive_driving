# Driving — Autonomous RL Experiments

## Project overview
This repo runs reinforcement learning experiments on autonomous-driving
environments. It combines two vendored open-source libraries (as git
submodules under `libs/`) with a thin experiment harness in `src/driving/`.

- `libs/HighwayEnv` — Gymnasium environments (highway-v0, merge-v0, roundabout-v0, etc.)
- `libs/stable-baselines3` — RL algorithms (PPO, SAC, DQN, TD3, A2C)
- `src/driving/` — shared env factories, training loop, evaluation utilities
- `experiments/NNN_name/` — one self-contained folder per experiment

Both libraries are installed in editable mode, so edits in `libs/` take
effect immediately without reinstall.

## Environment
- Python 3.10, venv at `.venv/`
- Activate with `source .venv/bin/activate` before any command
- GPU optional; default to CPU unless an experiment specifies otherwise

## How to run an experiment
1. Create `experiments/NNN_short_name/` (use next available number)
2. Write `config.yaml` specifying env, algo, hyperparameters, total_timesteps, seed
3. Write `train.py` that loads the config, calls `driving.train.run(config)`,
   and saves artifacts (model, tensorboard logs, eval metrics) to `results/`
4. Run: `python experiments/NNN_short_name/train.py`
5. Log results to `experiments/NNN_short_name/results/summary.md`

## Conventions
- Always set a random seed and record it in config.yaml
- Evaluate over at least 20 episodes with a fixed eval seed
- Report mean ± std of episode reward, episode length, and collision rate
- Never commit files under `experiments/*/results/` (add to .gitignore)
- One experiment = one hypothesis. Keep them small and comparable.
- When modifying code in `libs/`, commit inside the submodule first, then
  commit the submodule pointer update in the parent repo.

## Research ideas to explore
(Fill these in yourself — examples below)
1. Compare PPO vs SAC on `highway-v0` with identical observation wrappers
2. Measure sample efficiency of DQN on `merge-v0` across 5 seeds
3. Ablate observation type (Kinematics vs OccupancyGrid) with PPO
4. Curriculum: pretrain on `highway-v0`, fine-tune on `roundabout-v0`

## When running autonomously
- Start by reading the target experiment's `config.yaml`
- Before training, run a 1000-step smoke test to catch config errors
- For real runs, cap single-experiment training at the `total_timesteps` in config
- After training, always run evaluation and write `results/summary.md`
- If an experiment fails, write the failure and stack trace to `results/error.log`
  and move on — do not retry blindly
- Do not modify `libs/` unless the experiment explicitly calls for it

## Commands cheat sheet
- Smoke test: `python -m driving.train --config experiments/NNN/config.yaml --smoke`
- Full run:   `python experiments/NNN/train.py`
- Tensorboard: `tensorboard --logdir experiments/NNN/results/tb`
- Run tests:   `pytest src/driving/tests`