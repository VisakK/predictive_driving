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
- Use CPU only for these experiments

## Goal of the experiment and how to run one 
The main purpose of this autonomous agent is that to generate a policy for all environments in highway env, such as `highway-v0`, `merge-v0`, `roundabout-v0`, etc..
Each of these policies should be saved and the corresponding wandb run with all the training and eval data. Do the following for each highway environment. Before any experiments read all the `.py` files in src/driving/ and `.yaml` experiments/ folders. 
1. Create `experiments/<environment_name_PP)>/` (use next available number)
2. Write `config.yaml` specifying env, algo, hyperparameters, total_timesteps, seed
3. Write a new `train_<env>.py` that loads the config, calls `driving.train.run(config)`,
   and saves artifacts (model, tensorboard logs, eval metrics) to `results/`
4. Run a smoke test for each environment before you run a full training run.
5. Log results to `experiments/<environment_name_PP>/results/summary.md`
6. Make sure the run name is unique 

## Conventions
- Always set a random seed and record it in config.yaml
- Evaluate over at least 50 episodes with a fixed eval seed
- Use existing train file : src/driving/train.py for reference 
- Use existing config file: experiments/001_ppo_baseline/config.yaml
- Never commit files under `experiments/*/results/` (add to .gitignore)
- One experiment = one hypothesis. Keep them small and comparable.
- When modifying code in `libs/`, commit inside the submodule first, then
  commit the submodule pointer update in the parent repo.

## Working with submodules in libs/
- Before editing any file under `libs/`, cd into that submodule and
  checkout a branch (never commit in detached HEAD).
- Commit submodule changes inside the submodule first, then commit the
  pointer update in the parent repo as a separate commit.
- Parent-repo commits that bump a submodule pointer must explain WHY,
  referencing the experiment or issue that motivated the change.
- Do not modify files under `libs/` unless the current task explicitly
  requires it — prefer wrappers in `src/driving/` when possible.

## Research ideas to explore
(Fill these in yourself — examples below)
1. Ablate observation type (Kinematics vs OccupancyGrid) for all environments with PPO.
2. Example of how to set observation space is present in src/driving/observation_space_example.py


## When running autonomously
- Start by reading the target experiment's `config.yaml`
- Before training, run a 1000-step smoke test to catch config errors
- For real runs, cap single-experiment training at the `total_timesteps` in config
- After training, always run evaluation and write `results/summary.md`
- If an experiment fails, write the failure and stack trace to `results/error.log`
  and move on — do not retry blindly
- Do not modify `libs/` unless the experiment explicitly calls for it

## Commands cheat sheet
- Smoke test: `python -m driving.train --config experiments/NNN/config.yaml --run_name smoke_test_<unique_id> --smoke`
- make up a unique id here
- Full run:   `python -m driving.train --config experiments/NNN/config.yaml --run_name ppo_<env_name>_<unique_id>`