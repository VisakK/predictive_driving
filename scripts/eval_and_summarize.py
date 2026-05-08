"""Load each trained model, evaluate offline, and write summary.md per experiment."""
import json
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 — registers envs
import numpy as np
import yaml
from stable_baselines3 import PPO
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("001_ppo_baseline", "ppo_highway_v0_kin_b2a41"),
    ("002_ppo_merge", "ppo_merge_v0_kin_b2a41"),
    ("003_ppo_roundabout", "ppo_roundabout_v0_kin_b2a41"),
    ("004_ppo_intersection", "ppo_intersection_v0_kin_b2a41"),
    ("005_ppo_two_way", "ppo_two_way_v0_kin_b2a41"),
    ("006_ppo_u_turn", "ppo_u_turn_v0_kin_b2a41"),
    ("007_ppo_exit", "ppo_exit_v0_kin_b2a41"),
    ("008_ppo_racetrack", "ppo_racetrack_v0_kin_b2a41"),
    ("009_ppo_highway_occgrid", "ppo_highway_v0_occ_b2a41"),
    ("010_ppo_merge_occgrid", "ppo_merge_v0_occ_b2a41"),
    ("011_ppo_roundabout_occgrid", "ppo_roundabout_v0_occ_b2a41"),
    ("012_ppo_intersection_occgrid", "ppo_intersection_v0_occ_b2a41"),
    ("013_ppo_two_way_occgrid", "ppo_two_way_v0_occ_b2a41"),
    ("014_ppo_u_turn_occgrid", "ppo_u_turn_v0_occ_b2a41"),
    ("015_ppo_exit_occgrid", "ppo_exit_v0_occ_b2a41"),
    ("016_ppo_racetrack_occgrid", "ppo_racetrack_v0_occ_b2a41"),
]


def evaluate(model, env_id: str, n_episodes: int, env_config: dict | None, seed: int) -> dict:
    make_kwargs = {}
    if env_config:
        make_kwargs["config"] = env_config
    env = gym.make(env_id, **make_kwargs)
    rewards, lengths = [], []
    terminated_count = truncated_count = crashed_count = 0
    for ep in tqdm(range(n_episodes), desc=f"{env_id}"):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_r, ep_l = 0.0, 0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_r += float(r)
            ep_l += 1
            done = terminated or truncated
        rewards.append(ep_r)
        lengths.append(ep_l)
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if info.get("crashed", False):
            crashed_count += 1
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "n_episodes": n_episodes,
        "early_termination_rate": terminated_count / n_episodes,
        "truncation_rate": truncated_count / n_episodes,
        "crash_rate": crashed_count / n_episodes,
    }


def write_summary(exp_dir: Path, run_name: str, cfg: dict, metrics: dict):
    obs_type = cfg.get("env_config", {}).get("observation", {}).get("type", "default")
    md = f"""# PPO — {cfg['env_id']} ({obs_type})

- **Run name:** {run_name}
- **Config:** `{exp_dir.relative_to(REPO)}/config.yaml`
- **Algo:** PPO ({cfg['policy']}), seed={cfg['seed']}, n_envs={cfg['n_envs']}, total_timesteps={cfg['total_timesteps']}
- **Observation:** {obs_type}
- **simulation_frequency:** {cfg.get('env_config', {}).get('simulation_frequency', 'default')}
- **Eval:** {metrics['n_episodes']} episodes, deterministic actions, eval seed={cfg['seed'] + 10_000}

## Evaluation metrics
| metric | value |
|---|---|
| mean_reward | {metrics['mean_reward']:.4f} |
| std_reward | {metrics['std_reward']:.4f} |
| min_reward | {metrics['min_reward']:.4f} |
| max_reward | {metrics['max_reward']:.4f} |
| mean_episode_length | {metrics['mean_episode_length']:.2f} |
| crash_rate | {metrics['crash_rate']:.2f} |
| early_termination_rate | {metrics['early_termination_rate']:.2f} |
| truncation_rate | {metrics['truncation_rate']:.2f} |

Model saved to `results/model.zip`. Tensorboard logs under `results/tb/`.
"""
    out = exp_dir / "results" / "summary.md"
    out.write_text(md)
    return out


def main():
    all_metrics = {}
    for exp_name, run_name in EXPERIMENTS:
        exp_dir = REPO / "experiments" / exp_name
        with (exp_dir / "config.yaml").open() as f:
            cfg = yaml.safe_load(f)
        model_path = exp_dir / "results" / "model.zip"
        if not model_path.exists():
            print(f"[SKIP] {exp_name}: no model.zip")
            continue
        print(f"[EVAL] {exp_name}  ({cfg['env_id']})")
        model = PPO.load(str(model_path), device="cpu")
        metrics = evaluate(
            model,
            cfg["env_id"],
            n_episodes=cfg.get("eval_episodes", 50),
            env_config=cfg.get("env_config"),
            seed=cfg["seed"] + 10_000,
        )
        write_summary(exp_dir, run_name, cfg, metrics)
        all_metrics[exp_name] = metrics
        print(f"  mean_reward={metrics['mean_reward']:.3f}  crash_rate={metrics['crash_rate']:.2f}")
    (REPO / "experiments" / "ablation_summary.json").write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
