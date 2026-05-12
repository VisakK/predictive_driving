"""Standalone eval that records *every* episode as video and uploads to W&B.

Use after training when you want richer post-hoc media than the built-in
training-time eval (`evaluate_and_log` defaults to ~5 videos). Reads the
same config the model was trained with so the env/observation/action
specification is identical, then runs `--n_episodes` (default 50) episodes
deterministically with full frame recording.

Each episode is uploaded as a single W&B Video keyed
`eval/videos/ep_<seed>`; all 50 land in the same `eval/videos` group so
W&B renders them as one media panel rather than 50 separate panels.
Also logs aggregate metrics under `eval/*` and a per-episode `wandb.Table`.

Usage:
  python scripts/eval_model_with_videos.py \
      --config experiments/060_baseline_continuous_v3/config.yaml \
      --model experiments/060_baseline_continuous_v3/results/model.zip \
      --run_name baseline_060_full_video_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym
import tqdm
import wandb

from stable_baselines3 import PPO
from stable_baselines3 import SAC, DQN  # noqa: F401 — for future configs

import yaml

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401
import driving.adversarial_v3  # noqa: F401
import driving.adversarial_v3_ts  # noqa: F401 — registers v3ts envs
import driving.adversarial_v3i  # noqa: F401 — registers v3i (interaction) envs
import driving.baseline_continuous  # noqa: F401

ALGOS = {"PPO": PPO, "SAC": SAC, "DQN": DQN}

try:
    from driving.adversarial_ppo import AdversarialPPO
    ALGOS["AdversarialPPO"] = AdversarialPPO
except Exception:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--n_video_episodes", type=int, default=None,
                   help="Record video only for the first N episodes (default: all).")
    p.add_argument("--seed_offset", type=int, default=10_000,
                   help="Eval seeds = cfg.seed + seed_offset + ep_index")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()
    n_video = args.n_video_episodes if args.n_video_episodes is not None else args.n_episodes

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algo = cfg.get("algo", "PPO")
    AlgoCls = ALGOS[algo]
    model = AlgoCls.load(args.model, device="cpu")

    env_config = cfg.get("env_config")
    env = gym.make(cfg["env_id"], render_mode="rgb_array", config=env_config)

    run_name = args.run_name or (
        f"{Path(args.config).parent.name}_full_video_eval_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    wb_run = wandb.init(
        project="predictive_driving",
        name=run_name,
        config={
            **cfg,
            "eval/n_episodes": args.n_episodes,
            "eval/source_model": str(args.model),
            "eval/seed_offset": args.seed_offset,
        },
    )

    base_seed = cfg["seed"] + args.seed_offset
    rewards, lengths = [], []
    crashed_count = terminated_count = truncated_count = 0
    per_episode_rows = []
    videos = []  # collect all videos to log as a grouped media panel

    for ep in tqdm.tqdm(range(args.n_episodes), desc="Eval"):
        seed = base_seed + ep
        obs, info = env.reset(seed=seed)
        done = False
        r = 0.0
        l = 0
        record = ep < n_video
        frames = []
        while not done:
            if record:
                frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, rw, terminated, truncated, info = env.step(action)
            r += float(rw)
            l += 1
            done = terminated or truncated
        rewards.append(r)
        lengths.append(l)
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if info.get("crashed", False):
            crashed_count += 1
        per_episode_rows.append((ep, seed, r, l, bool(terminated),
                                 bool(truncated), bool(info.get("crashed", False))))

        if frames:
            video_arr = np.stack(frames, axis=0).transpose(0, 3, 1, 2).astype(np.uint8)
            videos.append(wandb.Video(video_arr, caption=f"ep_{ep}_seed_{seed}",
                                      fps=args.fps, format="mp4"))

    env.close()

    metrics = {
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/std_reward": float(np.std(rewards)),
        "eval/median_reward": float(np.median(rewards)),
        "eval/min_reward": float(np.min(rewards)),
        "eval/max_reward": float(np.max(rewards)),
        "eval/mean_episode_length": float(np.mean(lengths)),
        "eval/n_episodes": args.n_episodes,
        "eval/crash_rate": crashed_count / args.n_episodes,
        "eval/early_termination_rate": terminated_count / args.n_episodes,
        "eval/truncation_rate": truncated_count / args.n_episodes,
    }
    wb_run.log(metrics)
    wb_run.summary.update(metrics)

    # All 50 videos as one grouped media panel
    if videos:
        wb_run.log({"eval/videos": videos})

    # Per-episode table
    table = wandb.Table(columns=["episode", "seed", "reward", "length",
                                 "terminated", "truncated", "crashed"])
    for row in per_episode_rows:
        table.add_data(*row)
    wb_run.log({"eval/per_episode": table})

    # Persist alongside the model
    out_dir = Path(args.config).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_full_videos.json"
    with open(summary_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "per_episode": [
                {"episode": r[0], "seed": r[1], "reward": r[2], "length": r[3],
                 "terminated": r[4], "truncated": r[5], "crashed": r[6]}
                for r in per_episode_rows
            ],
        }, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nW&B: {wb_run.url}")
    print(f"Summary: {summary_path}")
    wb_run.finish()


if __name__ == "__main__":
    main()
