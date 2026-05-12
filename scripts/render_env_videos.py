"""Render N video episodes of a trained model on a chosen env, save mp4
files locally. Lightweight sibling of eval_model_with_videos.py without
the W&B dependency — used to preview new environments quickly.

Usage:
  python scripts/render_env_videos.py \
      --config experiments/061_baseline_v3ts/config.yaml \
      --model experiments/061_baseline_v3ts/results/model.zip \
      --env-id adversarial-highway-v3i-raw \
      --out-dir env_design/061 \
      --n-episodes 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import gymnasium as gym
import yaml

from stable_baselines3 import PPO, SAC, DQN  # noqa: F401

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401
import driving.adversarial_v3  # noqa: F401
import driving.adversarial_v3_ts  # noqa: F401
import driving.adversarial_v3i  # noqa: F401
import driving.baseline_continuous  # noqa: F401

ALGOS = {"PPO": PPO, "SAC": SAC, "DQN": DQN}
try:
    from driving.adversarial_ppo import AdversarialPPO
    ALGOS["AdversarialPPO"] = AdversarialPPO
except Exception:
    pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--env-id", required=True,
                   help="Override env_id from config (e.g. adversarial-highway-v3i-raw).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=10_000)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algo = cfg.get("algo", "PPO")
    AlgoCls = ALGOS[algo]
    model = AlgoCls.load(args.model, device="cpu")

    env_config = cfg.get("env_config")
    env = gym.make(args.env_id, render_mode="rgb_array", config=env_config)

    base_seed = cfg["seed"] + args.seed_offset
    rows = []
    for ep in range(args.n_episodes):
        seed = base_seed + ep
        obs, info = env.reset(seed=seed)
        frames = []
        r_total = 0.0
        length = 0
        done = False
        # Spawn-time stats
        base = env.unwrapped
        ego = base.vehicle
        others = [v for v in base.road.vehicles if v is not ego]
        dists = [float(np.linalg.norm(v.position - ego.position)) for v in others]
        adv = [getattr(v, "archetype", None) for v in others
               if getattr(v, "is_adversarial", False)]
        spawn_archetype = adv[0] if adv else "none"
        nearest_t0 = float(min(dists)) if dists else float("inf")
        in_30m_t0 = int(sum(1 for d in dists if d <= 30.0))

        while not done:
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, rw, terminated, truncated, info = env.step(action)
            r_total += float(rw)
            length += 1
            done = terminated or truncated

        crashed = bool(info.get("crashed", False))
        crashed_with = info.get("crashed_with_archetype")
        video_arr = np.stack(frames, axis=0).astype(np.uint8)
        out_path = out_dir / f"ep{ep:02d}_seed{seed}_arch-{spawn_archetype}.mp4"
        imageio.mimsave(out_path, video_arr, fps=args.fps,
                        codec="libx264", quality=8)
        rows.append({
            "episode": ep,
            "seed": seed,
            "reward": r_total,
            "length": length,
            "crashed": crashed,
            "crashed_with_archetype": crashed_with,
            "spawn_archetype": spawn_archetype,
            "nearest_t0": nearest_t0,
            "in_30m_t0": in_30m_t0,
            "video": str(out_path.name),
        })
        print(f"  ep{ep:02d} seed={seed} arch={spawn_archetype:14s} "
              f"r={r_total:6.2f} len={length:2d} crashed={crashed} "
              f"crashed_with={crashed_with}")

    env.close()
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "env_id": args.env_id,
            "config": args.config,
            "model": args.model,
            "n_episodes": args.n_episodes,
            "episodes": rows,
        }, f, indent=2)
    print(f"\nWrote {args.n_episodes} videos + summary.json → {out_dir}")


if __name__ == "__main__":
    main()
