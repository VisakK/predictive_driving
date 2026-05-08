"""Standalone eval for a saved PPO model against a config. Prints JSON metrics."""
import argparse, json, yaml
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n_episodes", type=int, default=50)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_config = cfg.get("env_config")
    env = gym.make(cfg["env_id"], config=env_config)
    model = PPO.load(args.model)

    seed = cfg["seed"] + 10_000
    ep_rewards, ep_lens = [], []
    term_ct = trunc_ct = crash_ct = 0
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        r = 0.0
        l = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, rw, terminated, truncated, info = env.step(a)
            r += float(rw)
            l += 1
            done = terminated or truncated
        ep_rewards.append(r)
        ep_lens.append(l)
        if terminated:
            term_ct += 1
        if truncated:
            trunc_ct += 1
        if info.get("crashed", False):
            crash_ct += 1
    env.close()
    metrics = {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "min_reward": float(np.min(ep_rewards)),
        "max_reward": float(np.max(ep_rewards)),
        "mean_episode_length": float(np.mean(ep_lens)),
        "n_episodes": args.n_episodes,
        "early_termination_rate": term_ct / args.n_episodes,
        "truncation_rate": trunc_ct / args.n_episodes,
        "crash_rate": crash_ct / args.n_episodes,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
