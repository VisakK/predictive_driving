import yaml
import numpy as np
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
import highway_env  # noqa: F401 — registers envs
import driving.envs  # noqa: F401 — registers custom envs
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import tqdm

ALGOS = {"PPO": PPO, "SAC": SAC, "DQN": DQN}


def evaluate_and_log(model, env_id: str, n_episodes: int = 50,
                     n_video_episodes: int = 5, deterministic: bool = True,
                     seed: int = 0, env_config: dict | None = None) -> dict:
    """Run episodes, log aggregate stats + a few rendered videos to wandb.

    Returns a dict of summary metrics. The first `n_video_episodes` are
    recorded frame-by-frame and uploaded as wandb.Video to the active run.
    """
    make_kwargs = {"render_mode": "rgb_array"}
    if env_config:
        make_kwargs["config"] = env_config
    env = gym.make(env_id, **make_kwargs)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    terminated_count = 0   # early termination (e.g. crash), not truncation
    truncated_count = 0
    crashed_count = 0

    for ep in tqdm.tqdm(range(n_episodes), desc="Evaluating"):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        frames: list[np.ndarray] = []
        record = ep < n_video_episodes

        while not done:
            if record:
                frames.append(env.render())
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if info.get("crashed", False):
            crashed_count += 1

        if record and frames:
            # wandb.Video expects (time, channel, height, width) uint8
            video = np.stack(frames, axis=0).transpose(0, 3, 1, 2).astype(np.uint8)
            wandb.log({f"eval/video_ep_{ep}": wandb.Video(video, fps=10, format="mp4")})

    env.close()

    metrics = {
        "eval/mean_reward": float(np.mean(episode_rewards)),
        "eval/std_reward": float(np.std(episode_rewards)),
        "eval/min_reward": float(np.min(episode_rewards)),
        "eval/max_reward": float(np.max(episode_rewards)),
        "eval/mean_episode_length": float(np.mean(episode_lengths)),
        "eval/n_episodes": n_episodes,
        "eval/early_termination_rate": terminated_count / n_episodes,
        "eval/truncation_rate": truncated_count / n_episodes,
        "eval/crash_rate": crashed_count / n_episodes,
    }
    wandb.log(metrics)
    print(f"Logged evaluation metrics")
    return metrics

def run(config_path: str, run_name: str, smoke: bool = False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run = wandb.init(project="predictive_driving",
                name=run_name,
                sync_tensorboard=True,
                monitor_gym=True,       # Auto-upload videos of the agent
                save_code=True,)
    env_config = cfg.get("env_config")
    env_kwargs = {"config": env_config} if env_config else {}
    env = make_vec_env(cfg["env_id"], n_envs=cfg.get("n_envs", 1),
                       seed=cfg["seed"], env_kwargs=env_kwargs,
                       vec_env_cls=SubprocVecEnv)
    AlgoCls = ALGOS[cfg["algo"]]
    algo_kwargs = dict(cfg.get("algo_kwargs", {}))
    timesteps = 1_000 if smoke else cfg["total_timesteps"]
    if smoke and cfg["algo"] in {"PPO", "A2C"}:
        # On-policy algos collect a full n_steps rollout before checking
        # total_timesteps, so shrink it to keep smoke tests actually small.
        algo_kwargs["n_steps"] = 128
        algo_kwargs.setdefault("batch_size", 64)
    model = AlgoCls(cfg["policy"], env, seed=cfg["seed"],
                    tensorboard_log=cfg["tb_dir"], verbose=cfg.get("verbose", 1), device="cpu", **algo_kwargs)
    model.learn(total_timesteps=timesteps, callback=WandbCallback(model_save_path=f"{cfg['model_path']}/{run.id}",
        verbose=cfg.get("verbose", 1)),progress_bar=True)
    model.save(cfg["model_path"])

    metrics = evaluate_and_log(
        model,
        cfg["env_id"],
        n_episodes=3 if smoke else cfg.get("eval_episodes", 50),
        n_video_episodes=1 if smoke else cfg.get("eval_video_episodes", 5),
        seed=cfg["seed"] + 10_000,
        env_config=env_config,
    )
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")
    parser.add_argument("--run_name", type=str, default="ppo_run")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.run_name, args.smoke)