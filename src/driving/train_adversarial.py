"""Training script for adversarial ViT+CVAE+Discriminator experiments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import tqdm
import wandb
import yaml
from wandb.integration.sb3 import WandbCallback

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401 — registers adversarial envs
import driving.adversarial_v3  # noqa: F401 — registers v3 archetype envs
import driving.adversarial_v3_ts  # noqa: F401 — registers v3-ts target-speed envs
import driving.adversarial_v3i  # noqa: F401 — registers v3i interaction envs

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from driving.adversarial import DictObsWrapper
from driving.adversarial_ppo import AdversarialPPO
from driving.vit_cvae import ViTCVAEExtractor


def evaluate_adversarial(
    model: AdversarialPPO,
    env_id: str,
    n_episodes: int = 50,
    n_video_episodes: int = 5,
    deterministic: bool = True,
    seed: int = 0,
    env_config: dict | None = None,
    video_dir: str | Path | None = None,
) -> dict:
    """Evaluate with adversarial-specific metrics."""
    make_kwargs = {"render_mode": "rgb_array"} if n_video_episodes > 0 else {}
    if env_config:
        make_kwargs["config"] = env_config
    env = gym.make(env_id, **make_kwargs)
    video_path = Path(video_dir) if video_dir is not None else None
    if video_path is not None:
        video_path.mkdir(parents=True, exist_ok=True)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    crashed_count = 0
    crashed_adversarial_count = 0
    crashed_nominal_count = 0
    terminated_count = 0
    truncated_count = 0

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
            if info.get("crashed_with_adversarial", False):
                crashed_adversarial_count += 1
            else:
                crashed_nominal_count += 1

        if record and frames:
            video = (
                np.stack(frames, axis=0)
                .transpose(0, 3, 1, 2)
                .astype(np.uint8)
            )
            if video_path is not None:
                imageio.mimsave(
                    video_path / f"eval_video_ep_{ep}.mp4",
                    frames,
                    fps=10,
                )
            run = getattr(wandb, "run", None)
            if run is not None and not getattr(run, "disabled", False):
                wandb.log(
                    {f"eval/video_ep_{ep}": wandb.Video(video, fps=10, format="mp4")}
                )

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
        "eval/crash_rate_adversarial": crashed_adversarial_count / n_episodes,
        "eval/crash_rate_nominal": crashed_nominal_count / n_episodes,
    }
    wandb.log(metrics)
    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def run(config_path: str, run_name: str, smoke: bool = False,
        device_override: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if device_override is not None:
        cfg["device"] = device_override

    adv_cfg = cfg.get("adversarial_config", {})

    wb_run = wandb.init(
        project="predictive_driving",
        name=run_name,
        config=cfg,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env_config = cfg.get("env_config")
    env_kwargs = {"config": env_config} if env_config else {}
    n_envs = 1 if smoke else cfg.get("n_envs", 4)
    vec_cls = None if smoke else SubprocVecEnv
    vec_env_kwargs = {}
    if vec_cls is not None and cfg.get("vec_env_start_method"):
        vec_env_kwargs["start_method"] = cfg["vec_env_start_method"]
    env = make_vec_env(
        cfg["env_id"],
        n_envs=n_envs,
        seed=cfg["seed"],
        env_kwargs=env_kwargs,
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    n_channels = len(env_config.get("observation", {}).get("features", []))
    grid_size = env_config.get("observation", {}).get("grid_size", [[-27.5, 27.5], [-27.5, 27.5]])
    grid_step = env_config.get("observation", {}).get("grid_step", [5, 5])
    grid_h = len(np.arange(grid_size[0][0], grid_size[0][1], grid_step[0]))
    grid_w = len(np.arange(grid_size[1][0], grid_size[1][1], grid_step[1]))

    policy_kwargs = {
        "features_extractor_class": ViTCVAEExtractor,
        "features_extractor_kwargs": {
            "features_dim": adv_cfg.get("vit_embed_dim", 64),
            "n_agents": adv_cfg.get("n_kinematic_agents", 15),
            "agent_feat_dim": 7,
            "vit_embed_dim": adv_cfg.get("vit_embed_dim", 64),
            "vit_n_heads": adv_cfg.get("vit_n_heads", 4),
            "vit_n_layers": adv_cfg.get("vit_n_layers", 3),
            "cvae_latent_dim": adv_cfg.get("cvae_latent_dim", 32),
            "cvae_hidden_dim": adv_cfg.get("cvae_hidden_dim", 128),
            "disc_hidden_dim": adv_cfg.get("disc_hidden_dim", 128),
            "use_kinematics_policy": adv_cfg.get("use_kinematics_policy", False),
            "n_history_frames": adv_cfg.get("n_history_frames", 10),
            "kin_encoder_hidden": adv_cfg.get("kin_encoder_hidden", 128),
            "kin_encoder_out": adv_cfg.get("kin_encoder_out", 64),
            "use_anomaly_policy": adv_cfg.get("use_anomaly_policy", False),
            "anomaly_encoder_hidden": adv_cfg.get("anomaly_encoder_hidden", 64),
            "anomaly_encoder_out": adv_cfg.get("anomaly_encoder_out", 32),
            "use_online_predictor": adv_cfg.get("use_online_predictor", False),
            "use_learned_anomaly_policy": adv_cfg.get(
                "use_learned_anomaly_policy", False
            ),
            "predictor_hidden_dim": adv_cfg.get("predictor_hidden_dim", 128),
            "use_anomaly_attention_policy": adv_cfg.get(
                "use_anomaly_attention_policy", False
            ),
            "anomaly_attn_embed_dim": adv_cfg.get("anomaly_attn_embed_dim", 64),
            "anomaly_attn_n_heads": adv_cfg.get("anomaly_attn_n_heads", 4),
            "anomaly_attn_spatial_sigma": adv_cfg.get(
                "anomaly_attn_spatial_sigma", 1.5
            ),
            "anomaly_attn_use_risk_bias": adv_cfg.get(
                "anomaly_attn_use_risk_bias", False
            ),
            "anomaly_attn_use_per_slot_gru": adv_cfg.get(
                "anomaly_attn_use_per_slot_gru", False
            ),
            "anomaly_attn_gru_hidden": adv_cfg.get(
                "anomaly_attn_gru_hidden", 32
            ),
        },
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    }

    algo_kwargs = dict(cfg.get("algo_kwargs", {}))
    timesteps = 1_000 if smoke else cfg["total_timesteps"]
    if smoke:
        algo_kwargs["n_steps"] = 128
        algo_kwargs.setdefault("batch_size", 64)

    model = AdversarialPPO(
        "MultiInputPolicy",
        env,
        alpha=adv_cfg.get("alpha", 0.1),
        beta=adv_cfg.get("beta", 0.1),
        anomaly_reward_weight=adv_cfg.get("anomaly_reward_weight", 0.5),
        pbs_weight=adv_cfg.get("pbs_weight", 0.0),
        truncation_bonus_weight=adv_cfg.get("truncation_bonus_weight", 0.0),
        predictor_loss_weight=adv_cfg.get("predictor_loss_weight", 0.0),
        seed=cfg["seed"],
        tensorboard_log=cfg["tb_dir"],
        verbose=cfg.get("verbose", 1),
        device=cfg.get("device", "auto"),
        policy_kwargs=policy_kwargs,
        **algo_kwargs,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=WandbCallback(
            model_save_path=f"{cfg['model_path']}/{wb_run.id}",
            verbose=cfg.get("verbose", 1),
        ),
        progress_bar=True,
    )
    model.save(cfg["model_path"])

    results_dir = Path(cfg["model_path"]).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_episodes = 0 if smoke else cfg.get("eval_episodes", 50)
    if eval_episodes > 0:
        metrics = evaluate_adversarial(
            model,
            cfg["env_id"],
            n_episodes=eval_episodes,
            n_video_episodes=0 if smoke else cfg.get("eval_video_episodes", 5),
            seed=cfg["seed"] + 10_000,
            env_config=env_config,
            video_dir=results_dir / "videos",
        )
    else:
        metrics = {"eval/skipped": 1.0}

    summary_path = results_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# {run_name}\n\n")
        f.write(f"## Config\n")
        f.write(f"- Environment: {cfg['env_id']}\n")
        f.write(f"- Timesteps: {timesteps}\n")
        f.write(f"- Seed: {cfg['seed']}\n")
        f.write(f"- Alpha (CVAE): {adv_cfg.get('alpha', 0.1)}\n")
        f.write(f"- Beta (Disc): {adv_cfg.get('beta', 0.1)}\n")
        f.write(f"- Anomaly reward weight: {adv_cfg.get('anomaly_reward_weight', 0.5)}\n")
        f.write(f"- PBS weight: {adv_cfg.get('pbs_weight', 0.0)}\n")
        f.write(f"- Truncation bonus weight: {adv_cfg.get('truncation_bonus_weight', 0.0)}\n\n")
        f.write(f"- Predictor loss weight: {adv_cfg.get('predictor_loss_weight', 0.0)}\n")
        f.write(f"- Anomaly policy input: {adv_cfg.get('use_anomaly_policy', False)}\n")
        f.write(f"- Learned anomaly policy input: {adv_cfg.get('use_learned_anomaly_policy', False)}\n\n")
        if eval_episodes > 0 and cfg.get("eval_video_episodes", 5) > 0:
            f.write(f"- Local eval videos: {results_dir / 'videos'}\n\n")
        f.write(f"## Results\n")
        for k, v in sorted(metrics.items()):
            f.write(f"- {k}: {v:.4f}\n")
        f.write(f"\n## W&B Run\n")
        f.write(f"- Run ID: {wb_run.id}\n")
        f.write(f"- URL: {wb_run.url}\n")

    wandb.finish()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="adversarial_run")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu/cuda/auto). Defaults to cfg.device or 'auto'.")
    args = parser.parse_args()
    run(args.config, args.run_name, args.smoke, device_override=args.device)
