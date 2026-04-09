import yaml
import gymnasium as gym
import highway_env  # noqa: F401 — registers envs
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

ALGOS = {"PPO": PPO, "SAC": SAC, "DQN": DQN}

def run(config_path: str, smoke: bool = False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env = make_vec_env(cfg["env_id"], n_envs=cfg.get("n_envs", 1),
                       seed=cfg["seed"])
    AlgoCls = ALGOS[cfg["algo"]]
    model = AlgoCls(cfg["policy"], env, seed=cfg["seed"],
                    tensorboard_log=cfg["tb_dir"], **cfg.get("algo_kwargs", {}))

    timesteps = 1_000 if smoke else cfg["total_timesteps"]
    model.learn(total_timesteps=timesteps)
    model.save(cfg["model_path"])

    eval_env = gym.make(cfg["env_id"])
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    return {"mean_reward": float(mean), "reward_std": float(std)}