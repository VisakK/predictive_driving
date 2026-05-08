"""Fair evaluation: all models on identical adversarial environments with matched seeds.

Each model faces the same 100 adversarial scenarios per environment type.
Models differ in architecture and observation space but face identical traffic.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tqdm

from stable_baselines3 import PPO

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401

from driving.adversarial import (
    AdversarialHighwayEnv,
    AdversarialHighwayV2Env,
    AdversarialRoundaboutEnv,
    AdversarialRoundaboutV2Env,
    DictObsWrapper,
    ExpectedObservedDictObsWrapper,
    HorizonExpectedObservedDictObsWrapper,
    KinHistoryDictObsWrapper,
    OccGridFrameStack,
)
from driving.adversarial_ppo import AdversarialPPO


SEEDS = list(range(1000, 1100))

OBS_CONFIG_BASELINE = {
    "type": "OccupancyGrid",
    "vehicles_count": 15,
    "features": ["presence", "vx", "vy", "on_road"],
    "features_range": {"vx": [-20, 20], "vy": [-20, 20]},
    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    "grid_step": [5, 5],
    "absolute": False,
}

OBS_CONFIG_5CH = {
    "type": "OccupancyGrid",
    "vehicles_count": 15,
    "features": ["presence", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {"vx": [-20, 20], "vy": [-20, 20]},
    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    "grid_step": [5, 5],
    "absolute": False,
}

HIGHWAY_BASE = {
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "duration": 80,
    "normalize_reward": False,
    "vehicles_count": 100,
}

ROUNDABOUT_BASE = {
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "duration": 22,
    "normalize_reward": False,
}


def make_env_config(base: dict, obs: dict) -> dict:
    return {**base, "observation": obs}


def make_raw_env(EnvCls, cfg):
    return EnvCls(config=cfg)


def make_dict_env(EnvCls, cfg):
    return DictObsWrapper(EnvCls(config=cfg))


def make_framestack_env(EnvCls, cfg):
    return OccGridFrameStack(DictObsWrapper(EnvCls(config=cfg)), n_frames=3)


def make_kinhistory_env(EnvCls, cfg):
    return KinHistoryDictObsWrapper(EnvCls(config=cfg))


def make_expected_env(EnvCls, cfg):
    return ExpectedObservedDictObsWrapper(EnvCls(config=cfg))


def make_expected_h10_env(EnvCls, cfg):
    return HorizonExpectedObservedDictObsWrapper(EnvCls(config=cfg), horizon=10)


# Evaluation uses the v2 environment (proximity-guaranteed adversarial placement)
# for ALL models. This is a harder test than the v0 env where old models were
# trained — it measures how each model generalizes to a more interactive
# adversarial distribution.
HIGHWAY_EVAL_ENV = AdversarialHighwayV2Env
ROUNDABOUT_EVAL_ENV = AdversarialRoundaboutV2Env


MODELS_HIGHWAY = [
    {
        "name": "Baseline PPO (027)",
        "short": "Baseline",
        "model_path": "experiments/027_ppo_highway_occgrid_rerun/results/model.zip",
        "model_cls": PPO,
        "env_factory": lambda cfg: make_raw_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_BASELINE),
    },
    {
        "name": "ViT+CVAE+Disc (033)",
        "short": "ViT+CVAE+Disc",
        "model_path": "experiments/033_adversarial_highway_350k/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_dict_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT-only (036)",
        "short": "ViT-only",
        "model_path": "experiments/036_ablation_vit_only_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_dict_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT+CVAE+Disc+3frame (038)",
        "short": "ViT+CVAE+Disc+3f",
        "model_path": "experiments/038_ablation_framestack_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_framestack_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT+CVAE+Disc+KinHist v2 (040)",
        "short": "ViT+CVAE+Disc+KinHist",
        "model_path": "experiments/040_adversarial_highway_v2/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_kinhistory_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput B (042)",
        "short": "ExpectedInput",
        "model_path": "experiments/042_expected_input_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_expected_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput-H10 (048)",
        "short": "ExpectedInput-H10",
        "model_path": "experiments/048_expected_horizon_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_expected_h10_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput+Reward D (044)",
        "short": "ExpectedInput+Reward",
        "model_path": "experiments/044_expected_input_reward_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_expected_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedOnline E (046)",
        "short": "ExpectedOnline",
        "model_path": "experiments/046_expected_online_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_expected_env(HIGHWAY_EVAL_ENV, cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
]

MODELS_ROUNDABOUT = [
    {
        "name": "Baseline PPO (030)",
        "short": "Baseline",
        "model_path": "experiments/030_ppo_roundabout_occgrid_rerun/results/model.zip",
        "model_cls": PPO,
        "env_factory": lambda cfg: make_raw_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_BASELINE),
    },
    {
        "name": "ViT+CVAE+Disc (034)",
        "short": "ViT+CVAE+Disc",
        "model_path": "experiments/034_adversarial_roundabout_350k/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_dict_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT-only (037)",
        "short": "ViT-only",
        "model_path": "experiments/037_ablation_vit_only_roundabout/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_dict_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT+CVAE+Disc+3frame (039)",
        "short": "ViT+CVAE+Disc+3f",
        "model_path": "experiments/039_ablation_framestack_roundabout/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_framestack_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT+CVAE+Disc+KinHist v2 (041)",
        "short": "ViT+CVAE+Disc+KinHist",
        "model_path": "experiments/041_adversarial_roundabout_v2/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_kinhistory_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput B (043)",
        "short": "ExpectedInput",
        "model_path": "experiments/043_expected_input_roundabout/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_expected_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput+Reward D (045)",
        "short": "ExpectedInput+Reward",
        "model_path": "experiments/045_expected_input_reward_roundabout/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_expected_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedOnline E (047)",
        "short": "ExpectedOnline",
        "model_path": "experiments/047_expected_online_roundabout/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_expected_env(ROUNDABOUT_EVAL_ENV, cfg),
        "env_config": make_env_config(ROUNDABOUT_BASE, OBS_CONFIG_5CH),
    },
]


def check_adversarial_crash(env) -> bool:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if not base_env.vehicle.crashed:
        return False
    for v in base_env.road.vehicles:
        if v is base_env.vehicle:
            continue
        if v.crashed and getattr(v, "is_adversarial", False):
            return True
    return False


def count_adversarial_agents(env) -> tuple[int, int]:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    ego = base_env.vehicle
    others = [v for v in base_env.road.vehicles if v is not ego]
    n_adv = sum(1 for v in others if getattr(v, "is_adversarial", False))
    return n_adv, len(others)


def evaluate_model(model, env, seeds: list[int], label: str) -> list[dict]:
    results = []
    for seed in tqdm.tqdm(seeds, desc=f"  {label}"):
        obs, info = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        ep_len = 0
        n_adv, n_total = count_adversarial_agents(env)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated

        crashed = info.get("crashed", False)
        if "crashed_with_adversarial" in info:
            crashed_adversarial = info["crashed_with_adversarial"]
        else:
            crashed_adversarial = check_adversarial_crash(env) if crashed else False

        results.append({
            "seed": seed,
            "reward": ep_reward,
            "length": ep_len,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": bool(crashed),
            "crashed_adversarial": bool(crashed_adversarial),
            "crashed_nominal": bool(crashed and not crashed_adversarial),
            "n_adversarial_agents": n_adv,
            "n_total_agents": n_total,
        })
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    return {
        "n_episodes": n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "crash_rate": sum(r["crashed"] for r in results) / n,
        "crash_rate_adversarial": sum(r["crashed_adversarial"] for r in results) / n,
        "crash_rate_nominal": sum(r["crashed_nominal"] for r in results) / n,
        "early_termination_rate": sum(r["terminated"] for r in results) / n,
        "truncation_rate": sum(r["truncated"] for r in results) / n,
        "mean_adversarial_agents": float(np.mean([r["n_adversarial_agents"] for r in results])),
        "mean_total_agents": float(np.mean([r["n_total_agents"] for r in results])),
    }


def run_scenario(scenario: str, models: list[dict]) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Fair Evaluation — {scenario.upper()}")
    print(f"{'=' * 60}")

    all_results = {}
    for m in models:
        print(f"\nLoading {m['name']}: {m['model_path']}")
        model = m["model_cls"].load(m["model_path"], device="cpu")

        print(f"Evaluating {m['name']}...")
        env = m["env_factory"](m["env_config"])
        eps = evaluate_model(model, env, SEEDS, m["short"])
        env.close()

        agg = aggregate(eps)
        print(f"  -> mean_reward={agg['mean_reward']:.4f}, crash_rate={agg['crash_rate']*100:.1f}%")

        all_results[m["short"]] = {
            "name": m["name"],
            "model_path": m["model_path"],
            "per_episode": eps,
            "aggregate": agg,
        }

    return {"scenario": scenario, "models": all_results}


def fmt(v, pct=False):
    return f"{v * 100:.1f}%" if pct else f"{v:.4f}"


def multi_model_table(models_data: dict) -> str:
    names = list(models_data.keys())
    rows = [
        ("Mean Reward", "mean_reward", False),
        ("Std Reward", "std_reward", False),
        ("Min Reward", "min_reward", False),
        ("Max Reward", "max_reward", False),
        ("Median Reward", "median_reward", False),
        ("Mean Episode Length", "mean_episode_length", False),
        ("Crash Rate (total)", "crash_rate", True),
        ("Crash Rate (adversarial)", "crash_rate_adversarial", True),
        ("Crash Rate (nominal)", "crash_rate_nominal", True),
        ("Early Termination Rate", "early_termination_rate", True),
        ("Truncation Rate (survival)", "truncation_rate", True),
    ]

    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|---|" + "|".join(["---"] * len(names)) + "|"
    lines = [header, sep]

    for label, key, pct in rows:
        vals = []
        raw_vals = [models_data[n]["aggregate"][key] for n in names]
        best_idx = None
        if key == "crash_rate" or key == "crash_rate_adversarial" or key == "crash_rate_nominal" or key == "early_termination_rate":
            best_idx = int(np.argmin(raw_vals))
        elif key in ("mean_reward", "max_reward", "median_reward", "truncation_rate", "mean_episode_length"):
            best_idx = int(np.argmax(raw_vals))

        for i, n in enumerate(names):
            v = models_data[n]["aggregate"][key]
            s = fmt(v, pct)
            if i == best_idx:
                s = f"**{s}**"
            vals.append(s)
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def pairwise_crash_analysis(models_data: dict) -> str:
    names = list(models_data.keys())
    lines = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            eps1 = models_data[n1]["per_episode"]
            eps2 = models_data[n2]["per_episode"]
            only1 = sum(1 for a, b in zip(eps1, eps2) if a["crashed"] and not b["crashed"])
            only2 = sum(1 for a, b in zip(eps1, eps2) if b["crashed"] and not a["crashed"])
            both = sum(1 for a, b in zip(eps1, eps2) if a["crashed"] and b["crashed"])
            neither = sum(1 for a, b in zip(eps1, eps2) if not a["crashed"] and not b["crashed"])
            lines.append(
                f"- **{n1} vs {n2}**: only {n1} crashed {only1}, "
                f"only {n2} crashed {only2}, both {both}, neither {neither}"
            )
    return "\n".join(lines)


def write_summary(hw_data: dict, rb_data: dict, path: str):
    with open(path, "w") as f:
        f.write("# Fair Evaluation: All Models on Adversarial Environments\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Methodology\n\n")
        f.write(
            "All models were evaluated on the **v2 adversarial environments** which use\n"
            "**proximity-guaranteed adversarial placement**: at least 3 of the 8 agents nearest\n"
            "to the ego on highway (and 2 of the 4 nearest on roundabout) are aggressive\n"
            "IDM/MOBIL vehicles, with the remainder sampled uniformly across the rest of\n"
            "the population. This is a harder distribution than the v0 environment on which\n"
            "models 033/034/036/037/038/039 were trained — it tests how each model\n"
            "generalizes to traffic where adversarial agents are guaranteed to interact with\n"
            "the ego. Each model ran 100 episodes with seeds 1000-1099 on identical traffic.\n\n"
        )

        f.write("### Models evaluated\n\n")
        f.write("| Model | Architecture | Trained on | Aux losses | Obs |\n")
        f.write("|---|---|---|---|---|\n")
        f.write("| Baseline | PPO + MlpPolicy | Nominal traffic | None | OccGrid 4ch |\n")
        f.write("| ViT+CVAE+Disc | PPO + ViT + CVAE + Disc | v0 adversarial | CVAE + Disc (a=0.1, b=0.1) | OccGrid 5ch |\n")
        f.write("| ViT-only | PPO + ViT (no aux) | v0 adversarial | None (a=0, b=0) | OccGrid 5ch |\n")
        f.write("| ViT+CVAE+Disc+3f | PPO + ViT + CVAE + Disc | v0 adversarial | CVAE + Disc (a=0.1, b=0.1) | OccGrid 3-frame stack |\n")
        f.write("| ViT+CVAE+Disc+KinHist | PPO + ViT + CVAE + Disc + KinMLP | **v2 adversarial** | CVAE + Disc (a=0.01, b=0.1), anomaly reward 0.5 | OccGrid + 10-frame agent kinematics |\n")
        f.write("| ExpectedInput | PPO + ViT + causal expected-vs-observed anomaly | v2 adversarial | None | OccGrid + causal anomaly features |\n")
        f.write("| ExpectedInput-H10 | PPO + ViT + causal 10-frame pending prediction anomaly | v2 adversarial | None | OccGrid + H10 causal anomaly features |\n")
        f.write("| ExpectedInput+Reward | PPO + ViT + causal expected-vs-observed anomaly | v2 adversarial | Anomaly reward 0.5 | OccGrid + causal anomaly features |\n")
        f.write("| ExpectedOnline | PPO + ViT + online learned predictor anomaly | v2 adversarial | Predictor loss 0.1, anomaly reward 0.5 | OccGrid + learned anomaly features |\n")

        f.write("\n---\n\n## Highway Results\n\n")
        f.write(multi_model_table(hw_data["models"]))
        f.write("\n\n### Pairwise Crash Analysis\n\n")
        f.write(pairwise_crash_analysis(hw_data["models"]))

        f.write("\n\n---\n\n## Roundabout Results\n\n")
        f.write(multi_model_table(rb_data["models"]))
        f.write("\n\n### Pairwise Crash Analysis\n\n")
        f.write(pairwise_crash_analysis(rb_data["models"]))

        # Key findings
        f.write("\n\n---\n\n## Key Findings\n\n")

        for scenario, data in [("Highway", hw_data), ("Roundabout", rb_data)]:
            f.write(f"### {scenario}\n\n")
            models = data["models"]
            ranked = sorted(models.items(), key=lambda x: x[1]["aggregate"]["crash_rate"])
            f.write(f"**Crash rate ranking** (lower is better):\n\n")
            for i, (name, mdata) in enumerate(ranked, 1):
                cr = mdata["aggregate"]["crash_rate"]
                mr = mdata["aggregate"]["mean_reward"]
                f.write(f"{i}. **{name}**: {cr*100:.1f}% crash rate, {mr:.2f} mean reward\n")
            f.write("\n")

        f.write("\n---\n\n## Experiment Details\n\n")
        f.write(f"- Seeds: 1000–1099 (100 episodes per model per scenario)\n")
        total_eps = len(SEEDS) * (len(hw_data["models"]) + len(rb_data["models"]))
        f.write(f"- Total episodes evaluated: {total_eps}\n")
        f.write(f"- Deterministic policy: yes\n")
        f.write(f"- All training: 350,000 timesteps, seed=42\n")
        f.write(f"- Evaluation environment: v2 (proximity-guaranteed adversarial placement)\n")

    json_path = path.replace(".md", ".json")
    raw = {}
    for scenario, data in [("highway", hw_data), ("roundabout", rb_data)]:
        raw[scenario] = {}
        for name, mdata in data["models"].items():
            raw[scenario][name] = {
                "aggregate": mdata["aggregate"],
                "per_episode": mdata["per_episode"],
            }
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\nSummary: {path}")
    print(f"Raw data: {json_path}")


if __name__ == "__main__":
    hw = run_scenario("highway", MODELS_HIGHWAY)
    rb = run_scenario("roundabout", MODELS_ROUNDABOUT)

    out = Path("experiments/035_fair_eval")
    out.mkdir(parents=True, exist_ok=True)
    write_summary(hw, rb, str(out / "fair_eval_summary.md"))
