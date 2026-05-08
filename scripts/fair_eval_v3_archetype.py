"""V3 fair eval — per-archetype crash analysis, 500 seeds.

Models compared (highway only, all evaluated on AdversarialHighwayV3Env with
default mixture: tailgater 0.40 / sudden_braker 0.20 / lane_drifter 0.20 /
erratic_speed 0.20):

  - Baseline PPO (027) — nominal-trained
  - ViT+CVAE+Disc (033) — trained on v0 adversarial
  - ExpectedInput-H10 (048) — trained on v2 adversarial
  - ExpectedInput-H10+Aux (049) — trained on v2 adversarial with aux losses
  - H10 + ExpectedInput on v3 (050) — trained on v3 adversarial
  - ViT-only on v3 (051) — trained on v3 adversarial, no anomaly input

Outputs:
  experiments/050_h10_v3_highway/results/fair_eval_v3.{md,json}
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import tqdm

from stable_baselines3 import PPO

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401
import driving.adversarial_v3  # noqa: F401

from driving.adversarial import DictObsWrapper
from driving.adversarial_v3 import (
    AdversarialHighwayV3Env,
    DictObsWrapperV3,
    HorizonExpectedObservedDictObsWrapperV3,
    KinHistoryDictObsWrapperV3,
)
from driving.adversarial_ppo import AdversarialPPO


SEEDS = list(range(1000, 1500))  # 500 episodes per model
ARCHETYPES = ["tailgater", "sudden_braker", "lane_drifter", "erratic_speed"]


HIGHWAY_BASE = {
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "duration": 80,
    "normalize_reward": False,
    "vehicles_count": 100,
}

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


def make_env_config(base: dict, obs: dict) -> dict:
    return {**base, "observation": obs}


def make_v3_raw(cfg):
    return AdversarialHighwayV3Env(config=cfg)


def make_v3_dict(cfg):
    """DictObsWrapper (4ch obs) — matches 033's training wrapper.
    Uses the v0/v2-style DictObsWrapper for obs compatibility, but archetype
    info is queried from the underlying env (which is v3)."""
    return DictObsWrapper(AdversarialHighwayV3Env(config=cfg))


def make_v3_dict_v3wrapped(cfg):
    """DictObsWrapperV3 — exposes archetype info; same obs space as DictObsWrapper.
    Used for 051 (ViT-only on v3) since 051 trained against this exact wrapper."""
    return DictObsWrapperV3(AdversarialHighwayV3Env(config=cfg))


def make_v3_h10(cfg):
    return HorizonExpectedObservedDictObsWrapperV3(
        AdversarialHighwayV3Env(config=cfg), horizon=10
    )


MODELS_HIGHWAY = [
    {
        "name": "Baseline PPO (027)",
        "short": "Baseline",
        "model_path": "experiments/027_ppo_highway_occgrid_rerun/results/model.zip",
        "model_cls": PPO,
        "env_factory": lambda cfg: make_v3_raw(cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_BASELINE),
    },
    {
        "name": "ViT+CVAE+Disc (033)",
        "short": "ViT+CVAE+Disc",
        "model_path": "experiments/033_adversarial_highway_350k/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_v3_dict(cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput-H10 (048)",
        "short": "ExpectedInput-H10",
        "model_path": "experiments/048_expected_horizon_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_v3_h10(cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput-H10+Aux (049)",
        "short": "ExpectedInput-H10+Aux",
        "model_path": "experiments/049_h10_aux_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_v3_h10(cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "H10 on v3 (050)",
        "short": "H10-v3",
        "model_path": "experiments/050_h10_v3_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_v3_h10(cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT-only on v3 (051)",
        "short": "ViT-only-v3",
        "model_path": "experiments/051_vit_only_v3_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: make_v3_dict_v3wrapped(cfg),
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
]


def archetype_of_collider(env) -> str | None:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    if not base.vehicle.crashed:
        return None
    for v in base.road.vehicles:
        if v is base.vehicle:
            continue
        if v.crashed and getattr(v, "is_adversarial", False):
            return getattr(v, "archetype", "generic")
    return None


def episode_archetype_counts(env) -> dict[str, int]:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    ego = base.vehicle
    out = {a: 0 for a in ARCHETYPES}
    out["generic"] = 0
    for v in base.road.vehicles:
        if v is ego:
            continue
        if not getattr(v, "is_adversarial", False):
            continue
        a = getattr(v, "archetype", "generic")
        out[a] = out.get(a, 0) + 1
    return out


def evaluate_model(model, env, seeds: list[int], label: str) -> list[dict]:
    results = []
    for seed in tqdm.tqdm(seeds, desc=f"  {label}"):
        obs, info = env.reset(seed=seed)
        arche_at_start = episode_archetype_counts(env)

        ep_reward = 0.0
        ep_len = 0
        done = False
        terminated = truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated

        crashed = info.get("crashed", terminated)
        collider_archetype = archetype_of_collider(env)
        crashed_with_adversarial = collider_archetype is not None
        crashed_with_nominal = bool(
            (terminated or crashed) and (collider_archetype is None)
        )

        results.append({
            "seed": seed,
            "reward": ep_reward,
            "length": ep_len,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": bool(terminated and not truncated),
            "crashed_with_adversarial": crashed_with_adversarial,
            "crashed_with_nominal": crashed_with_nominal,
            "collider_archetype": collider_archetype,
            "archetype_counts_at_start": arche_at_start,
        })
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    crashed = [r["crashed"] for r in results]

    # Per-archetype crash rate: fraction of episodes where ego collided with that archetype
    per_arch = {a: 0 for a in ARCHETYPES}
    per_arch["generic"] = 0
    for r in results:
        a = r["collider_archetype"]
        if a is not None:
            per_arch[a] = per_arch.get(a, 0) + 1
    per_arch_rate = {a: per_arch[a] / n for a in per_arch}

    # Mean exposure (avg count of each archetype per episode at episode start)
    exposure = {a: 0.0 for a in ARCHETYPES}
    exposure["generic"] = 0.0
    for r in results:
        for a, c in r["archetype_counts_at_start"].items():
            exposure[a] = exposure.get(a, 0.0) + c
    exposure = {a: exposure[a] / n for a in exposure}

    return {
        "n_episodes": n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "crash_rate": sum(crashed) / n,
        "crash_rate_adversarial": sum(r["crashed_with_adversarial"] for r in results) / n,
        "crash_rate_nominal": sum(r["crashed_with_nominal"] for r in results) / n,
        "per_archetype_crash_rate": per_arch_rate,
        "mean_exposure_per_episode": exposure,
        "early_termination_rate": sum(r["terminated"] for r in results) / n,
        "truncation_rate": sum(r["truncated"] for r in results) / n,
    }


def run_all() -> dict:
    print(f"\n{'=' * 60}")
    print(f"  V3 Fair Evaluation — HIGHWAY ({len(SEEDS)} episodes)")
    print(f"{'=' * 60}")
    out = {}
    for m in MODELS_HIGHWAY:
        if not Path(m["model_path"]).exists():
            print(f"\n[SKIP] {m['name']}: model not found at {m['model_path']}")
            continue
        print(f"\nLoading {m['name']}: {m['model_path']}")
        model = m["model_cls"].load(m["model_path"], device="cpu")
        env = m["env_factory"](m["env_config"])
        try:
            eps = evaluate_model(model, env, SEEDS, m["short"])
        finally:
            env.close()
        agg = aggregate(eps)
        print(
            f"  -> mean_reward={agg['mean_reward']:.3f}, "
            f"crash_rate={agg['crash_rate']*100:.1f}%, "
            f"per-arch={ {a: round(agg['per_archetype_crash_rate'][a]*100,1) for a in ARCHETYPES} }"
        )
        out[m["short"]] = {
            "name": m["name"],
            "model_path": m["model_path"],
            "per_episode": eps,
            "aggregate": agg,
        }
    return out


def fmt_pct(v):
    return f"{v * 100:.1f}%"


def write_report(models_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# V3 Fair Evaluation — Highway, per-archetype crash analysis\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Seeds:** {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} episodes/model)\n")
        f.write("**Env:** AdversarialHighwayV3Env (default mixture: tailgater 0.40, sudden_braker 0.20, lane_drifter 0.20, erratic_speed 0.20)\n\n")

        f.write("## Aggregate metrics\n\n")
        names = list(models_data.keys())
        f.write("| Metric | " + " | ".join(names) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(names)) + "|\n")

        rows = [
            ("Mean Reward", "mean_reward", False),
            ("Median Reward", "median_reward", False),
            ("Mean Episode Length", "mean_episode_length", False),
            ("Crash Rate (total)", "crash_rate", True),
            ("Crash Rate (adversarial collider)", "crash_rate_adversarial", True),
            ("Crash Rate (nominal collider)", "crash_rate_nominal", True),
            ("Truncation Rate (survival)", "truncation_rate", True),
        ]
        for label, key, pct in rows:
            vals = []
            for n in names:
                v = models_data[n]["aggregate"][key]
                vals.append(fmt_pct(v) if pct else f"{v:.3f}")
            f.write(f"| {label} | " + " | ".join(vals) + " |\n")

        f.write("\n## Per-archetype crash rate\n\n")
        f.write(
            "Each cell is the fraction of episodes where the ego's collision "
            "involved an adversary of that archetype. Lower is better.\n\n"
        )
        f.write("| Archetype | " + " | ".join(names) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(names)) + "|\n")
        for a in ARCHETYPES:
            vals = []
            for n in names:
                v = models_data[n]["aggregate"]["per_archetype_crash_rate"].get(a, 0.0)
                vals.append(fmt_pct(v))
            f.write(f"| {a} | " + " | ".join(vals) + " |\n")

        f.write("\n## Mean per-episode exposure (count at episode start)\n\n")
        # Use the first model's exposure as canonical (same for all on identical seeds)
        any_model = next(iter(models_data.values()))
        exposure = any_model["aggregate"]["mean_exposure_per_episode"]
        for a in ARCHETYPES:
            f.write(f"- {a}: {exposure.get(a, 0.0):.2f}\n")

        f.write("\n## Crash-rate ranking\n\n")
        ranked = sorted(
            models_data.items(),
            key=lambda x: x[1]["aggregate"]["crash_rate"],
        )
        for i, (name, mdata) in enumerate(ranked, 1):
            cr = mdata["aggregate"]["crash_rate"]
            mr = mdata["aggregate"]["mean_reward"]
            per_a = mdata["aggregate"]["per_archetype_crash_rate"]
            arch_str = ", ".join(f"{a}={fmt_pct(per_a.get(a,0))}" for a in ARCHETYPES)
            f.write(f"{i}. **{name}**: total {fmt_pct(cr)}, reward {mr:.2f} | {arch_str}\n")

    json_path = path.with_suffix(".json")
    raw = {
        name: {
            "aggregate": mdata["aggregate"],
            "per_episode": mdata["per_episode"],
        }
        for name, mdata in models_data.items()
    }
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)

    print(f"\nSummary: {path}")
    print(f"Raw data: {json_path}")


if __name__ == "__main__":
    data = run_all()
    out = Path("experiments/050_h10_v3_highway/results")
    write_report(data, out / "fair_eval_v3.md")
