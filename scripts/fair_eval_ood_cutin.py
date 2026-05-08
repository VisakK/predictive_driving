"""OOD Cut-In fair eval — held-out adversarial archetype.

Tests whether policies trained on the V3 archetype mixture (tailgater,
sudden_braker, lane_drifter, erratic_speed) generalize to a novel
deliberate cut-in adversary that none of them saw at training time.

Two evaluation environments:
  - V4-mixed: cut_in replaces erratic_speed in the mixture
    (0.40 tailgater, 0.20 sudden_braker, 0.20 lane_drifter, 0.20 cut_in)
  - V4-pure: only cut_in adversaries (1.0)

Models: 027 (baseline), 048, 051, 052, 054, 059. Same 500 seeds
(1000–1499) used by the V3 fair eval, so OOD vs ID comparison is
bit-comparable on a per-seed basis.

Per (model, env) combination:
  - records first N_VIDEO_EPISODES episodes as MP4 (rendered) and
    uploads each to W&B under videos/<model>/<env>/ep_<seed>.
  - all 500 episodes contribute to aggregate metrics, written to
    a single W&B Table for cross-model comparison.

Outputs:
  experiments/ood_cutin_eval/results/
    ood_cutin_full.json                    full per-episode data
    ood_cutin_report.md                    markdown comparison
    videos/<env>_<model>/ep-*.mp4          recorded rollouts
"""
from __future__ import annotations

import gc
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tqdm
import gymnasium as gym
import wandb

from stable_baselines3 import PPO

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401
import driving.adversarial_v3  # noqa: F401

from driving.adversarial_v3 import (
    AdversarialHighwayV3Env,
    DictObsWrapperV3,
    HorizonExpectedObservedDictObsWrapperV3,
    V4_MIXED_WEIGHTS,
    V4_PURE_CUTIN_WEIGHTS,
)
from driving.adversarial_ppo import AdversarialPPO


SEEDS = list(range(1000, 1500))                 # 500 episodes per (model, env)
N_VIDEO_EPISODES = 50                            # first N recorded as video
ARCHETYPES_V4 = [
    "tailgater", "sudden_braker", "lane_drifter", "erratic_speed", "cut_in",
]


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


def make_env_config(base, obs):
    return {**base, "observation": obs}


# ---- Env factories. Each takes (env_config, archetype_weights, render_mode). ----

def make_v4_raw(cfg, weights, render_mode=None):
    env = AdversarialHighwayV3Env(config=cfg, render_mode=render_mode)
    env.config["archetype_weights"] = dict(weights)
    return env


def make_v4_dict_v3wrapped(cfg, weights, render_mode=None):
    env = AdversarialHighwayV3Env(config=cfg, render_mode=render_mode)
    env.config["archetype_weights"] = dict(weights)
    return DictObsWrapperV3(env)


def make_v4_h10(cfg, weights, render_mode=None):
    env = AdversarialHighwayV3Env(config=cfg, render_mode=render_mode)
    env.config["archetype_weights"] = dict(weights)
    return HorizonExpectedObservedDictObsWrapperV3(env, horizon=10)


# ---- Models ----

MODELS = [
    {
        "name": "Baseline PPO (027)",
        "short": "027_Baseline",
        "model_path": "experiments/027_ppo_highway_occgrid_rerun/results/model.zip",
        "model_cls": PPO,
        "env_factory": make_v4_raw,
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_BASELINE),
    },
    {
        "name": "ExpectedInput-H10 (048)",
        "short": "048_H10",
        "model_path": "experiments/048_expected_horizon_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": make_v4_h10,
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "ViT-only-v3 (051)",
        "short": "051_ViTonly",
        "model_path": "experiments/051_vit_only_v3_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": make_v4_dict_v3wrapped,
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "AnomAttn-v3 (052)",
        "short": "052_AnomAttn",
        "model_path": "experiments/052_h10_attn_v3_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": make_v4_h10,
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "AnomAttn-Risk-GRU-v3 (054)",
        "short": "054_AnomAttn-Risk-GRU",
        "model_path": "experiments/054_h10_attn_risk_gru_v3_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": make_v4_h10,
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
    {
        "name": "AnomAttn-Risk-GRU-PBS-TruncBonus-v3 (059)",
        "short": "059_PBS_TruncBonus",
        "model_path": "experiments/059_h10_attn_risk_gru_pbs_truncbonus_v3_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": make_v4_h10,
        "env_config": make_env_config(HIGHWAY_BASE, OBS_CONFIG_5CH),
    },
]


ENVS = [
    {"short": "v4_mixed",
     "name": "V4-mixed (cut_in replaces erratic_speed)",
     "weights": V4_MIXED_WEIGHTS},
    {"short": "v4_pure",
     "name": "V4-pure (cut_in only)",
     "weights": V4_PURE_CUTIN_WEIGHTS},
]


# ---- Helpers ----

def archetype_of_collider(env):
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    if not base.vehicle.crashed:
        return None
    for v in base.road.vehicles:
        if v is base.vehicle:
            continue
        if v.crashed and getattr(v, "is_adversarial", False):
            return getattr(v, "archetype", "generic")
    return None


def episode_archetype_counts(env):
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    ego = base.vehicle
    out = {a: 0 for a in ARCHETYPES_V4}
    out["generic"] = 0
    for v in base.road.vehicles:
        if v is ego:
            continue
        if not getattr(v, "is_adversarial", False):
            continue
        a = getattr(v, "archetype", "generic")
        out[a] = out.get(a, 0) + 1
    return out


def evaluate_combo(model, model_short, env_short, env_factory, env_cfg, weights,
                   seeds, video_dir, n_video_episodes):
    """Run all `seeds` episodes, recording the first n_video_episodes as MP4."""
    # Build a single env with render_mode so RecordVideo can hook in.
    render_mode = "rgb_array" if n_video_episodes > 0 else None
    env = env_factory(env_cfg, weights, render_mode=render_mode)
    if n_video_episodes > 0:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda i: i < n_video_episodes,
            name_prefix=f"{env_short}_{model_short}",
        )

    results = []
    label = f"{env_short}/{model_short}"
    try:
        for seed in tqdm.tqdm(seeds, desc=label):
            obs, info = env.reset(seed=seed)
            arche_at_start = episode_archetype_counts(env)
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = terminated or truncated
            collider = archetype_of_collider(env)
            results.append({
                "seed": seed,
                "reward": ep_reward,
                "length": ep_len,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "crashed": bool(terminated and not truncated),
                "crashed_with_adversarial": collider is not None,
                "crashed_with_nominal": bool(
                    (terminated or info.get("crashed", terminated))
                    and (collider is None)
                ),
                "collider_archetype": collider,
                "archetype_counts_at_start": arche_at_start,
            })
    finally:
        env.close()
    return results


def aggregate(results):
    n = len(results)
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    crashed = [r["crashed"] for r in results]
    per_arch = {a: 0 for a in ARCHETYPES_V4}
    per_arch["generic"] = 0
    for r in results:
        a = r["collider_archetype"]
        if a is not None:
            per_arch[a] = per_arch.get(a, 0) + 1
    per_arch_rate = {a: per_arch[a] / n for a in per_arch}
    exposure = {a: 0.0 for a in ARCHETYPES_V4}
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
        "mean_episode_length": float(np.mean(lengths)),
        "crash_rate": sum(crashed) / n,
        "crash_rate_adversarial": sum(r["crashed_with_adversarial"] for r in results) / n,
        "crash_rate_nominal": sum(r["crashed_with_nominal"] for r in results) / n,
        "per_archetype_crash_rate": per_arch_rate,
        "mean_exposure_per_episode": exposure,
        "truncation_rate": sum(r["truncated"] for r in results) / n,
    }


def fmt_pct(v):
    return f"{v * 100:.1f}%"


def write_md_report(all_results, path):
    """Write a comparison MD across all (model, env) combos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# OOD Cut-In Fair Eval — Generalization to a Held-Out Archetype\n")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append(f"**Seeds:** {SEEDS[0]}–{SEEDS[-1]} ({len(SEEDS)} episodes per (model, env))\n")
    md.append("**Cut-in spec:** see `CutInIDMVehicle` in `src/driving/adversarial.py`. "
              "Triggers a deliberate target-lane change toward the ego when the ego is "
              "behind in an adjacent lane within range, closing, and cooldown elapsed. "
              "Held for COMMIT_STEPS=10 sim-steps (~2s), then COOLDOWN_STEPS=25 (~5s).\n")
    md.append("**Envs:**\n"
              "- `v4_mixed`: tailgater 0.40, sudden_braker 0.20, lane_drifter 0.20, **cut_in 0.20** "
              "(replaces erratic_speed). Tests partial OOD generalization.\n"
              "- `v4_pure`: cut_in 1.0. Maximum-stress OOD.\n")
    md.append(f"**Models:** {', '.join(m['short'] for m in MODELS)}\n")

    for env_cfg in ENVS:
        env_short = env_cfg["short"]
        md.append(f"\n## {env_cfg['name']}\n")

        # Per-env table
        rows = []
        for m in MODELS:
            key = f"{env_short}_{m['short']}"
            if key not in all_results:
                continue
            ag = all_results[key]["aggregate"]
            rows.append((m["short"], ag))

        if not rows:
            md.append("(no models evaluated)\n")
            continue

        md.append("\n| Model | Crash Rate | Crash (adv) | Crash (nom) | Mean Reward | Mean Ep Len | Survival |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, ag in rows:
            md.append(
                f"| {name} | {fmt_pct(ag['crash_rate'])} | "
                f"{fmt_pct(ag['crash_rate_adversarial'])} | "
                f"{fmt_pct(ag['crash_rate_nominal'])} | "
                f"{ag['mean_reward']:.2f} | "
                f"{ag['mean_episode_length']:.2f} | "
                f"{fmt_pct(ag['truncation_rate'])} |"
            )

        md.append("\n### Per-archetype crash rate\n")
        md.append("| Model | " + " | ".join(ARCHETYPES_V4) + " |")
        md.append("|---|" + "|".join(["---:"] * len(ARCHETYPES_V4)) + "|")
        for name, ag in rows:
            per_a = ag["per_archetype_crash_rate"]
            md.append(f"| {name} | " + " | ".join(fmt_pct(per_a.get(a, 0.0)) for a in ARCHETYPES_V4) + " |")

        md.append("\n### Crash-rate ranking\n")
        ranked = sorted(rows, key=lambda x: x[1]["crash_rate"])
        for i, (name, ag) in enumerate(ranked, 1):
            md.append(f"{i}. **{name}**: {fmt_pct(ag['crash_rate'])}, "
                      f"mean reward {ag['mean_reward']:.2f}")

    # Generalization gap section
    md.append("\n## Generalization gap (vs in-distribution V3 fair eval)\n")
    md.append("Where prior V3 ID results exist, this column shows OOD crash rate − ID crash rate. "
              "Positive ⇒ degradation OOD; near-zero ⇒ generalizes; negative ⇒ improves OOD (rare).\n")
    md.append("\n| Model | V3 ID | V4-mixed | gap (mixed) | V4-pure | gap (pure) |")
    md.append("|---|---:|---:|---:|---:|---:|")
    id_rates = _load_id_rates()
    for m in MODELS:
        s = m["short"]
        id_rate = id_rates.get(s)
        mixed = all_results.get(f"v4_mixed_{s}")
        pure = all_results.get(f"v4_pure_{s}")
        id_str = fmt_pct(id_rate) if id_rate is not None else "—"
        mixed_str = fmt_pct(mixed["aggregate"]["crash_rate"]) if mixed else "—"
        pure_str = fmt_pct(pure["aggregate"]["crash_rate"]) if pure else "—"
        gap_mixed = (
            fmt_pct(mixed["aggregate"]["crash_rate"] - id_rate)
            if (mixed and id_rate is not None) else "—"
        )
        gap_pure = (
            fmt_pct(pure["aggregate"]["crash_rate"] - id_rate)
            if (pure and id_rate is not None) else "—"
        )
        md.append(f"| {s} | {id_str} | {mixed_str} | {gap_mixed} | {pure_str} | {gap_pure} |")

    path.write_text("\n".join(md) + "\n")


def _load_id_rates():
    """Load in-distribution V3 crash rates per model from existing fair_eval files."""
    rates: dict[str, float] = {}
    sources = {
        "027_Baseline": ("experiments/050_h10_v3_highway/results/fair_eval_v3.json", "Baseline"),
        "048_H10": ("experiments/050_h10_v3_highway/results/fair_eval_v3.json", "ExpectedInput-H10"),
        "051_ViTonly": ("experiments/050_h10_v3_highway/results/fair_eval_v3.json", "ViT-only-v3"),
        "052_AnomAttn": ("experiments/052_h10_attn_v3_highway/results/fair_eval_focused.json", "AnomAttn-v3"),
        "054_AnomAttn-Risk-GRU": ("experiments/054_h10_attn_risk_gru_v3_highway/results/fair_eval_focused.json", "AnomAttn-Risk-GRU-v3"),
        "059_PBS_TruncBonus": ("experiments/059_h10_attn_risk_gru_pbs_truncbonus_v3_highway/results/fair_eval_focused.json", "AnomAttn-Risk-GRU-PBS-TruncBonus-v3"),
    }
    for short, (path, key) in sources.items():
        try:
            with open(path) as f:
                data = json.load(f)
            rates[short] = data[key]["aggregate"]["crash_rate"]
        except Exception:
            pass
    return rates


def main():
    out_dir = Path("experiments/ood_cutin_eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"ood_cutin_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_run = wandb.init(
        project="predictive_driving",
        name=run_name,
        config={
            "seed_range": [SEEDS[0], SEEDS[-1]],
            "n_seeds": len(SEEDS),
            "n_video_episodes": N_VIDEO_EPISODES,
            "envs": [{"short": e["short"], "weights": e["weights"]} for e in ENVS],
            "models": [{"short": m["short"], "name": m["name"]} for m in MODELS],
        },
    )

    all_results = {}

    for env_cfg in ENVS:
        env_short = env_cfg["short"]
        for m in MODELS:
            if not Path(m["model_path"]).exists():
                print(f"\n[SKIP] {m['name']}: model not found at {m['model_path']}")
                continue
            short_key = f"{env_short}_{m['short']}"
            print(f"\n=== {env_cfg['name']} | {m['name']} ===")

            video_dir = out_dir / "videos" / short_key
            video_dir.mkdir(parents=True, exist_ok=True)

            model = m["model_cls"].load(m["model_path"], device="cpu")
            eps = evaluate_combo(
                model=model,
                model_short=m["short"],
                env_short=env_short,
                env_factory=m["env_factory"],
                env_cfg=m["env_config"],
                weights=env_cfg["weights"],
                seeds=SEEDS,
                video_dir=video_dir,
                n_video_episodes=N_VIDEO_EPISODES,
            )
            agg = aggregate(eps)
            all_results[short_key] = {
                "env": env_short,
                "model": m["short"],
                "model_name": m["name"],
                "aggregate": agg,
                "per_episode": eps,
            }

            print(
                f"  -> mean_reward={agg['mean_reward']:.3f}, "
                f"crash_rate={agg['crash_rate']*100:.1f}%, "
                f"per-arch={ {a: round(agg['per_archetype_crash_rate'][a]*100,1) for a in ARCHETYPES_V4} }"
            )

            # Save partial JSON immediately so a crash mid-eval still gives us
            # what we collected so far.
            with open(out_dir / f"results_{short_key}.json", "w") as f:
                json.dump({short_key: all_results[short_key]}, f, indent=2, default=str)

            # Upload videos to wandb under a per-(model, env) media key so
            # each combo gets its own panel/tab in the workspace. Logging
            # the list as a single key keeps them grouped instead of
            # spawning 50 separate panels.
            videos = sorted(video_dir.glob("*.mp4"))[:N_VIDEO_EPISODES]
            wandb_videos = []
            for v in videos:
                try:
                    wandb_videos.append(
                        wandb.Video(str(v), caption=v.stem, format="mp4")
                    )
                except Exception as e:
                    print(f"  [warn] failed to load {v.name}: {e}")
            if wandb_videos:
                key = f"videos/{m['short']}/{env_short}"
                try:
                    wandb_run.log({key: wandb_videos})
                except Exception as e:
                    print(f"  [warn] failed to upload videos for {short_key}: {e}")

            # Per-(model, env) scalar metrics
            wandb_run.log({
                f"crash_rate/{env_short}/{m['short']}": agg["crash_rate"],
                f"crash_adv/{env_short}/{m['short']}": agg["crash_rate_adversarial"],
                f"crash_nom/{env_short}/{m['short']}": agg["crash_rate_nominal"],
                f"mean_reward/{env_short}/{m['short']}": agg["mean_reward"],
                f"mean_ep_len/{env_short}/{m['short']}": agg["mean_episode_length"],
                f"survival/{env_short}/{m['short']}": agg["truncation_rate"],
            })

            del model, eps
            gc.collect()

    # Build summary tables: one per env, plus a combined gap table.
    columns = ["env", "model", "crash_rate", "crash_adv", "crash_nominal",
               "mean_reward", "mean_ep_len", "survival",
               "tailgater", "sudden_braker", "lane_drifter", "erratic_speed", "cut_in"]
    summary_table = wandb.Table(columns=columns)
    for short_key, data in all_results.items():
        ag = data["aggregate"]
        per_a = ag["per_archetype_crash_rate"]
        summary_table.add_data(
            data["env"], data["model"],
            ag["crash_rate"], ag["crash_rate_adversarial"], ag["crash_rate_nominal"],
            ag["mean_reward"], ag["mean_episode_length"], ag["truncation_rate"],
            per_a.get("tailgater", 0), per_a.get("sudden_braker", 0),
            per_a.get("lane_drifter", 0), per_a.get("erratic_speed", 0),
            per_a.get("cut_in", 0),
        )
    wandb_run.log({"ood_eval_summary": summary_table})

    # Generalization-gap table
    id_rates = _load_id_rates()
    gap_columns = ["model", "id_v3", "ood_v4_mixed", "gap_mixed",
                   "ood_v4_pure", "gap_pure"]
    gap_table = wandb.Table(columns=gap_columns)
    for m in MODELS:
        s = m["short"]
        id_rate = id_rates.get(s)
        mixed = all_results.get(f"v4_mixed_{s}")
        pure = all_results.get(f"v4_pure_{s}")
        gap_table.add_data(
            s,
            id_rate if id_rate is not None else float("nan"),
            mixed["aggregate"]["crash_rate"] if mixed else float("nan"),
            (mixed["aggregate"]["crash_rate"] - id_rate) if (mixed and id_rate is not None) else float("nan"),
            pure["aggregate"]["crash_rate"] if pure else float("nan"),
            (pure["aggregate"]["crash_rate"] - id_rate) if (pure and id_rate is not None) else float("nan"),
        )
    wandb_run.log({"generalization_gap": gap_table})

    # Persist full results
    with open(out_dir / "ood_cutin_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    write_md_report(all_results, out_dir / "ood_cutin_report.md")

    print(f"\nSummary: {out_dir / 'ood_cutin_report.md'}")
    print(f"Raw data: {out_dir / 'ood_cutin_full.json'}")
    print(f"WandB run: {wandb_run.url}")
    wandb_run.finish()


if __name__ == "__main__":
    main()
