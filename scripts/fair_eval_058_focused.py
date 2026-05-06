"""Focused fair eval for experiment 058 (057 + Tier 1 reward-shaping fixes).

058 = 054 architecture + reward shaping with three fixes vs. 057:
  - Penalty masked on terminal transitions (no free pass for crashes)
  - Per-env score is top-3 mean of presence-masked risk (concentrates on
    threats without the single-point spikiness of max())
  - anomaly_reward_weight reduced to 0.05 (10x lower than 057's 0.5)

Same 500-seed protocol, bit-comparable to 052/053/054/055/056/057.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tqdm

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401
import driving.adversarial_v3  # noqa: F401

from driving.adversarial_v3 import (
    AdversarialHighwayV3Env,
    HorizonExpectedObservedDictObsWrapperV3,
)
from driving.adversarial_ppo import AdversarialPPO


SEEDS = list(range(1000, 1500))
ARCHETYPES = ["tailgater", "sudden_braker", "lane_drifter", "erratic_speed"]

HIGHWAY_BASE = {
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "duration": 80,
    "normalize_reward": False,
    "vehicles_count": 100,
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

MODEL_PATH = "experiments/058_h10_attn_risk_gru_rewardshape_fixed_v3_highway/results/model.zip"
SHORT = "AnomAttn-Risk-GRU-RewShape-Fixed-v3"
NAME = "AnomalyAttention + Risk-Temp + GRU + reward shaping (anom_w=0.05, terminal-masked, max-risk) on v3 (058)"


def make_env():
    cfg = {**HIGHWAY_BASE, "observation": OBS_CONFIG_5CH}
    return HorizonExpectedObservedDictObsWrapperV3(
        AdversarialHighwayV3Env(config=cfg), horizon=10
    )


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
    out = {a: 0 for a in ARCHETYPES}
    out["generic"] = 0
    for v in base.road.vehicles:
        if v is ego or not getattr(v, "is_adversarial", False):
            continue
        a = getattr(v, "archetype", "generic")
        out[a] = out.get(a, 0) + 1
    return out


def evaluate(model, env, seeds):
    results = []
    for seed in tqdm.tqdm(seeds, desc=f"  {SHORT}"):
        obs, info = env.reset(seed=seed)
        arche_at_start = episode_archetype_counts(env)
        ep_reward, ep_len = 0.0, 0
        terminated = truncated = False
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        collider_archetype = archetype_of_collider(env)
        results.append({
            "seed": seed,
            "reward": ep_reward,
            "length": ep_len,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": bool(terminated and not truncated),
            "crashed_with_adversarial": collider_archetype is not None,
            "crashed_with_nominal": bool(
                (terminated or info.get("crashed", terminated))
                and (collider_archetype is None)
            ),
            "collider_archetype": collider_archetype,
            "archetype_counts_at_start": arche_at_start,
        })
    return results


def aggregate(results):
    n = len(results)
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    crashed = [r["crashed"] for r in results]
    per_arch = {a: 0 for a in ARCHETYPES}
    per_arch["generic"] = 0
    for r in results:
        a = r["collider_archetype"]
        if a is not None:
            per_arch[a] = per_arch.get(a, 0) + 1
    per_arch_rate = {a: per_arch[a] / n for a in per_arch}
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
        "mean_episode_length": float(np.mean(lengths)),
        "crash_rate": sum(crashed) / n,
        "crash_rate_adversarial": sum(r["crashed_with_adversarial"] for r in results) / n,
        "crash_rate_nominal": sum(r["crashed_with_nominal"] for r in results) / n,
        "per_archetype_crash_rate": per_arch_rate,
        "mean_exposure_per_episode": exposure,
        "truncation_rate": sum(r["truncated"] for r in results) / n,
    }


def fmt_pct(v): return f"{v * 100:.1f}%"


def main():
    print(f"Loading {NAME}: {MODEL_PATH}")
    model = AdversarialPPO.load(MODEL_PATH, device="cpu")
    env = make_env()
    try:
        eps = evaluate(model, env, SEEDS)
    finally:
        env.close()
    agg = aggregate(eps)
    print(
        f"  -> mean_reward={agg['mean_reward']:.3f}, "
        f"crash_rate={agg['crash_rate']*100:.1f}%, "
        f"per-arch={ {a: round(agg['per_archetype_crash_rate'][a]*100,1) for a in ARCHETYPES} }"
    )

    out_dir = Path("experiments/058_h10_attn_risk_gru_rewardshape_fixed_v3_highway/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fair_eval_focused.json", "w") as f:
        json.dump(
            {SHORT: {"name": NAME, "model_path": MODEL_PATH,
                     "aggregate": agg, "per_episode": eps}},
            f, indent=2, default=str,
        )

    prior = json.load(open("experiments/050_h10_v3_highway/results/fair_eval_v3.json"))
    prev_057 = json.load(open(
        "experiments/057_h10_attn_risk_gru_rewardshape_v3_highway/results/fair_eval_focused.json"
    ))
    prev_056 = json.load(open(
        "experiments/056_h10_attn_refined_v3_highway/results/fair_eval_focused.json"
    ))
    prev_055 = json.load(open(
        "experiments/055_h10_attn_risk_gru_aux_v3_highway/results/fair_eval_focused.json"
    ))
    prev_054 = json.load(open(
        "experiments/054_h10_attn_risk_gru_v3_highway/results/fair_eval_focused.json"
    ))
    prev_053 = json.load(open(
        "experiments/053_h10_attn_risk_v3_highway/results/fair_eval_focused.json"
    ))
    prev_052 = json.load(open(
        "experiments/052_h10_attn_v3_highway/results/fair_eval_focused.json"
    ))
    rows = [
        (SHORT, agg),
        ("AnomAttn-Risk-GRU-RewShape-v3 (057)",
         prev_057["AnomAttn-Risk-GRU-RewShape-v3"]["aggregate"]),
        ("AnomAttn-Refined-v3 (056)",
         prev_056["AnomAttn-Refined-v3"]["aggregate"]),
        ("AnomAttn-Risk-GRU-Aux-v3 (055)",
         prev_055["AnomAttn-Risk-GRU-Aux-v3"]["aggregate"]),
        ("AnomAttn-Risk-GRU-v3 (054)",
         prev_054["AnomAttn-Risk-GRU-v3"]["aggregate"]),
        ("AnomAttn-Risk-v3 (053)",
         prev_053["AnomAttn-Risk-v3"]["aggregate"]),
        ("AnomAttn-v3 (052)",
         prev_052["AnomAttn-v3"]["aggregate"]),
        ("ExpectedInput-H10 (048)", prior["ExpectedInput-H10"]["aggregate"]),
        ("ViT-only-v3 (051)", prior["ViT-only-v3"]["aggregate"]),
        ("Baseline (027)", prior["Baseline"]["aggregate"]),
    ]

    md = []
    md.append(f"# 058 — Anomaly Attention + Risk-Temp + GRU + Fixed Reward Shaping vs. priors\n")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append(f"**Seeds:** {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} episodes/model), "
              f"AdversarialHighwayV3 default mixture.\n")
    md.append("058 = 054 architecture + Tier 1 reward-shaping fixes:\n"
              "- `anomaly_reward_weight=0.05` (10x lower than 057's 0.5)\n"
              "- Penalty masked on terminal transitions (no early-termination free pass)\n"
              "- Per-env score is top-3 mean of presence-masked risk (concentrates on "
              "threats without single-agent spikiness; an earlier max-based version "
              "NaN'd at iter 21 from advantage-variance blowup)\n"
              "Tests whether the 057 regression was due to the implementation defects "
              "(early-termination loophole, threat dilution, oversized weight) rather "
              "than reward shaping being fundamentally redundant with the 054 architecture.\n")

    md.append("\n## Aggregate metrics\n")
    md.append("| Metric | " + " | ".join(n for n, _ in rows) + " |")
    md.append("|---|" + "|".join(["---"] * len(rows)) + "|")
    metric_rows = [
        ("Mean Reward", "mean_reward", False),
        ("Median Reward", "median_reward", False),
        ("Mean Episode Length", "mean_episode_length", False),
        ("Crash Rate (total)", "crash_rate", True),
        ("Crash Rate (adversarial collider)", "crash_rate_adversarial", True),
        ("Crash Rate (nominal collider)", "crash_rate_nominal", True),
        ("Truncation Rate (survival)", "truncation_rate", True),
    ]
    for label, key, pct in metric_rows:
        vals = []
        for _, ag in rows:
            v = ag[key]
            vals.append(fmt_pct(v) if pct else f"{v:.3f}")
        md.append(f"| {label} | " + " | ".join(vals) + " |")

    md.append("\n## Per-archetype crash rate\n")
    md.append("| Archetype | " + " | ".join(n for n, _ in rows) + " |")
    md.append("|---|" + "|".join(["---"] * len(rows)) + "|")
    for a in ARCHETYPES:
        vals = [fmt_pct(ag["per_archetype_crash_rate"].get(a, 0.0)) for _, ag in rows]
        md.append(f"| {a} | " + " | ".join(vals) + " |")

    md.append("\n## Crash-rate ranking\n")
    ranked = sorted(rows, key=lambda x: x[1]["crash_rate"])
    for i, (n, ag) in enumerate(ranked, 1):
        per_a = ag["per_archetype_crash_rate"]
        arch_str = ", ".join(f"{a}={fmt_pct(per_a.get(a,0))}" for a in ARCHETYPES)
        md.append(f"{i}. **{n}**: total {fmt_pct(ag['crash_rate'])}, "
                  f"reward {ag['mean_reward']:.2f} | {arch_str}")

    md_path = out_dir / "fair_eval_focused.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nSummary: {md_path}")
    print(f"Raw data: {out_dir / 'fair_eval_focused.json'}")


if __name__ == "__main__":
    main()
