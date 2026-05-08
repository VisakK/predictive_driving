"""Anomaly-zero ablation for experiment 055 (054 + CVAE/Discriminator aux losses).

Same protocol as anom_zero_ablation_054.py: load the trained 055
model, evaluate over the same 500 seeds, but zero `agent_anomaly`
in every observation.

Comparison context:
- 053 ablation Δ: +8.0 pp (H10 strongly load-bearing)
- 054 ablation Δ: +5.2 pp (GRU absorbed some H10's role)
- 055 ablation Δ: ?  -- does adding CVAE/Disc aux losses change how
  load-bearing the H10 channel is in 054's architecture?

Two outcomes are equally informative:
- If 055 Δ ≈ 054 Δ (≈5 pp), aux losses didn't shift which input the
  policy reads from; they only shaped the representation.
- If 055 Δ < 054 Δ, aux supervision pushed the policy to lean more on
  the patch-derived signal (which the aux losses now shape) rather than
  the H10 channel.
- If 055 Δ > 054 Δ, aux supervision sharpened the H10 pathway too
  (e.g. by tightening patches the cross-attention reads, which makes
  the per-query temperature scaling more decisive).
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
MODEL_PATH = "experiments/055_h10_attn_risk_gru_aux_v3_highway/results/model.zip"
SHORT = "AnomAttn-Risk-GRU-Aux-v3-zeroed"
NAME = "AnomalyAttention+RiskTemp+GRU+Aux 055 with agent_anomaly zeroed"

HIGHWAY_BASE = {
    "simulation_frequency": 5, "policy_frequency": 1, "duration": 80,
    "normalize_reward": False, "vehicles_count": 100,
}
OBS_CONFIG_5CH = {
    "type": "OccupancyGrid", "vehicles_count": 15,
    "features": ["presence", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {"vx": [-20, 20], "vy": [-20, 20]},
    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]], "grid_step": [5, 5],
    "absolute": False,
}


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


def zero_anomaly(obs):
    if "agent_anomaly" in obs:
        obs["agent_anomaly"] = np.zeros_like(obs["agent_anomaly"])
    return obs


def evaluate(model, env, seeds):
    results = []
    for seed in tqdm.tqdm(seeds, desc=f"  {SHORT}"):
        obs, info = env.reset(seed=seed)
        obs = zero_anomaly(obs)
        arche_at_start = episode_archetype_counts(env)
        ep_reward, ep_len = 0.0, 0
        terminated = truncated = False
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = zero_anomaly(obs)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        collider_archetype = archetype_of_collider(env)
        results.append({
            "seed": seed, "reward": ep_reward, "length": ep_len,
            "terminated": bool(terminated), "truncated": bool(truncated),
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
    per_arch = {a: 0 for a in ARCHETYPES}; per_arch["generic"] = 0
    for r in results:
        a = r["collider_archetype"]
        if a is not None:
            per_arch[a] = per_arch.get(a, 0) + 1
    per_arch_rate = {a: per_arch[a] / n for a in per_arch}
    return {
        "n_episodes": n,
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "crash_rate": sum(crashed) / n,
        "crash_rate_adversarial": sum(r["crashed_with_adversarial"] for r in results) / n,
        "crash_rate_nominal": sum(r["crashed_with_nominal"] for r in results) / n,
        "per_archetype_crash_rate": per_arch_rate,
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
    agg_zero = aggregate(eps)
    print(
        f"  -> mean_reward={agg_zero['mean_reward']:.3f}, "
        f"crash_rate={agg_zero['crash_rate']*100:.1f}%, "
        f"per-arch={ {a: round(agg_zero['per_archetype_crash_rate'][a]*100,1) for a in ARCHETYPES} }"
    )

    out_dir = Path("experiments/055_h10_attn_risk_gru_aux_v3_highway/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "anom_zero_ablation.json", "w") as f:
        json.dump(
            {SHORT: {"name": NAME, "model_path": MODEL_PATH,
                     "aggregate": agg_zero, "per_episode": eps}},
            f, indent=2, default=str,
        )

    # Build comparison report against original 055, plus 054 and 053 priors.
    focused_055 = json.load(open(out_dir / "fair_eval_focused.json"))
    focused_054 = json.load(open(
        "experiments/054_h10_attn_risk_gru_v3_highway/results/fair_eval_focused.json"
    ))
    abl_054 = json.load(open(
        "experiments/054_h10_attn_risk_gru_v3_highway/results/anom_zero_ablation.json"
    ))
    focused_053 = json.load(open(
        "experiments/053_h10_attn_risk_v3_highway/results/fair_eval_focused.json"
    ))
    abl_053 = json.load(open(
        "experiments/053_h10_attn_risk_v3_highway/results/anom_zero_ablation.json"
    ))

    rows = [
        ("AnomAttn-Risk-GRU-Aux-v3 (orig)",
         focused_055["AnomAttn-Risk-GRU-Aux-v3"]["aggregate"]),
        ("AnomAttn-Risk-GRU-Aux-v3 (zeroed)", agg_zero),
        ("AnomAttn-Risk-GRU-v3 (054 orig)",
         focused_054["AnomAttn-Risk-GRU-v3"]["aggregate"]),
        ("AnomAttn-Risk-GRU-v3 (054 zeroed)",
         abl_054["AnomAttn-Risk-GRU-v3-zeroed"]["aggregate"]),
        ("AnomAttn-Risk-v3 (053 orig)",
         focused_053["AnomAttn-Risk-v3"]["aggregate"]),
        ("AnomAttn-Risk-v3 (053 zeroed)",
         abl_053["AnomAttn-Risk-v3-zeroed"]["aggregate"]),
    ]
    md = []
    md.append("# 055 — Anomaly-Zero Ablation\n")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append(f"**Seeds:** {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} episodes), "
              f"AdversarialHighwayV3 default mixture.\n")
    md.append("`AnomAttn-Risk-GRU-Aux-v3 (zeroed)` = the 055 model evaluated "
              "with `agent_anomaly` set to zero in every observation. The "
              "comparison vs. 053/054 deltas tells us whether adding the "
              "CVAE+Discriminator aux losses on top of 054 changed how "
              "load-bearing the H10 anomaly is for the policy.\n")

    md.append("\n## Aggregate metrics\n")
    md.append("| Metric | " + " | ".join(n for n, _ in rows) + " |")
    md.append("|---|" + "|".join(["---"] * len(rows)) + "|")
    metric_rows = [
        ("Mean Reward", "mean_reward", False),
        ("Median Reward", "median_reward", False),
        ("Mean Episode Length", "mean_episode_length", False),
        ("Crash Rate (total)", "crash_rate", True),
        ("Crash Rate (adversarial)", "crash_rate_adversarial", True),
        ("Crash Rate (nominal)", "crash_rate_nominal", True),
        ("Truncation (survival)", "truncation_rate", True),
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

    md.append("\n## Delta (zeroed - original)\n")
    o055 = focused_055["AnomAttn-Risk-GRU-Aux-v3"]["aggregate"]
    z055 = agg_zero
    o054 = focused_054["AnomAttn-Risk-GRU-v3"]["aggregate"]
    z054 = abl_054["AnomAttn-Risk-GRU-v3-zeroed"]["aggregate"]
    o053 = focused_053["AnomAttn-Risk-v3"]["aggregate"]
    z053 = abl_053["AnomAttn-Risk-v3-zeroed"]["aggregate"]
    md.append("| Metric | 055 Δ | 054 Δ | 053 Δ |")
    md.append("|---|---|---|---|")
    md.append(
        f"| Crash rate | {(z055['crash_rate']-o055['crash_rate'])*100:+.1f} pp | "
        f"{(z054['crash_rate']-o054['crash_rate'])*100:+.1f} pp | "
        f"{(z053['crash_rate']-o053['crash_rate'])*100:+.1f} pp |"
    )
    md.append(
        f"| Crash (adversarial) | "
        f"{(z055['crash_rate_adversarial']-o055['crash_rate_adversarial'])*100:+.1f} pp | "
        f"{(z054['crash_rate_adversarial']-o054['crash_rate_adversarial'])*100:+.1f} pp | "
        f"{(z053['crash_rate_adversarial']-o053['crash_rate_adversarial'])*100:+.1f} pp |"
    )
    md.append(
        f"| Crash (nominal) | "
        f"{(z055['crash_rate_nominal']-o055['crash_rate_nominal'])*100:+.1f} pp | "
        f"{(z054['crash_rate_nominal']-o054['crash_rate_nominal'])*100:+.1f} pp | "
        f"{(z053['crash_rate_nominal']-o053['crash_rate_nominal'])*100:+.1f} pp |"
    )
    md.append(
        f"| Mean reward | {z055['mean_reward']-o055['mean_reward']:+.2f} | "
        f"{z054['mean_reward']-o054['mean_reward']:+.2f} | "
        f"{z053['mean_reward']-o053['mean_reward']:+.2f} |"
    )
    md.append(
        f"| Survival | "
        f"{(z055['truncation_rate']-o055['truncation_rate'])*100:+.1f} pp | "
        f"{(z054['truncation_rate']-o054['truncation_rate'])*100:+.1f} pp | "
        f"{(z053['truncation_rate']-o053['truncation_rate'])*100:+.1f} pp |"
    )

    md.append("\n## Per-archetype delta (zeroed - original)\n")
    md.append("| Archetype | 055 orig | 055 zeroed | 055 Δ | 054 Δ | 053 Δ |")
    md.append("|---|---|---|---|---|---|")
    for a in ARCHETYPES:
        ov55 = o055["per_archetype_crash_rate"].get(a, 0.0)
        zv55 = z055["per_archetype_crash_rate"].get(a, 0.0)
        ov54 = o054["per_archetype_crash_rate"].get(a, 0.0)
        zv54 = z054["per_archetype_crash_rate"].get(a, 0.0)
        ov53 = o053["per_archetype_crash_rate"].get(a, 0.0)
        zv53 = z053["per_archetype_crash_rate"].get(a, 0.0)
        md.append(
            f"| {a} | {fmt_pct(ov55)} | {fmt_pct(zv55)} | "
            f"{(zv55-ov55)*100:+.1f} | {(zv54-ov54)*100:+.1f} | "
            f"{(zv53-ov53)*100:+.1f} |"
        )

    md_path = out_dir / "anom_zero_ablation.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nSummary: {md_path}")
    print(f"Raw data: {out_dir / 'anom_zero_ablation.json'}")


if __name__ == "__main__":
    main()
