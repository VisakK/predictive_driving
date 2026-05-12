"""500-episode eval on the v3i interaction env, with shared seeds across
all 4 models so head-to-head comparisons are over identical scenarios.

Adds two things on top of the original 500-seed eval:

  1. ``--pack-n-vehicles`` overrides ``env_config.pack_n_vehicles`` so the
     same model can be evaluated under different pack densities without
     editing yaml.

  2. Crash categorization. For each crashed episode we snapshot vehicle
     state immediately before the step that crashed the ego and classify
     the partner / scenario into one of:

         ran_into_wreck   — partner was already ``crashed`` before the step
         ran_into_static  — partner had pre-step speed < 2 m/s, not adversarial
         ego_initiated    — ego closing rate to partner exceeds partner's by ≥1 m/s
         partner_initiated — partner closing rate to ego exceeds ego's by ≥1 m/s
         mutual           — closing rates within 1 m/s of each other

     Plus partner adversary flag/archetype, closing rates, approach angle
     (ego frame, 0° = ahead), and partner speed at crash.

Only the first ``--n-videos`` episodes are saved as mp4.

Usage:
  python scripts/eval_v3i_500.py \
      --config experiments/065_baseline_v3i/config.yaml \
      --model experiments/065_baseline_v3i/results/model.zip \
      --env-id adversarial-highway-v3i-raw \
      --out-dir experiments/065_baseline_v3i/results/eval_v3i_500 \
      --n-episodes 500 \
      --n-videos 10 \
      --base-seed 20000 \
      [--pack-n-vehicles 5]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
import gymnasium as gym
import tqdm
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


def _spawn_archetype(env) -> str:
    """Inspect the underlying env to find the spawned adversary archetype
    (if any). Returns 'none' if the episode has no adversary."""
    base = env.unwrapped
    ego = base.vehicle
    for v in base.road.vehicles:
        if v is ego:
            continue
        if getattr(v, "is_adversarial", False):
            return getattr(v, "archetype", "unknown") or "unknown"
    return "none"


def _snapshot_vehicles(env) -> dict:
    """Capture ego + other-vehicle kinematic state for crash analysis.

    Returns a dict with keys 'ego' and 'others'. Vehicle ``id(v)`` is
    used to align pre/post snapshots."""
    base = env.unwrapped
    ego = base.vehicle
    ego_snap = {
        "pos": np.asarray(ego.position, dtype=np.float64).copy(),
        "vel": np.asarray(ego.velocity, dtype=np.float64).copy(),
        "speed": float(ego.speed),
        "heading": float(ego.heading),
        "crashed": bool(ego.crashed),
    }
    others = []
    for v in base.road.vehicles:
        if v is ego:
            continue
        others.append({
            "id": id(v),
            "pos": np.asarray(v.position, dtype=np.float64).copy(),
            "vel": np.asarray(v.velocity, dtype=np.float64).copy(),
            "speed": float(v.speed),
            "heading": float(v.heading),
            "crashed": bool(v.crashed),
            "is_adversarial": bool(getattr(v, "is_adversarial", False)),
            "archetype": getattr(v, "archetype", None),
        })
    return {"ego": ego_snap, "others": others}


def _categorize_crash(pre: dict, post: dict) -> dict:
    """Identify the crash partner and classify the collision.

    Strategy: at post-step, find the other vehicle whose centre is
    closest to the ego (a collision implies near-overlapping polygons,
    so the partner sits within ~6 m centre-to-centre). Then look up
    that vehicle's pre-step state by id and compute closing rates and
    approach angle in the ego's pre-step frame."""
    ego_pre = pre["ego"]
    ego_post = post["ego"]

    if not post["others"]:
        return {"category": "unknown", "reason": "no_other_vehicles"}

    closest_dist = float("inf")
    closest_idx = None
    for i, other_post in enumerate(post["others"]):
        d = float(np.linalg.norm(other_post["pos"] - ego_post["pos"]))
        if d < closest_dist:
            closest_dist = d
            closest_idx = i

    if closest_idx is None:
        return {"category": "unknown", "reason": "no_partner"}

    if closest_dist > 12.0:
        return {
            "category": "unknown",
            "reason": "no_close_partner",
            "partner_dist_post": closest_dist,
        }

    partner_post = post["others"][closest_idx]
    partner_id = partner_post["id"]
    partner_pre = next((o for o in pre["others"] if o["id"] == partner_id), None)
    if partner_pre is None:
        partner_pre = partner_post

    diff = partner_pre["pos"] - ego_pre["pos"]
    dist_pre = float(np.linalg.norm(diff))
    if dist_pre < 1e-6:
        unit = np.array([1.0, 0.0])
    else:
        unit = diff / dist_pre

    ego_closing = float(np.dot(ego_pre["vel"], unit))
    partner_closing = float(np.dot(partner_pre["vel"], -unit))

    ego_dir = np.array([np.cos(ego_pre["heading"]), np.sin(ego_pre["heading"])])
    ego_right = np.array([np.sin(ego_pre["heading"]), -np.cos(ego_pre["heading"])])
    fwd = float(np.dot(diff, ego_dir))
    lat = float(np.dot(diff, ego_right))
    approach_angle_deg = float(np.degrees(np.arctan2(lat, fwd)))

    partner_was_static = bool(partner_pre["speed"] < 2.0)
    partner_was_already_crashed = bool(partner_pre["crashed"])
    partner_is_adv = bool(partner_pre.get("is_adversarial", False))

    if partner_was_already_crashed:
        category = "ran_into_wreck"
    elif partner_was_static and not partner_is_adv:
        category = "ran_into_static"
    elif ego_closing > partner_closing + 1.0:
        category = "ego_initiated"
    elif partner_closing > ego_closing + 1.0:
        category = "partner_initiated"
    else:
        category = "mutual"

    return {
        "category": category,
        "partner_dist_pre": dist_pre,
        "partner_dist_post": closest_dist,
        "ego_speed_pre": float(ego_pre["speed"]),
        "partner_speed_pre": float(partner_pre["speed"]),
        "ego_closing": ego_closing,
        "partner_closing": partner_closing,
        "rel_speed": float(np.linalg.norm(
            np.asarray(ego_pre["vel"]) - np.asarray(partner_pre["vel"])
        )),
        "approach_angle_deg": approach_angle_deg,
        "partner_was_adversary": partner_is_adv,
        "partner_archetype": partner_pre.get("archetype") if partner_is_adv else None,
        "partner_was_static": partner_was_static,
        "partner_was_already_crashed": partner_was_already_crashed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--env-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-episodes", type=int, default=500)
    p.add_argument("--n-videos", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=20_000,
                   help="Shared base seed; episode i uses seed = base + i.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--pack-n-vehicles", type=int, default=None,
                   help="Override env_config.pack_n_vehicles for density sweeps.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algo = cfg.get("algo", "PPO")
    AlgoCls = ALGOS[algo]
    model = AlgoCls.load(args.model, device="cpu")

    env_config = cfg.get("env_config") or {}
    env_config = dict(env_config)
    if args.pack_n_vehicles is not None:
        env_config["pack_n_vehicles"] = int(args.pack_n_vehicles)
    env = gym.make(args.env_id, render_mode="rgb_array", config=env_config)

    rows = []
    pbar = tqdm.tqdm(range(args.n_episodes), desc="eval")
    for ep in pbar:
        seed = args.base_seed + ep
        obs, info = env.reset(seed=seed)
        record = ep < args.n_videos
        frames = []
        r_total = 0.0
        length = 0
        spawn_arch = _spawn_archetype(env)
        crash_details: dict | None = None
        done = False
        terminated = False
        truncated = False
        while not done:
            if record:
                frames.append(env.render())
            pre_snap = _snapshot_vehicles(env)
            action, _ = model.predict(obs, deterministic=True)
            obs, rw, terminated, truncated, info = env.step(action)
            r_total += float(rw)
            length += 1
            done = terminated or truncated
            if (
                terminated
                and crash_details is None
                and bool(info.get("crashed", False))
            ):
                post_snap = _snapshot_vehicles(env)
                crash_details = _categorize_crash(pre_snap, post_snap)
                crash_details["crash_step"] = length

        crashed = bool(info.get("crashed", False))
        crashed_with = info.get("crashed_with_archetype")

        if frames:
            video_arr = np.stack(frames, axis=0).astype(np.uint8)
            video_path = out_dir / f"ep{ep:03d}_seed{seed}_arch-{spawn_arch}.mp4"
            imageio.mimsave(video_path, video_arr, fps=args.fps,
                            codec="libx264", quality=8)
        rows.append({
            "episode": ep,
            "seed": seed,
            "reward": r_total,
            "length": length,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": crashed,
            "spawn_archetype": spawn_arch,
            "crashed_with_archetype": crashed_with,
            "crash": crash_details,
        })
    env.close()

    rewards = np.array([r["reward"] for r in rows])
    lengths = np.array([r["length"] for r in rows])
    crashes = np.array([r["crashed"] for r in rows])
    truncs = np.array([r["truncated"] for r in rows])

    per_arch: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "crash": 0, "reward_sum": 0.0, "length_sum": 0
    })
    for r in rows:
        a = r["spawn_archetype"]
        per_arch[a]["n"] += 1
        per_arch[a]["crash"] += int(r["crashed"])
        per_arch[a]["reward_sum"] += r["reward"]
        per_arch[a]["length_sum"] += r["length"]
    per_arch_out = {
        a: {
            "n": d["n"],
            "crash_rate": d["crash"] / d["n"] if d["n"] else 0.0,
            "mean_reward": d["reward_sum"] / d["n"] if d["n"] else 0.0,
            "mean_length": d["length_sum"] / d["n"] if d["n"] else 0.0,
        }
        for a, d in per_arch.items()
    }

    cat_acc: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "ego_speed_sum": 0.0,
        "partner_speed_sum": 0.0,
        "rel_speed_sum": 0.0,
        "approach_angle_sum": 0.0,
        "step_sum": 0,
        "partner_adv_count": 0,
    })
    for r in rows:
        if not r["crashed"] or r["crash"] is None:
            continue
        c = r["crash"]
        cat = c.get("category", "unknown")
        d = cat_acc[cat]
        d["n"] += 1
        if "ego_speed_pre" in c:
            d["ego_speed_sum"] += float(c["ego_speed_pre"])
            d["partner_speed_sum"] += float(c["partner_speed_pre"])
            d["rel_speed_sum"] += float(c["rel_speed"])
            d["approach_angle_sum"] += float(c["approach_angle_deg"])
        d["step_sum"] += int(c.get("crash_step", 0))
        if c.get("partner_was_adversary"):
            d["partner_adv_count"] += 1
    n_crashes = int(crashes.sum())
    per_cat_out = {
        cat: {
            "n": d["n"],
            "frac_of_crashes": d["n"] / n_crashes if n_crashes else 0.0,
            "frac_of_episodes": d["n"] / len(rows) if rows else 0.0,
            "mean_ego_speed_pre": d["ego_speed_sum"] / d["n"] if d["n"] else 0.0,
            "mean_partner_speed_pre": d["partner_speed_sum"] / d["n"] if d["n"] else 0.0,
            "mean_rel_speed": d["rel_speed_sum"] / d["n"] if d["n"] else 0.0,
            "mean_approach_angle_deg": d["approach_angle_sum"] / d["n"] if d["n"] else 0.0,
            "mean_crash_step": d["step_sum"] / d["n"] if d["n"] else 0.0,
            "adversary_partner_frac": d["partner_adv_count"] / d["n"] if d["n"] else 0.0,
        }
        for cat, d in cat_acc.items()
    }

    cat_by_arch: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r["crashed"] or r["crash"] is None:
            continue
        cat_by_arch[r["spawn_archetype"]][r["crash"].get("category", "unknown")] += 1
    cat_by_arch_out = {a: dict(d) for a, d in cat_by_arch.items()}

    summary = {
        "model": args.model,
        "env_id": args.env_id,
        "n_episodes": args.n_episodes,
        "base_seed": args.base_seed,
        "pack_n_vehicles": env_config.get("pack_n_vehicles"),
        "metrics": {
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "median_reward": float(np.median(rewards)),
            "mean_episode_length": float(lengths.mean()),
            "crash_rate": float(crashes.mean()),
            "truncation_rate": float(truncs.mean()),
        },
        "per_archetype": per_arch_out,
        "per_crash_category": per_cat_out,
        "category_by_archetype": cat_by_arch_out,
        "episodes": rows,
    }
    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.model} (pack={env_config.get('pack_n_vehicles')}) ===")
    print(f"  mean_reward = {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"  crash_rate = {crashes.mean():.1%}")
    print(f"  mean_length = {lengths.mean():.1f}")
    print(f"  truncation_rate = {truncs.mean():.1%}")
    print("  per archetype:")
    for a, d in sorted(per_arch_out.items()):
        print(f"    {a:14s} n={d['n']:3d}  crash={d['crash_rate']:.1%}  "
              f"reward={d['mean_reward']:.2f}  len={d['mean_length']:.1f}")
    print("  per crash category:")
    for cat, d in sorted(per_cat_out.items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {cat:18s} n={d['n']:3d}  frac_of_crashes={d['frac_of_crashes']:.1%}  "
              f"ego_v={d['mean_ego_speed_pre']:.1f}  partner_v={d['mean_partner_speed_pre']:.1f}  "
              f"angle={d['mean_approach_angle_deg']:.0f}°")
    print(f"  summary → {out_path}")


if __name__ == "__main__":
    main()
