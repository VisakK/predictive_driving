"""Focused highway fair eval: 4 models, 300 matched seeds.

Models compared:
  - Baseline PPO (027_ppo_highway_occgrid_rerun)
  - ViT+CVAE+Disc (033_adversarial_highway_350k)
  - ExpectedInput-H10 (048_expected_horizon_highway)
  - H10 + CVAE/Disc aux (049_h10_aux_highway)

All evaluated on the v2 (proximity-guaranteed) highway environment with seeds
1000-1299. Reuses the helpers in scripts/fair_eval.py.
"""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

import highway_env  # noqa: F401
import driving.envs  # noqa: F401
import driving.adversarial  # noqa: F401

from driving.adversarial_ppo import AdversarialPPO

from scripts import fair_eval as fe


SEEDS = list(range(1000, 1300))


MODELS_HIGHWAY = [
    {
        "name": "Baseline PPO (027)",
        "short": "Baseline",
        "model_path": "experiments/027_ppo_highway_occgrid_rerun/results/model.zip",
        "model_cls": PPO,
        "env_factory": lambda cfg: fe.make_raw_env(fe.HIGHWAY_EVAL_ENV, cfg),
        "env_config": fe.make_env_config(fe.HIGHWAY_BASE, fe.OBS_CONFIG_BASELINE),
    },
    {
        "name": "ViT+CVAE+Disc (033)",
        "short": "ViT+CVAE+Disc",
        "model_path": "experiments/033_adversarial_highway_350k/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: fe.make_dict_env(fe.HIGHWAY_EVAL_ENV, cfg),
        "env_config": fe.make_env_config(fe.HIGHWAY_BASE, fe.OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput-H10 (048)",
        "short": "ExpectedInput-H10",
        "model_path": "experiments/048_expected_horizon_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: fe.make_expected_h10_env(fe.HIGHWAY_EVAL_ENV, cfg),
        "env_config": fe.make_env_config(fe.HIGHWAY_BASE, fe.OBS_CONFIG_5CH),
    },
    {
        "name": "ExpectedInput-H10 + Aux (049)",
        "short": "ExpectedInput-H10+Aux",
        "model_path": "experiments/049_h10_aux_highway/results/model.zip",
        "model_cls": AdversarialPPO,
        "env_factory": lambda cfg: fe.make_expected_h10_env(fe.HIGHWAY_EVAL_ENV, cfg),
        "env_config": fe.make_env_config(fe.HIGHWAY_BASE, fe.OBS_CONFIG_5CH),
    },
]


def run_focused() -> dict:
    fe.SEEDS = SEEDS  # patch so evaluate_model uses the wider seed range
    return fe.run_scenario("highway", MODELS_HIGHWAY)


def write_focused_summary(hw_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Focused Highway Fair Evaluation (4 models, 300 seeds)\n\n")
        f.write(f"Seeds: {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} episodes/model)\n")
        f.write("Env: AdversarialHighwayV2Env (proximity-guaranteed)\n\n")
        f.write("## Models\n\n")
        for m in MODELS_HIGHWAY:
            f.write(f"- **{m['short']}** — {m['name']} — `{m['model_path']}`\n")
        f.write("\n## Results\n\n")
        f.write(fe.multi_model_table(hw_data["models"]))
        f.write("\n\n## Pairwise Crash Analysis\n\n")
        f.write(fe.pairwise_crash_analysis(hw_data["models"]))
        f.write("\n\n## Crash-rate ranking\n\n")
        ranked = sorted(
            hw_data["models"].items(),
            key=lambda x: x[1]["aggregate"]["crash_rate"],
        )
        for i, (name, mdata) in enumerate(ranked, 1):
            cr = mdata["aggregate"]["crash_rate"]
            mr = mdata["aggregate"]["mean_reward"]
            f.write(f"{i}. **{name}**: {cr*100:.1f}% crash rate, {mr:.2f} mean reward\n")

    json_path = path.with_suffix(".json")
    import json
    raw = {
        name: {
            "aggregate": mdata["aggregate"],
            "per_episode": mdata["per_episode"],
        }
        for name, mdata in hw_data["models"].items()
    }
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Summary: {path}")
    print(f"Raw data: {json_path}")


if __name__ == "__main__":
    hw = run_focused()
    out = Path("experiments/049_h10_aux_highway/results")
    write_focused_summary(hw, out / "fair_eval_focused.md")
