"""Map offline wandb runs to experiment summary.md files and inject wandb URLs."""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTITY = "visakii"
PROJECT = "predictive_driving"

RUN_NAME_TO_EXP = {
    "ppo_highway_v0_kin_b2a41": "001_ppo_baseline",
    "ppo_merge_v0_kin_b2a41": "002_ppo_merge",
    "ppo_roundabout_v0_kin_b2a41": "003_ppo_roundabout",
    "ppo_intersection_v0_kin_b2a41": "004_ppo_intersection",
    "ppo_two_way_v0_kin_b2a41": "005_ppo_two_way",
    "ppo_u_turn_v0_kin_b2a41": "006_ppo_u_turn",
    "ppo_exit_v0_kin_b2a41": "007_ppo_exit",
    "ppo_racetrack_v0_kin_b2a41": "008_ppo_racetrack",
    "ppo_highway_v0_occ_b2a41": "009_ppo_highway_occgrid",
    "ppo_merge_v0_occ_b2a41": "010_ppo_merge_occgrid",
    "ppo_roundabout_v0_occ_b2a41": "011_ppo_roundabout_occgrid",
    "ppo_intersection_v0_occ_b2a41": "012_ppo_intersection_occgrid",
    "ppo_two_way_v0_occ_b2a41": "013_ppo_two_way_occgrid",
    "ppo_u_turn_v0_occ_b2a41": "014_ppo_u_turn_occgrid",
    "ppo_exit_v0_occ_b2a41": "015_ppo_exit_occgrid",
    "ppo_racetrack_v0_occ_b2a41": "016_ppo_racetrack_occgrid",
}


def extract_run_info(offline_dir: Path) -> tuple[str, str] | None:
    meta_path = offline_dir / "files" / "wandb-metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    args = meta.get("args", [])
    run_name = None
    for i, a in enumerate(args):
        if a == "--run_name" and i + 1 < len(args):
            run_name = args[i + 1]
            break
    if run_name is None:
        return None
    run_id = offline_dir.name.rsplit("-", 1)[-1]
    return run_name, run_id


def main():
    name_to_id = {}
    for d in sorted((REPO / "wandb").glob("offline-run-*")):
        info = extract_run_info(d)
        if info is None:
            continue
        name, rid = info
        name_to_id[name] = rid

    missing = [n for n in RUN_NAME_TO_EXP if n not in name_to_id]
    if missing:
        print(f"Missing offline runs for: {missing}")

    for run_name, exp_name in RUN_NAME_TO_EXP.items():
        run_id = name_to_id.get(run_name)
        if run_id is None:
            print(f"[SKIP] {run_name}: no offline run found")
            continue
        url = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}"
        summary_path = REPO / "experiments" / exp_name / "results" / "summary.md"
        if not summary_path.exists():
            print(f"[SKIP] {summary_path}: missing")
            continue
        text = summary_path.read_text()
        line = f"- **wandb:** {url}"
        if "**wandb:**" in text:
            text = re.sub(r"^- \*\*wandb:\*\* .*$", line, text, count=1, flags=re.MULTILINE)
        else:
            text = re.sub(
                r"(- \*\*Run name:\*\* .*\n)",
                lambda m: m.group(1) + line + "\n",
                text,
                count=1,
            )
        summary_path.write_text(text)
        print(f"[OK] {exp_name}  ->  {url}")


if __name__ == "__main__":
    main()
