"""Aggregate v3i 500-episode eval results across (model, pack_size) pairs.

Reads summary.json from each ``experiments/<exp>/results/eval_v3i_500*``
folder and writes a comparison markdown to ``experiments/v3i_density_sweep.md``.

Tries the following directory names per (exp, pack) pair:
  - pack=N → ``eval_v3i_500_pack{N:02d}``
  - pack=10 fallback (no categorization) → ``eval_v3i_500``

Reports:
  - overall metrics per (model, pack)
  - per-archetype crash rates
  - crash-category distribution per (model, pack), with mean ego/partner speed
    and approach angle per category
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPS = [
    ("065", "065_baseline_v3i", "MLP baseline"),
    ("066", "066_vit_only_v3i", "ViT-only"),
    ("067", "067_anom_attn_risk_gru_v3i", "AnomAttn+Risk+GRU"),
    ("068", "068_anom_attn_risk_gru_pbs_truncbonus_v3i", "067 + PBS + truncbonus"),
]

CATEGORIES = [
    "ego_initiated",
    "partner_initiated",
    "ran_into_wreck",
    "ran_into_static",
    "mutual",
    "unknown",
]

ARCHETYPES_ORDER = [
    "none", "tailgater", "sudden_braker", "lane_drifter",
    "erratic_speed", "rear_ender",
]


def _find_summary(exp_folder: str, pack: int) -> Path | None:
    """Resolve the eval directory for (exp, pack)."""
    base = REPO_ROOT / "experiments" / exp_folder / "results"
    cand_specific = base / f"eval_v3i_500_pack{pack:02d}" / "summary.json"
    if cand_specific.exists():
        return cand_specific
    if pack == 10:
        cand_default = base / "eval_v3i_500" / "summary.json"
        if cand_default.exists():
            return cand_default
    return None


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def _fmt_f(x: float | None, n: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{n}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack-sizes", type=int, nargs="+", default=[5, 7, 10])
    p.add_argument("--out", type=str,
                   default=str(REPO_ROOT / "experiments" / "v3i_density_sweep.md"))
    args = p.parse_args()

    table: dict[tuple[str, int], dict] = {}
    for exp_id, exp_folder, _label in EXPS:
        for pack in args.pack_sizes:
            path = _find_summary(exp_folder, pack)
            if path is None:
                continue
            try:
                table[(exp_id, pack)] = _load(path)
            except Exception as e:
                print(f"WARN: failed to load {path}: {e}")

    lines: list[str] = []
    add = lines.append

    add("# v3i Density Sweep + Crash Categorization Eval")
    add("")
    add("4 models × pack ∈ {5, 7, 10} × 500 shared seeds (20000..20499). "
        "Each crashed episode is categorized by inspecting the partner "
        "vehicle at the crashing step.")
    add("")

    # ---- 1. Overall metrics
    add("## 1. Overall metrics")
    add("")
    add("| Exp | Label | Pack | Crash% | Reward | Length | Trunc% |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for exp_id, _exp_folder, label in EXPS:
        for pack in args.pack_sizes:
            s = table.get((exp_id, pack))
            if s is None:
                add(f"| {exp_id} | {label} | {pack} | — | — | — | — |")
                continue
            m = s["metrics"]
            add(
                f"| {exp_id} | {label} | {pack} "
                f"| {_fmt_pct(m['crash_rate'])} "
                f"| {m['mean_reward']:.2f} ± {m['std_reward']:.2f} "
                f"| {m['mean_episode_length']:.1f} "
                f"| {_fmt_pct(m['truncation_rate'])} |"
            )
        add("|  |  |  |  |  |  |  |")
    add("")

    # ---- 2. Per-archetype crash rate (cross-tab by pack)
    add("## 2. Per-archetype crash rate, by pack size")
    add("")
    for pack in args.pack_sizes:
        any_present = any((exp, pack) in table for exp, _, _ in EXPS)
        if not any_present:
            continue
        add(f"### pack = {pack}")
        add("")
        archs_present = set()
        for exp_id, _e, _l in EXPS:
            s = table.get((exp_id, pack))
            if s is None:
                continue
            archs_present.update(s.get("per_archetype", {}).keys())
        archs = [a for a in ARCHETYPES_ORDER if a in archs_present]
        archs += sorted(archs_present - set(archs))

        header = "| Archetype | n | " + " | ".join(e for e, _, _ in EXPS) + " |"
        sep = "|---|---:|" + "|".join(["---:"] * len(EXPS)) + "|"
        add(header)
        add(sep)
        for a in archs:
            n_val = None
            row_cells: list[str] = []
            for exp_id, _e, _l in EXPS:
                s = table.get((exp_id, pack))
                if s is None:
                    row_cells.append("—")
                    continue
                cell = s["per_archetype"].get(a)
                if cell is None:
                    row_cells.append("—")
                    continue
                if n_val is None:
                    n_val = cell["n"]
                row_cells.append(_fmt_pct(cell["crash_rate"]))
            add(f"| {a} | {n_val if n_val is not None else '—'} | "
                + " | ".join(row_cells) + " |")
        add("")

    # ---- 3. Crash category distribution per (model, pack)
    add("## 3. Crash category distribution")
    add("")
    add("Fraction of *crashes* in each category (rows do not sum across "
        "all episodes; uncrashed episodes are excluded). `(n)` is the raw "
        "crash count.")
    add("")
    add("| Exp | Pack | n crashes | ego_initiated | partner_initiated | ran_into_wreck | ran_into_static | mutual | unknown |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for exp_id, _exp_folder, _label in EXPS:
        for pack in args.pack_sizes:
            s = table.get((exp_id, pack))
            if s is None:
                continue
            pc = s.get("per_crash_category") or {}
            n_total = sum(d["n"] for d in pc.values())
            row = [exp_id, str(pack), str(n_total)]
            for cat in ["ego_initiated", "partner_initiated", "ran_into_wreck",
                        "ran_into_static", "mutual", "unknown"]:
                d = pc.get(cat)
                if d and n_total:
                    row.append(f"{d['n']:3d} ({100*d['n']/n_total:4.1f}%)")
                else:
                    row.append("—")
            add("| " + " | ".join(row) + " |")
    add("")

    # ---- 4. Per-category kinematic details (only at the most informative pack)
    add("## 4. Per-category kinematic detail")
    add("")
    add("Mean ego speed, partner speed, approach angle (deg, 0 = ahead, "
        "±90 = side, ±180 = behind) at crash, by (exp, pack).")
    add("")
    for exp_id, _exp_folder, label in EXPS:
        printed_header = False
        for pack in args.pack_sizes:
            s = table.get((exp_id, pack))
            if s is None:
                continue
            pc = s.get("per_crash_category") or {}
            if not pc:
                continue
            if not printed_header:
                add(f"### {exp_id} — {label}")
                add("")
                add("| Pack | Category | n | ego_v | partner_v | rel_v | angle° | crash step | adv partner% |")
                add("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
                printed_header = True
            for cat in ["ego_initiated", "partner_initiated", "ran_into_wreck",
                        "ran_into_static", "mutual", "unknown"]:
                d = pc.get(cat)
                if not d or d["n"] == 0:
                    continue
                add(
                    f"| {pack} | {cat} | {d['n']} "
                    f"| {_fmt_f(d['mean_ego_speed_pre'])} "
                    f"| {_fmt_f(d['mean_partner_speed_pre'])} "
                    f"| {_fmt_f(d['mean_rel_speed'])} "
                    f"| {_fmt_f(d['mean_approach_angle_deg'], 0)} "
                    f"| {_fmt_f(d['mean_crash_step'], 1)} "
                    f"| {_fmt_pct(d['adversary_partner_frac'])} |"
                )
        if printed_header:
            add("")

    # ---- 5. Category by archetype at pack=10 (most informative)
    interesting_pack = 10 if any((e, 10) in table for e, _, _ in EXPS) else max(
        (p for (_e, p) in table.keys()), default=None)
    if interesting_pack is not None:
        add(f"## 5. Crash category × spawn archetype at pack={interesting_pack}")
        add("")
        for exp_id, _e, label in EXPS:
            s = table.get((exp_id, interesting_pack))
            if s is None:
                continue
            cba = s.get("category_by_archetype") or {}
            if not cba:
                continue
            add(f"### {exp_id} — {label}")
            add("")
            cats = list({c for d in cba.values() for c in d.keys()})
            cats = [c for c in CATEGORIES if c in cats]
            header = "| Archetype | total crashes | " + " | ".join(cats) + " |"
            sep = "|---|---:|" + "|".join(["---:"] * len(cats)) + "|"
            add(header)
            add(sep)
            for a in ARCHETYPES_ORDER:
                d = cba.get(a)
                if not d:
                    continue
                total = sum(d.values())
                row = [a, str(total)]
                for c in cats:
                    n = d.get(c, 0)
                    row.append(f"{n} ({100*n/total:.0f}%)" if total else "—")
                add("| " + " | ".join(row) + " |")
            add("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
