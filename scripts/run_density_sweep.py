"""Density-sweep eval runner.

For each (model, pack_size) pair in {065,066,067,068} × {5,7}, runs the
500-episode v3i eval (shared seeds 20000..20499) and writes the result
under ``experiments/<exp>/results/eval_v3i_500_pack<N>/summary.json``.

Each eval call is a subprocess so a crash in one config doesn't kill
the rest. Output is streamed to stdout and to a per-run log file.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / ".predictive_driving" / "bin" / "python"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_v3i_500.py"

# (exp_folder, env_id)
MODELS = [
    ("065_baseline_v3i", "adversarial-highway-v3i-raw"),
    ("066_vit_only_v3i", "adversarial-highway-v3i-dict"),
    ("067_anom_attn_risk_gru_v3i", "adversarial-highway-v3i-h10"),
    ("068_anom_attn_risk_gru_pbs_truncbonus_v3i", "adversarial-highway-v3i-h10"),
]


def run_one(exp_folder: str, env_id: str, pack: int, n_episodes: int,
            n_videos: int, base_seed: int) -> tuple[bool, float, Path]:
    config = REPO_ROOT / "experiments" / exp_folder / "config.yaml"
    model = REPO_ROOT / "experiments" / exp_folder / "results" / "model.zip"
    out_dir = REPO_ROOT / "experiments" / exp_folder / "results" / f"eval_v3i_500_pack{pack:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    cmd = [
        str(PY), str(EVAL_SCRIPT),
        "--config", str(config),
        "--model", str(model),
        "--env-id", env_id,
        "--out-dir", str(out_dir),
        "--n-episodes", str(n_episodes),
        "--n-videos", str(n_videos),
        "--base-seed", str(base_seed),
        "--pack-n-vehicles", str(pack),
    ]
    print(f"\n→ {exp_folder} | pack={pack}", flush=True)
    print("  " + " ".join(cmd), flush=True)

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logf.write(proc.stdout or "")
    dt = time.time() - t0
    ok = proc.returncode == 0

    # Echo the final summary lines if eval succeeded.
    if proc.stdout:
        tail = proc.stdout.strip().splitlines()
        for line in tail[-20:]:
            print("  " + line, flush=True)
    if not ok:
        print(f"  FAILED (returncode={proc.returncode}). See {log_path}", flush=True)
    else:
        print(f"  done in {dt:.1f}s — summary at {out_dir/'summary.json'}", flush=True)
    return ok, dt, out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack-sizes", type=int, nargs="+", default=[5, 7])
    p.add_argument("--n-episodes", type=int, default=500)
    p.add_argument("--n-videos", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=20_000)
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Subset of exp folders to run; default = all 065-068.")
    p.add_argument("--cooldown", type=int, default=30,
                   help="Seconds to sleep between consecutive runs so the GPU/PSU "
                        "settle before the next CUDA init. The system rebooted "
                        "on 2026-05-11 between back-to-back evals with no idle gap.")
    args = p.parse_args()

    selected = MODELS
    if args.models:
        selected = [(f, e) for (f, e) in MODELS if f in args.models]
        if not selected:
            print("No matching models found.", flush=True)
            sys.exit(1)

    pairs = [(f, e, pack) for (f, e) in selected for pack in args.pack_sizes]

    t_start = time.time()
    results = []
    for i, (exp_folder, env_id, pack) in enumerate(pairs):
        if i > 0 and args.cooldown > 0:
            print(f"\n  cooldown: sleeping {args.cooldown}s before next run", flush=True)
            time.sleep(args.cooldown)
        ok, dt, out_dir = run_one(
            exp_folder, env_id, pack,
            args.n_episodes, args.n_videos, args.base_seed,
        )
        results.append((exp_folder, pack, ok, dt, out_dir))

    print("\n=== sweep summary ===")
    for exp_folder, pack, ok, dt, out_dir in results:
        mark = "OK " if ok else "FAIL"
        print(f"  {mark}  {exp_folder:50s} pack={pack:2d}  {dt:6.1f}s  → {out_dir}")
    print(f"total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
