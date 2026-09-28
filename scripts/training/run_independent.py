"""One or two GPU slots; each worker runs an independent single-GPU experiment."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.independent_runs import build_jobs, run_jobs, validate_gpu_ids  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/training/train_runs.yaml")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--runs", nargs="+", help="Explicit run IDs from the selected config.")
    selection.add_argument("--matrix", action="store_true", help="All 79 main-model tasks across three explicit seeds; ALBEF excluded.")
    selection.add_argument("--gate", action="store_true", help="Three CLIP gate branches, one seed, stopping at p020.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Run each selected branch with these seeds in independent directories.")
    parser.add_argument("--stop-after-step", type=int, help="Stop each task at a saved checkpoint without shortening the full LR budget.")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="One or two physical GPU indices; default: 0.")
    parser.add_argument("--execute", action="store_true", help="Validate all tasks, then start training; default only prints a plan.")
    args = parser.parse_args()
    validate_gpu_ids(args.gpus)
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds
    selected_runs = args.runs
    if args.matrix:
        seeds = seeds if seeds is not None else payload["controls"].get("seeds")
        if seeds is None or len(seeds) != 3:
            raise ValueError("The formal main matrix requires three confirmed seeds (--seeds).")
        selected_runs = [name for name, run in payload["runs"].items() if run["model"] != "albef"]
    if args.gate:
        seeds = seeds if seeds is not None else [payload["controls"].get("seed")]
        if len(seeds) != 1 or seeds[0] is None:
            raise ValueError("The gate requires one confirmed seed (--seeds).")
        selected_runs = ["clip_fixed_3m_fn_on", "clip_fixed_3m_fn_off", "clip_full_3m_fn_off"]
        from training.checkpoint_plan import build_checkpoint_plan
        plan = build_checkpoint_plan(payload["controls"]["budget"]["max_steps"])
        gate_step = next(p.optimizer_step for p in plan.trajectory_points if p.label == "p020")
        if args.stop_after_step is not None and args.stop_after_step != gate_step:
            raise ValueError("The gate stops at p020; do not redefine its budget.")
        args.stop_after_step = gate_step
    jobs = build_jobs(args.config, selected_runs, PROJECT_ROOT, seeds, stop_after_step=args.stop_after_step)
    if args.matrix:
        jobs = tuple(j for j in jobs if j.config_run_id != "vista_fixed_2m" or j.seed == seeds[0])
    print(json.dumps({
        "mode": "independent_single_gpu", "execute": args.execute,
        "gpu_slots": args.gpus, "world_size_per_job": 1,
        "note": "A launch plan does not certify the training protocol or GPU memory compatibility.",
        "jobs": [{"run": j.run_id, "output_dir": str(j.output_dir), "command": j.command} for j in jobs],
    }, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return 0

    # Read the actual controlled config; the launcher never overrides experiment settings.
    from training.config import load_run_config
    for job in jobs:
        config = load_run_config(args.config.resolve(), job.config_run_id or job.run_id, PROJECT_ROOT, {"seed": job.seed})
        if config.output_dir.resolve() != job.output_dir.resolve():
            raise ValueError("Launch plan and training config disagree on the task output directory.")
        if config.micro_batch_size != 36 or config.gradient_accumulation != 1:
            raise ValueError(f"{job.run_id}: independent formal tasks require batch=36 and accumulation=1.")

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_jobs(jobs, args.gpus, PROJECT_ROOT)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps({"exit_codes": results, "not_started": [name for name, code in results.items() if code is None]}, indent=2))
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
