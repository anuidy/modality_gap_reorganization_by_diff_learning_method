from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import EXPECTED_RUNS, load_run_config  # noqa: E402
from training.engine import run_training, validate_training_inputs  # noqa: E402
from training.independent_runs import training_run_lock  # noqa: E402


def resolve_resume_path(value: Path) -> Path:
    path = value.resolve()
    if path.suffix != ".json":
        return path
    pointer = json.loads(path.read_text(encoding="utf-8"))
    target = Path(str(pointer["path"]))
    target = (path.parent / target).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"{path} points at {target}, which does not exist.")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one seed of a formal single-GPU experiment.")
    parser.add_argument("--run", choices=tuple(EXPECTED_RUNS), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "train_runs.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, help="Full-state checkpoints only; unavailable in trajectory-only mode.")
    parser.add_argument("--resume-source-manifest", type=Path, help="Continue in a new output with recording-only changes; all training controls must match.")
    parser.add_argument("--extend-training-budget", action="store_true", help="Append updates after a completed budget while preserving the original LR decay horizon.")
    parser.add_argument("--retime-cosine-for-extension", action="store_true", help="Explicitly retime the cosine horizon; this continuation is marked reference-only.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-after-step", type=int, help="Stop at a saved checkpoint without shortening the LR schedule.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen settings, files, hashes, and the training manifest without loading a model.",
    )
    args = parser.parse_args()

    config = load_run_config(args.config.resolve(), args.run, PROJECT_ROOT, {"seed": args.seed})
    if args.validate_only:
        print(validate_training_inputs(config))
        return
    # Acquire before model loading; applies to both direct and scheduled CLI runs.
    with training_run_lock(config.output_dir):
        final_checkpoint = run_training(
            config,
            device_name=args.device,
            resume_checkpoint=resolve_resume_path(args.resume) if args.resume else None,
            stop_after_step=args.stop_after_step,
            resume_source_manifest=args.resume_source_manifest.resolve() if args.resume_source_manifest else None,
            extend_training_budget=args.extend_training_budget,
            retime_cosine_for_extension=args.retime_cosine_for_extension,
        )
    print(f"final_checkpoint={final_checkpoint}")


if __name__ == "__main__":
    main()
