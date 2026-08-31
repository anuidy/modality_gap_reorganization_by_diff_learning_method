from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import EXPECTED_RUNS, load_run_config  # noqa: E402
from training.engine import run_training, validate_training_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one branch of the locked eight-run experiment matrix.")
    parser.add_argument("--run", choices=tuple(EXPECTED_RUNS), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "train_runs.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen settings, files, hashes, and the training manifest without loading a model.",
    )
    args = parser.parse_args()

    config = load_run_config(args.config.resolve(), args.run, PROJECT_ROOT)
    if args.validate_only:
        print(validate_training_inputs(config))
        return
    final_checkpoint = run_training(
        config,
        device_name=args.device,
        resume_checkpoint=args.resume.resolve() if args.resume else None,
    )
    print(f"final_checkpoint={final_checkpoint}")


if __name__ == "__main__":
    main()
