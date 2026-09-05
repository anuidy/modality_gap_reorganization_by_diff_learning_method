from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.audit_config import load_audit_config  # noqa: E402
from training.config import EXPECTED_RUNS  # noqa: E402
from training.gradient_audit import run_gradient_audit  # noqa: E402


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("Expected a finite number.")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight and backward-only audit of one locked training branch.")
    parser.add_argument("--run", choices=tuple(EXPECTED_RUNS), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/training/train_runs.yaml")
    parser.add_argument("--seed", type=int, required=True, help="Diagnostic seed; does not freeze the training seed.")
    parser.add_argument("--batch-size", type=int, required=True, help="One physical batch, at least two pairs.")
    parser.add_argument("--augmentation", choices=("resize_center_crop", "random_resized_crop"), required=True)
    parser.add_argument("--flip-probability", type=finite_float, required=True)
    parser.add_argument("--crop-scale", type=finite_float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--alpha", type=finite_float, help="Required for ALBEF: diagnostic soft-target mixing coefficient.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--validate-only", action="store_true", help="Check resource/data identity without loading a model.")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = PROJECT_ROOT / "outputs/pilot/gradient_audit" / args.run / stamp
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = load_audit_config(
            args.config.resolve(), args.run, PROJECT_ROOT, seed=args.seed, batch_size=args.batch_size,
            augmentation=args.augmentation, flip_probability=args.flip_probability,
            crop_scale=tuple(args.crop_scale) if args.crop_scale is not None else None,
            alpha=args.alpha, precision=args.precision,
        )
        if args.validate_only:
            report = {"schema_version": 1, "kind": "preflight_identity_only", "run_id": args.run,
                      "status": "pass", "identities": config.identities,
                      "gradient_audit_performed": False, "model_loaded": False}
        else:
            report = run_gradient_audit(config, args.device)
    except Exception as error:
        report = {"schema_version": 1, "run_id": args.run, "status": "error",
                  "error": {"type": type(error).__name__, "message": str(error)}}
    report["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    destination = output / "report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"status={report['status']} report={destination}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
