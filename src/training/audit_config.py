"""Read only the settings needed for a one-batch audit, without training defaults."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from training.config import EXPECTED_RUNS, validate_experiment_matrix
from training.data_control import _sha256_file, validate_formal_data_identity


@dataclass(frozen=True)
class AuditConfig:
    run_id: str
    model_name: str
    branch: str
    checkpoint: Path
    resources: dict[str, Path]
    data: SimpleNamespace
    seed: int
    batch_size: int
    precision: str
    augmentation: dict[str, Any]
    options: dict[str, Any]
    identities: dict[str, Any]


def load_audit_config(
    path: Path, run_id: str, project_root: Path, *, seed: int, batch_size: int,
    augmentation: str, flip_probability: float, crop_scale: tuple[float, float] | None = None,
    alpha: float | None = None, precision: str = "bf16",
) -> AuditConfig:
    if not 0 <= seed < 2**32 or batch_size < 2:
        raise ValueError("Audit seed must be in [0, 2**32), and batch size must be >= 2.")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("Audit precision must be bf16 or fp32.")
    if not math.isfinite(flip_probability) or not 0 <= flip_probability <= 1:
        raise ValueError("Specify a finite flip probability in [0, 1].")
    transform: dict[str, Any] = {"name": augmentation, "horizontal_flip_probability": flip_probability}
    if augmentation == "random_resized_crop":
        if crop_scale is None or not (0 < crop_scale[0] <= crop_scale[1] <= 1):
            raise ValueError("Random crop requires explicit --crop-scale MIN MAX in (0, 1].")
        transform.update(scale_min=crop_scale[0], scale_max=crop_scale[1])
    elif augmentation != "resize_center_crop" or crop_scale is not None:
        raise ValueError("Use resize_center_crop without crop scale, or random_resized_crop with scale.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_experiment_matrix(payload)
    model_name, branch = EXPECTED_RUNS[run_id]
    model = payload["models"][model_name]

    def resolve(value: str) -> Path:
        p = Path(value)
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()

    data_values = dict(payload["controls"]["data"])
    for key in ("dataset_spec", "split_lock", "train_manifest", "validation_manifest",
                "lcs_probe_manifest", "coco_probe_manifest", "image_root"):
        data_values[key] = resolve(data_values[key])
    data = SimpleNamespace(**data_values)
    # This validator reads dataset identity fields only, not optimizer/budget.
    data_identity = validate_formal_data_identity(data, _sha256_file(data.train_manifest))
    if data_identity["mode"] != "formal_locked_lcs_558k":
        raise ValueError("Audit requires the locked LCS Train manifest.")
    if not data.image_root.is_dir():
        raise FileNotFoundError(data.image_root)
    checkpoint = resolve(model["checkpoint"])
    checkpoint_sha = _sha256_file(checkpoint)
    if checkpoint_sha != model["checkpoint_sha256"]:
        raise ValueError("Audit M0 checkpoint SHA-256 mismatch.")
    resources = {k: resolve(v) for k, v in model["resources"].items()}
    for resource in resources.values():
        if not resource.exists():
            raise FileNotFoundError(resource)
    options = dict(model.get("options", {}))
    if model_name == "albef":
        if alpha is None or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("ALBEF audit requires an explicit --alpha in [0, 1].")
        # A single diagnostic batch has no warmup schedule. Report the explicit
        # target-mixing coefficient instead of inventing training progress.
        options.update(alpha=alpha, alpha_warmup_steps=0)
    elif alpha is not None:
        raise ValueError("--alpha is only applicable to ALBEF.")
    return AuditConfig(run_id, model_name, branch, checkpoint, resources, data, seed, batch_size,
                       precision, transform, options,
                       {"config_sha256": _sha256_file(path), "checkpoint_sha256": checkpoint_sha,
                        "data": data_identity})
