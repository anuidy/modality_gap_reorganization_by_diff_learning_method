from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from training.checkpoint_plan import (
    DEFAULT_RESUME_PROGRESS_INTERVAL, DEFAULT_RESUME_RETENTION, build_checkpoint_plan,
)


from objectives.contrastive import BRANCH_DEFINITIONS


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate YAML configuration keys are not allowed.")
        return super().construct_mapping(node, deep=deep)

EXPECTED_RUNS: dict[str, tuple[str, str]] = {
    f"{model}_{branch}": (model, branch)
    for model in ("clip", "beit3", "vista") for branch in BRANCH_DEFINITIONS
}
EXPECTED_RUNS.update({"albef_itc_only": ("albef", "itc_only"), "albef_full": ("albef", "full_albef")})


ALLOWED_RUN_KEYS = {"model", "branch", "output_dir"}


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    model_name: str
    branch: str
    checkpoint: Path
    checkpoint_sha256: str
    resources: Mapping[str, Path]
    train_manifest: Path
    image_root: Path
    output_dir: Path
    seed: int
    deterministic: bool
    precision: str
    num_workers: int
    micro_batch_size: int
    gradient_accumulation: int
    optimizer_type: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    scheduler_type: str
    warmup_steps: int
    min_lr_ratio: float
    max_steps: int
    trajectory_progress_fractions: tuple[float, ...]
    resume_progress_interval: float
    resume_retention: int
    log_interval: int
    gradient_clip_norm: float | None
    augmentation: Mapping[str, Any]
    model_options: Mapping[str, Any]
    save_resume_checkpoints: bool = True
    validation_progress_interval: float | None = None
    dataset_name: str | None = None
    dataset_spec: Path | None = None
    split_lock: Path | None = None
    split_lock_sha256: str | None = None
    train_manifest_sha256: str | None = None
    validation_manifest: Path | None = None
    validation_manifest_sha256: str | None = None
    validation_sample_count: int | None = None
    lcs_probe_manifest: Path | None = None
    lcs_probe_manifest_sha256: str | None = None
    coco_probe_manifest: Path | None = None
    coco_probe_manifest_sha256: str | None = None
    progress_reference_steps: int | None = None
    scheduler_decay_steps: int | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=UniqueKeyLoader)
    if not isinstance(payload, dict):
        raise ValueError("Training config must contain a YAML mapping.")
    return payload


def validate_experiment_matrix(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported training config schema_version.")
    runs = payload.get("runs")
    models = payload.get("models")
    controls = payload.get("controls")
    if not isinstance(runs, dict) or not isinstance(models, dict) or not isinstance(controls, dict):
        raise ValueError("Training config requires controls, models, and runs mappings.")
    if set(runs) != set(EXPECTED_RUNS):
        missing = sorted(set(EXPECTED_RUNS) - set(runs))
        extra = sorted(set(runs) - set(EXPECTED_RUNS))
        raise ValueError(f"The formal matrix must contain all nine main-model branches and the two ALBEF branches; missing={missing}, extra={extra}.")

    for run_id, (expected_model, expected_branch) in EXPECTED_RUNS.items():
        run = runs[run_id]
        if not isinstance(run, dict):
            raise ValueError(f"Run {run_id} must be a mapping.")
        unexpected_keys = set(run) - ALLOWED_RUN_KEYS
        if unexpected_keys:
            raise ValueError(
                f"Run {run_id} overrides controlled fields {sorted(unexpected_keys)}. "
                "Put shared settings under controls or the model block."
            )
        if run.get("model") != expected_model or run.get("branch") != expected_branch:
            raise ValueError(f"Run {run_id} does not match the locked model/branch definition.")
        if expected_model not in models:
            raise ValueError(f"Missing model block for {expected_model}.")
        if models[expected_model].get("trainable_scope") != "full_parameter":
            raise ValueError(f"Model {expected_model} must use trainable_scope=full_parameter.")


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _resolved_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _required(values: Mapping[str, Any]) -> None:
    missing = sorted(name for name, value in values.items() if value is None)
    if missing:
        raise ValueError(
            "Training choices are not frozen yet. Set these fields in train_runs.yaml or pass overrides: "
            + ", ".join(missing)
        )


def load_run_config(
    config_path: Path,
    run_id: str,
    project_root: Path,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    payload = _load_payload(config_path)
    validate_experiment_matrix(payload)
    if run_id not in EXPECTED_RUNS:
        raise ValueError(f"Unknown run_id: {run_id}")

    run = payload["runs"][run_id]
    model = payload["models"][run["model"]]
    controls = payload["controls"]
    supplied = dict(overrides or {})
    checkpointing = controls.get("checkpointing")
    if not isinstance(checkpointing, Mapping):
        raise ValueError("Training config requires a controls.checkpointing mapping.")

    def choose(name: str, value: Any) -> Any:
        return supplied[name] if name in supplied and supplied[name] is not None else value

    def model_control(section: str, name: str) -> Any:
        model_value = _nested(model, section, name)
        return model_value if model_value is not None else _nested(controls, section, name)

    values = {
        "dataset_name": choose("dataset_name", _nested(controls, "data", "dataset_name")),
        "dataset_spec": choose("dataset_spec", _nested(controls, "data", "dataset_spec")),
        "split_lock": choose("split_lock", _nested(controls, "data", "split_lock")),
        "split_lock_sha256": choose(
            "split_lock_sha256", _nested(controls, "data", "split_lock_sha256")
        ),
        "train_manifest": choose("train_manifest", _nested(controls, "data", "train_manifest")),
        "train_manifest_sha256": choose(
            "train_manifest_sha256", _nested(controls, "data", "train_manifest_sha256")
        ),
        "validation_manifest": choose(
            "validation_manifest", _nested(controls, "data", "validation_manifest")
        ),
        "validation_manifest_sha256": choose(
            "validation_manifest_sha256",
            _nested(controls, "data", "validation_manifest_sha256"),
        ),
        "validation_sample_count": choose(
            "validation_sample_count", _nested(controls, "data", "validation_sample_count")
        ),
        "lcs_probe_manifest": choose(
            "lcs_probe_manifest", _nested(controls, "data", "lcs_probe_manifest")
        ),
        "lcs_probe_manifest_sha256": choose(
            "lcs_probe_manifest_sha256",
            _nested(controls, "data", "lcs_probe_manifest_sha256"),
        ),
        "coco_probe_manifest": choose(
            "coco_probe_manifest", _nested(controls, "data", "coco_probe_manifest")
        ),
        "coco_probe_manifest_sha256": choose(
            "coco_probe_manifest_sha256",
            _nested(controls, "data", "coco_probe_manifest_sha256"),
        ),
        "image_root": choose("image_root", _nested(controls, "data", "image_root")),
        "seed": choose("seed", controls.get("seed")),
        "micro_batch_size": choose("micro_batch_size", _nested(model, "batch", "micro_batch_size")),
        "gradient_accumulation": choose(
            "gradient_accumulation", _nested(model, "batch", "gradient_accumulation")
        ),
        "optimizer_type": choose("optimizer_type", model_control("optimizer", "type")),
        "learning_rate": choose("learning_rate", model_control("optimizer", "learning_rate")),
        "weight_decay": choose("weight_decay", model_control("optimizer", "weight_decay")),
        "scheduler_type": choose("scheduler_type", model_control("scheduler", "type")),
        "warmup_steps": choose("warmup_steps", model_control("scheduler", "warmup_steps")),
        "min_lr_ratio": choose("min_lr_ratio", model_control("scheduler", "min_lr_ratio")),
        "max_steps": choose("max_steps", model_control("budget", "max_steps")),
        "trajectory_progress_fractions": choose(
            "trajectory_progress_fractions", checkpointing.get("trajectory_progress_fractions")
        ),
        "resume_progress_interval": choose(
            "resume_progress_interval", checkpointing.get("resume_progress_interval", DEFAULT_RESUME_PROGRESS_INTERVAL)
        ),
        "resume_retention": choose(
            "resume_retention", checkpointing.get("resume_retention", DEFAULT_RESUME_RETENTION)
        ),
        "augmentation_name": choose("augmentation_name", _nested(model, "augmentation", "name")),
        "augmentation_scale_min": choose(
            "augmentation_scale_min", _nested(model, "augmentation", "scale_min")
        ),
        "augmentation_scale_max": choose(
            "augmentation_scale_max", _nested(model, "augmentation", "scale_max")
        ),
        "augmentation_hflip": choose(
            "augmentation_hflip", _nested(model, "augmentation", "horizontal_flip_probability")
        ),
    }
    _required(values)

    integer_positive = (
        "micro_batch_size",
        "gradient_accumulation",
        "max_steps",
    )
    for name in integer_positive:
        if not isinstance(values[name], int) or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(values["seed"], int) or values["seed"] < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(values["warmup_steps"], int) or values["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be a non-negative integer.")
    if values["warmup_steps"] >= values["max_steps"]:
        raise ValueError("warmup_steps must be smaller than max_steps.")
    scheduler_decay_steps = controls.get("scheduler", {}).get("decay_steps")
    if scheduler_decay_steps is not None and (type(scheduler_decay_steps) is not int or
            not values["warmup_steps"] < scheduler_decay_steps <= values["max_steps"]):
        raise ValueError("scheduler.decay_steps must be an integer after warmup and no later than max_steps.")
    try:
        trajectory_progress_fractions = tuple(
            float(fraction) for fraction in values["trajectory_progress_fractions"]
        )
    except (TypeError, ValueError) as error:
        raise ValueError("trajectory_progress_fractions must be a numeric sequence.") from error
    try:
        resume_progress_interval = float(values["resume_progress_interval"])
        resume_retention = int(values["resume_retention"])
    except (TypeError, ValueError) as error:
        raise ValueError("resume checkpoint settings must be numeric.") from error
    save_resume_checkpoints = checkpointing.get("save_resume_checkpoints", True)
    if type(save_resume_checkpoints) is not bool:
        raise ValueError("save_resume_checkpoints must be a boolean.")
    interval_value = checkpointing.get("validation_progress_interval")
    validation_progress_interval = float(interval_value) if interval_value is not None else None
    progress_reference_steps = checkpointing.get("progress_reference_steps")
    build_checkpoint_plan(
        max_steps=int(values["max_steps"]),
        trajectory_progress_fractions=trajectory_progress_fractions,
        resume_progress_interval=resume_progress_interval,
        resume_retention=resume_retention,
        save_resume_checkpoints=save_resume_checkpoints,
        validation_progress_interval=validation_progress_interval,
        progress_reference_steps=progress_reference_steps,
    )
    if values["optimizer_type"] != "adamw":
        raise ValueError("The current training engine supports optimizer.type=adamw.")
    if values["scheduler_type"] != "cosine":
        raise ValueError("The current training engine supports scheduler.type=cosine.")
    if float(values["learning_rate"]) <= 0 or float(values["weight_decay"]) < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    if not 0 <= float(values["min_lr_ratio"]) <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1].")
    if not 0 <= float(values["augmentation_hflip"]) <= 1:
        raise ValueError("augmentation horizontal flip probability must be in [0, 1].")
    scale_min = float(values["augmentation_scale_min"])
    scale_max = float(values["augmentation_scale_max"])
    if not 0 < scale_min <= scale_max <= 1:
        raise ValueError("augmentation scale must satisfy 0 < scale_min <= scale_max <= 1.")
    if int(values["micro_batch_size"]) < 2:
        raise ValueError("micro_batch_size must be at least 2 for contrastive training.")
    if not isinstance(values["validation_sample_count"], int) or values["validation_sample_count"] <= 0:
        raise ValueError("validation_sample_count must be a positive integer.")
    if values["validation_sample_count"] < int(values["micro_batch_size"]):
        raise ValueError("Validation must contain at least one complete training-sized batch.")
    if values["dataset_name"] != "lcs_558k":
        raise ValueError("The formal training matrix must use dataset_name=lcs_558k.")
    precision = str(controls.get("precision", "bf16"))
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32.")
    num_workers = int(_nested(controls, "data", "num_workers") or 0)
    log_interval = int(model_control("budget", "log_interval") or 10)
    if num_workers < 0 or log_interval <= 0:
        raise ValueError("num_workers must be non-negative and log_interval positive.")
    beta1 = float(model_control("optimizer", "beta1") or 0.9)
    beta2 = float(model_control("optimizer", "beta2") or 0.999)
    epsilon = float(model_control("optimizer", "epsilon") or 1e-8)
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or epsilon <= 0:
        raise ValueError("AdamW beta values must be in [0, 1) and epsilon must be positive.")
    gradient_clip_norm = model_control("optimizer", "gradient_clip_norm")
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0:
        raise ValueError("gradient_clip_norm must be positive when enabled.")

    checkpoint = _resolved_path(project_root, model["checkpoint"])
    resources = {
        name: _resolved_path(project_root, value)
        for name, value in model.get("resources", {}).items()
    }
    output_value = run.get("output_dir", f"outputs/training/{run_id}")
    augmentation = {
        "name": values["augmentation_name"],
        "scale_min": scale_min,
        "scale_max": scale_max,
        "horizontal_flip_probability": float(values["augmentation_hflip"]),
    }
    if run["model"] == "albef":
        queue_size = int(_nested(model, "options", "queue_size") or 65536)
        if queue_size < int(values["micro_batch_size"]) * int(values["gradient_accumulation"]):
            raise ValueError("ALBEF batch cannot exceed queue capacity.")
    if run["branch"].startswith("mixed_") and int(values["micro_batch_size"]) % 3:
        raise ValueError("Mixed requires micro_batch_size divisible by three.")
    # Seed is part of the task identity; GPU count never changes the experiment.
    output_value = str(Path(output_value) / f"seed_{int(values['seed'])}")

    return RunConfig(
        run_id=f"{run_id}_seed_{int(values['seed'])}",
        model_name=run["model"],
        branch=run["branch"],
        checkpoint=checkpoint,
        checkpoint_sha256=model["checkpoint_sha256"],
        resources=resources,
        train_manifest=_resolved_path(project_root, str(values["train_manifest"])),
        image_root=_resolved_path(project_root, str(values["image_root"])),
        output_dir=_resolved_path(project_root, output_value),
        seed=int(values["seed"]),
        deterministic=bool(controls.get("deterministic", True)),
        precision=precision,
        num_workers=num_workers,
        micro_batch_size=int(values["micro_batch_size"]),
        gradient_accumulation=int(values["gradient_accumulation"]),
        optimizer_type=str(values["optimizer_type"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        scheduler_type=str(values["scheduler_type"]),
        warmup_steps=int(values["warmup_steps"]),
        min_lr_ratio=float(values["min_lr_ratio"]),
        max_steps=int(values["max_steps"]),
        trajectory_progress_fractions=trajectory_progress_fractions,
        resume_progress_interval=resume_progress_interval,
        resume_retention=resume_retention,
        log_interval=log_interval,
        gradient_clip_norm=(float(gradient_clip_norm) if gradient_clip_norm is not None else None),
        augmentation=augmentation,
        model_options=dict(model.get("options", {})),
        save_resume_checkpoints=save_resume_checkpoints,
        validation_progress_interval=validation_progress_interval,
        dataset_name=str(values["dataset_name"]),
        dataset_spec=_resolved_path(project_root, str(values["dataset_spec"])),
        split_lock=_resolved_path(project_root, str(values["split_lock"])),
        split_lock_sha256=str(values["split_lock_sha256"]),
        train_manifest_sha256=str(values["train_manifest_sha256"]),
        validation_manifest=_resolved_path(project_root, str(values["validation_manifest"])),
        validation_manifest_sha256=str(values["validation_manifest_sha256"]),
        validation_sample_count=int(values["validation_sample_count"]),
        lcs_probe_manifest=_resolved_path(project_root, str(values["lcs_probe_manifest"])),
        lcs_probe_manifest_sha256=str(values["lcs_probe_manifest_sha256"]),
        coco_probe_manifest=_resolved_path(project_root, str(values["coco_probe_manifest"])),
        coco_probe_manifest_sha256=str(values["coco_probe_manifest_sha256"]),
        progress_reference_steps=progress_reference_steps,
        scheduler_decay_steps=scheduler_decay_steps,
    )
