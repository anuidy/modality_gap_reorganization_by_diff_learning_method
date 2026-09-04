from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.training_pairs import (
    DeterministicEpochSampler,
    PairedTrainingDataset,
    RawTrainingBatch,
    collate_raw_training_batch,
    load_training_pairs,
)
from training.backends import TrainingBackend, create_training_backend
from training.checkpoint_plan import TrajectoryPoint, build_checkpoint_plan
from training.config import RunConfig
from training.data_control import validate_formal_data_identity
from training.optim import build_weight_decay_parameter_groups
from training.validation import run_validation


CHECKPOINT_SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _config_signature(config: RunConfig) -> str:
    canonical = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _seed_everything(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _seed_model_forward(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("precision=bf16 requires a CUDA GPU with BF16 support.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("precision must be bf16 or fp32.")


def _learning_rate(config: RunConfig, optimizer_step: int) -> float:
    if optimizer_step < config.warmup_steps and config.warmup_steps > 0:
        return config.learning_rate * (optimizer_step + 1) / config.warmup_steps
    remaining_steps = config.max_steps - config.warmup_steps
    progress = (optimizer_step - config.warmup_steps) / max(1, remaining_steps - 1)
    progress = min(1.0, max(0.0, progress))
    multiplier = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return config.learning_rate * multiplier


class BatchStream:
    def __init__(
        self,
        loader: DataLoader[RawTrainingBatch],
        sampler: DeterministicEpochSampler,
        epoch: int = 0,
        batch_index: int = 0,
    ) -> None:
        if len(loader) == 0:
            raise ValueError("The training set is smaller than one drop-last micro-batch.")
        self.loader = loader
        self.sampler = sampler
        self.epoch = epoch
        self.batch_index = batch_index
        self._iterator: Iterator[RawTrainingBatch] | None = None

    def _open_epoch(self) -> None:
        self.sampler.set_epoch(self.epoch)
        self._iterator = iter(self.loader)
        for _ in range(self.batch_index):
            try:
                next(self._iterator)
            except StopIteration as error:
                raise ValueError("Resume checkpoint batch_index exceeds the epoch length.") from error

    def next(self) -> tuple[int, int, RawTrainingBatch]:
        if self._iterator is None:
            self._open_epoch()
        assert self._iterator is not None
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batch_index = 0
            self._open_epoch()
            assert self._iterator is not None
            batch = next(self._iterator)
        epoch = self.epoch
        batch_index = self.batch_index
        self.batch_index += 1
        return epoch, batch_index, batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_index": self.batch_index}


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _trainable_signature(backend: TrainingBackend) -> str:
    names = [name for name, parameter in backend.named_parameters() if parameter.requires_grad]
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _validate_full_parameter_scope(backend: TrainingBackend, model_name: str) -> None:
    frozen = [name for name, parameter in backend.named_parameters() if not parameter.requires_grad]
    if model_name == "albef":
        native_momentum_prefixes = (
            "model.visual_encoder_m.",
            "model.vision_proj_m.",
            "model.text_encoder_m.",
            "model.text_proj_m.",
        )
        frozen = [name for name in frozen if not name.startswith(native_momentum_prefixes)]
    if frozen:
        raise RuntimeError(
            f"trainable_scope=full_parameter but parameters are frozen: {frozen[:5]}"
        )


def validate_training_inputs(config: RunConfig) -> dict[str, Any]:
    if not config.checkpoint.is_file():
        raise FileNotFoundError(config.checkpoint)
    actual_checkpoint_sha = sha256_file(config.checkpoint)
    if actual_checkpoint_sha != config.checkpoint_sha256:
        raise ValueError(
            f"M0 checkpoint SHA-256 mismatch for {config.model_name}: "
            f"expected {config.checkpoint_sha256}, found {actual_checkpoint_sha}."
        )
    for name, path in config.resources.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {config.model_name} resource {name}: {path}")
    train_manifest_sha256 = sha256_file(config.train_manifest)
    formal_data = validate_formal_data_identity(config, train_manifest_sha256)
    pairs = load_training_pairs(config.train_manifest, config.image_root)
    if len(pairs) < config.micro_batch_size:
        raise ValueError("Training data must contain at least one full micro-batch.")
    return {
        "checkpoint_sha256": actual_checkpoint_sha,
        "train_manifest_sha256": train_manifest_sha256,
        "training_pair_count": len(pairs),
        "formal_data": formal_data,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _checkpoint_provenance(
    config: RunConfig,
    completed_steps: int,
    progress_fraction: float,
    code_commit: str | None,
    evaluation_status: str,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "model_name": config.model_name,
        "branch": config.branch,
        "optimizer_step": completed_steps,
        "progress_fraction": progress_fraction,
        "m0_checkpoint_sha256": config.checkpoint_sha256,
        "config_sha256": _config_signature(config),
        "probe_manifests": {
            "lcs": config.lcs_probe_manifest_sha256,
            "coco": config.coco_probe_manifest_sha256,
        },
        "code_commit": code_commit,
        "evaluation": {"status": evaluation_status},
    }


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _append_checkpoint_index(output_directory: Path, payload: dict[str, Any]) -> None:
    _append_jsonl(output_directory / "checkpoint_index.jsonl", payload)


def _save_full_resume_checkpoint(
    output_directory: Path,
    config: RunConfig,
    backend: TrainingBackend,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    stream: BatchStream,
) -> tuple[Path, str]:
    checkpoint_directory = output_directory / "checkpoints" / "resume"
    destination = checkpoint_directory / f"step_{completed_steps:08d}.pt"
    provenance = _checkpoint_provenance(
        config,
        completed_steps,
        completed_steps / config.max_steps,
        _current_git_commit(),
        evaluation_status="not_applicable",
    )
    _atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "full_resume",
            "run_id": config.run_id,
            "config_sha256": _config_signature(config),
            "completed_steps": completed_steps,
            "stream": stream.state_dict(),
            "model": backend.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _rng_state(),
            "provenance": provenance,
        },
        destination,
    )
    artifact_sha256 = sha256_file(destination)
    _append_checkpoint_index(
        output_directory,
        {
            "event": "saved_full_resume",
            "path": str(destination.relative_to(output_directory)),
            "artifact_sha256": artifact_sha256,
            "checkpoint_sha256": artifact_sha256,
            "provenance": provenance,
        },
    )
    _write_json(
        checkpoint_directory / "latest.json",
        {
            "completed_steps": completed_steps,
            "path": destination.name,
            "artifact_sha256": artifact_sha256,
        },
    )
    return destination, artifact_sha256


def _retain_latest_resume_checkpoints(output_directory: Path, retention: int) -> None:
    checkpoint_directory = output_directory / "checkpoints" / "resume"
    checkpoints = sorted(checkpoint_directory.glob("step_*.pt"))
    while len(checkpoints) > retention:
        retired = checkpoints.pop(0)
        retired.unlink()
        _append_checkpoint_index(
            output_directory,
            {
                "event": "retired_full_resume",
                "path": str(retired.relative_to(output_directory)),
            },
        )


def _save_trajectory_snapshot(
    output_directory: Path,
    config: RunConfig,
    backend: TrainingBackend,
    point: TrajectoryPoint,
) -> tuple[Path, str]:
    if point.is_final:
        raise ValueError("The final trajectory point must reuse the final full checkpoint.")
    checkpoint_directory = output_directory / "checkpoints" / "trajectory"
    destination = checkpoint_directory / f"step_{point.optimizer_step:08d}_{point.label}_model.pt"
    provenance = _checkpoint_provenance(
        config,
        point.optimizer_step,
        point.progress_fraction,
        _current_git_commit(),
        evaluation_status="pending",
    )
    _atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "trajectory_model",
            "run_id": config.run_id,
            "config_sha256": _config_signature(config),
            "completed_steps": point.optimizer_step,
            "model": backend.state_dict(),
            "provenance": provenance,
        },
        destination,
    )
    artifact_sha256 = sha256_file(destination)
    metadata = {
        "checkpoint_kind": "trajectory_model",
        "path": destination.name,
        "artifact_sha256": artifact_sha256,
        "model_state_sha256": artifact_sha256,
        "provenance": provenance,
    }
    _write_json(destination.with_suffix(".json"), metadata)
    _append_checkpoint_index(output_directory, {"event": "saved_trajectory_model", **metadata})
    return destination, artifact_sha256


def _record_final_checkpoint(
    output_directory: Path,
    config: RunConfig,
    checkpoint_path: Path,
    artifact_sha256: str,
) -> None:
    provenance = _checkpoint_provenance(
        config,
        config.max_steps,
        1.0,
        _current_git_commit(),
        evaluation_status="pending",
    )
    payload = {
        "checkpoint_kind": "final_full_resume",
        "path": str(checkpoint_path.relative_to(output_directory)),
        "artifact_sha256": artifact_sha256,
        "checkpoint_sha256": artifact_sha256,
        "provenance": provenance,
    }
    _write_json(output_directory / "checkpoints" / "final.json", payload)
    _append_checkpoint_index(output_directory, {"event": "marked_final", **payload})


def run_training(
    config: RunConfig,
    device_name: str = "cuda",
    resume_checkpoint: Path | None = None,
) -> Path:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The formal first-stage implementation targets the rented single A800 only.")
    device = torch.device(device_name)
    _seed_everything(config.seed, config.deterministic)
    input_metadata = validate_training_inputs(config)
    checkpoint_plan = build_checkpoint_plan(
        max_steps=config.max_steps,
        trajectory_progress_fractions=config.trajectory_progress_fractions,
        resume_progress_interval=config.resume_progress_interval,
        resume_retention=config.resume_retention,
    )

    model_options = dict(config.model_options)
    if config.model_name == "albef" and model_options.get("alpha_warmup_steps") is None:
        model_options["alpha_warmup_steps"] = config.warmup_steps
    backend = create_training_backend(
        model_name=config.model_name,
        checkpoint=config.checkpoint,
        resources=config.resources,
        device=device,
        augmentation=config.augmentation,
        options=model_options,
    )
    _validate_full_parameter_scope(backend, config.model_name)
    backend.train()
    trainable_parameters = [parameter for parameter in backend.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("The training backend exposed no trainable parameters.")
    optimizer_parameter_groups = build_weight_decay_parameter_groups(
        backend,
        config.weight_decay,
        backend.no_weight_decay_parameter_names(),
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=0.0,
    )

    train_manifest_sha256 = sha256_file(config.train_manifest)
    formal_data = validate_formal_data_identity(config, train_manifest_sha256)
    pairs = load_training_pairs(config.train_manifest, config.image_root)
    dataset = PairedTrainingDataset(pairs)
    sampler = DeterministicEpochSampler(dataset, config.seed)
    loader: DataLoader[RawTrainingBatch] = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_raw_training_batch,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        multiprocessing_context="spawn" if config.num_workers > 0 else None,
    )
    validation_loader: DataLoader[RawTrainingBatch] | None = None
    if config.validation_manifest is not None:
        validation_pairs = load_training_pairs(config.validation_manifest, config.image_root)
        if (
            config.validation_sample_count is not None
            and len(validation_pairs) != config.validation_sample_count
        ):
            raise ValueError(
                f"Expected {config.validation_sample_count} Validation pairs, "
                f"found {len(validation_pairs)}."
            )
        if len(validation_pairs) % config.micro_batch_size != 0:
            raise ValueError(
                "Validation sample count must be divisible by micro_batch_size so every "
                "Validation query sees the same candidate count."
            )
        validation_loader = DataLoader(
            PairedTrainingDataset(validation_pairs),
            batch_size=config.micro_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_raw_training_batch,
            drop_last=False,
            persistent_workers=config.num_workers > 0,
            multiprocessing_context="spawn" if config.num_workers > 0 else None,
        )
        input_metadata["validation_pair_count"] = len(validation_pairs)
        input_metadata["validation_batch_count"] = len(validation_loader)

    completed_steps = 0
    stream_state = {"epoch": 0, "batch_index": 0}
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported training checkpoint schema.")
        if payload.get("checkpoint_kind") != "full_resume":
            raise ValueError("Only full resume checkpoints can continue training.")
        if payload.get("run_id") != config.run_id:
            raise ValueError("Resume checkpoint belongs to a different run.")
        if payload.get("config_sha256") != _config_signature(config):
            raise ValueError("Resume checkpoint was created with a different controlled configuration.")
        backend.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        completed_steps = int(payload["completed_steps"])
        stream_state = dict(payload["stream"])
        _restore_rng_state(payload["rng"])

    stream = BatchStream(loader, sampler, **stream_state)
    output_directory = config.output_dir
    run_manifest_path = output_directory / "run_manifest.json"
    if resume_checkpoint is None and run_manifest_path.exists():
        raise FileExistsError(
            f"Run output already exists: {run_manifest_path}. Use --resume instead of overwriting it."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    parameter_counts = backend.parameter_counts()
    _write_json(
        run_manifest_path,
        {
            "schema_version": 1,
            "status": "running",
            "config": _jsonable(config),
            "config_sha256": _config_signature(config),
            "input_metadata": input_metadata,
            "parameter_counts": parameter_counts,
            "checkpoint_policy": {
                "m0_progress_fraction": 0.0,
                "trajectory_points": [
                    {
                        "progress_fraction": point.progress_fraction,
                        "optimizer_step": point.optimizer_step,
                        "label": point.label,
                    }
                    for point in checkpoint_plan.trajectory_points
                ],
                "resume_steps": sorted(checkpoint_plan.resume_steps),
                "resume_retention": checkpoint_plan.resume_retention,
                "trajectory_evaluation_mode": "batch_after_branch_completion",
            },
            "optimizer_parameter_groups": [
                {
                    "group_name": str(group["group_name"]),
                    "weight_decay": float(group["weight_decay"]),
                    "parameter_tensor_count": len(group["params"]),
                    "parameter_count": sum(parameter.numel() for parameter in group["params"]),
                }
                for group in optimizer.param_groups
            ],
            "trainable_parameter_name_sha256": _trainable_signature(backend),
            "control_guarantees": {
                "fresh_m0_initialization": resume_checkpoint is None,
                "stateless_per_batch_augmentation_seed": True,
                "stateless_per_microbatch_model_rng_seed": True,
                "relation_cycle_unit": "optimizer_step",
                "single_gpu_in_batch_negatives": True,
                "albef_state_updates_per_optimizer_step": True,
                "validation_at_full_resume_checkpoint_steps": True,
                "validation_parameter_updates": False,
                "validation_checkpoint_selection": False,
                "probe_evaluation_during_training": False,
            },
        },
    )
    metrics_path = output_directory / "train_metrics.jsonl"
    validation_metrics_path = output_directory / "validation_metrics.jsonl"
    last_validation_step: int | None = None
    final_checkpoint: Path | None = None

    while completed_steps < config.max_steps:
        learning_rate = _learning_rate(config, completed_steps)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        backend.begin_optimizer_step(config.gradient_accumulation)
        accumulated: dict[str, float] = {}
        audits: list[dict[str, Any]] = []

        for micro_step in range(config.gradient_accumulation):
            epoch, batch_index, raw_batch = stream.next()
            augmentation_seed = config.seed + 1_000_003 * epoch + batch_index
            prepared = backend.prepare_batch(raw_batch, augmentation_seed)
            forward_seed = (
                config.seed
                + 2_000_003
                + completed_steps * config.gradient_accumulation
                + micro_step
            )
            _seed_model_forward(forward_seed)
            with _autocast(device, config.precision):
                result = backend(prepared, config.branch, completed_steps)
                scaled_loss = result.loss / config.gradient_accumulation
            if not torch.isfinite(result.loss.detach()).item():
                raise FloatingPointError(f"Non-finite loss at optimizer step {completed_steps}.")
            scaled_loss.backward()
            for name, value in result.metrics.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value.detach().float().cpu())
            if result.audit is not None:
                audits.append(dataclasses.asdict(result.audit))

        backend.before_optimizer_step()
        relation_audit = None
        if audits:
            if len({audit["relation"] for audit in audits}) != 1:
                raise RuntimeError("All accumulation micro-batches in one optimizer step must use one relation.")
            for field in ("batch_size", "positive_terms", "candidates_per_query", "negatives_per_query"):
                if len({audit[field] for audit in audits}) != 1:
                    raise RuntimeError(f"Count matching changed within an optimizer step: {field}.")
            first_audit = audits[0]
            relation_audit = {
                "relation": first_audit["relation"],
                "micro_batches": len(audits),
                "micro_batch_size": first_audit["batch_size"],
                "positive_terms_per_microbatch": first_audit["positive_terms"],
                "positive_terms_per_optimizer_step": sum(audit["positive_terms"] for audit in audits),
                "candidates_per_query": first_audit["candidates_per_query"],
                "negatives_per_query": first_audit["negatives_per_query"],
            }
        if config.gradient_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(config.gradient_clip_norm)
            )
            if not torch.isfinite(gradient_norm).item():
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {completed_steps}.")
        else:
            gradient_norm = torch.tensor(float("nan"))

        optimizer.step()
        backend.after_optimizer_step()
        completed_steps += 1
        record = {
            "completed_steps": completed_steps,
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm.detach().float().cpu()),
            "metrics": {
                name: value / config.gradient_accumulation for name, value in accumulated.items()
            },
            "relation_audit": relation_audit,
            "data_stream": stream.state_dict(),
        }
        if completed_steps % config.log_interval == 0 or completed_steps == 1:
            _append_jsonl(metrics_path, record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        trajectory_point = checkpoint_plan.trajectory_point_at(completed_steps)
        if trajectory_point is not None and not trajectory_point.is_final:
            _save_trajectory_snapshot(output_directory, config, backend, trajectory_point)

        if checkpoint_plan.is_resume_step(completed_steps):
            checkpoint_path, checkpoint_sha256 = _save_full_resume_checkpoint(
                output_directory, config, backend, optimizer, completed_steps, stream
            )
            _retain_latest_resume_checkpoints(output_directory, checkpoint_plan.resume_retention)
            if trajectory_point is not None and trajectory_point.is_final:
                _record_final_checkpoint(
                    output_directory, config, checkpoint_path, checkpoint_sha256
                )
                final_checkpoint = checkpoint_path
            if validation_loader is not None:
                validation_record = run_validation(
                    backend=backend,
                    loader=validation_loader,
                    branch=config.branch,
                    optimizer_step=completed_steps,
                    seed=config.seed,
                    device=device,
                    precision=config.precision,
                )
                validation_record["checkpoint"] = str(checkpoint_path)
                _append_jsonl(validation_metrics_path, validation_record)
                print(
                    json.dumps({"validation": validation_record}, ensure_ascii=False),
                    flush=True,
                )
                last_validation_step = completed_steps

    if final_checkpoint is None:
        raise RuntimeError("The checkpoint plan did not save a final full checkpoint.")
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["completed_steps"] = completed_steps
    manifest["final_checkpoint"] = str(final_checkpoint.relative_to(output_directory))
    manifest["last_validation_step"] = last_validation_step
    _write_json(run_manifest_path, manifest)
    return final_checkpoint
