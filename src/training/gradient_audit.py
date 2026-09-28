"""Independent, backward-only audits; never create an optimizer or checkpoint."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import os
from pathlib import Path
import time
from typing import Any

import torch

from datasets.training_pairs import (
    DeterministicEpochSampler, PairedTrainingDataset, collate_raw_training_batch, load_training_pairs,
)
from training.audit_config import AuditConfig
from training.data_control import _sha256_file
from training.audit_groups import inspect_gradients
from training.backends import TrainingBackend, create_training_backend
from training.engine import _autocast, _current_git_commit, _seed_everything
from training.validation import _preserve_rng, _seed_validation
from objectives.contrastive import BRANCH_DEFINITIONS


def audit_cases(branch: str) -> list[tuple[str, int, str]]:
    if branch == "standard":
        return [("I<->T", 0, "total")]
    if branch in BRANCH_DEFINITIONS:
        return [(branch, 0, "total")]
    if branch == "itc_only":
        return [("ITC", 0, "ITC")]
    if branch == "full_albef":
        return [(component, 0, component) for component in ("total", "ITC", "ITM", "MLM")]
    raise ValueError(f"Unknown audit branch: {branch}")


def state_hashes(backend: torch.nn.Module) -> dict[str, str]:
    result = {}
    for name, value in backend.state_dict().items():
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def _changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))


def _allowed_forward_change(model: str, name: str) -> bool:
    return model == "albef" and (name == "model.temp" or name.startswith((
        "model.visual_encoder_m.", "model.vision_proj_m.", "model.text_encoder_m.", "model.text_proj_m.",
    )))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def _count_vista_paths(backend: TrainingBackend):
    """Instrument methods, including image->encode_mm reuse; restore on errors."""
    counts = {name: 0 for name in ("encode_image", "encode_text", "encode_mm")}
    if backend.model_name != "vista":
        yield {}
        return
    sentinel = object()
    previous = {}
    model = backend.model
    try:
        for name in counts:
            previous[name] = model.__dict__.get(name, sentinel)
            method = getattr(model, name)

            def counted(*args, _name=name, _method=method, **kwargs):
                counts[_name] += 1
                return _method(*args, **kwargs)

            object.__setattr__(model, name, counted)
        yield counts
    finally:
        for name, value in previous.items():
            if value is sentinel:
                object.__delattr__(model, name)
            else:
                object.__setattr__(model, name, value)


def audit_backend(
    backend: TrainingBackend, prepared: Any, *, branch: str, seed: int, precision: str,
) -> list[dict[str, Any]]:
    """Audit a disposable backend; restore tensor state, modes and RNG on exit.

    Existing gradients are cleared, so callers must not pass a live trainer.
    Full ALBEF recomputes the same seeded full forward for each selected loss.
    """
    if backend.model_name == "albef" and backend._deferred_state_updates.active:
        raise RuntimeError("Audit requires an idle, disposable ALBEF backend.")
    initial = {k: v.detach().cpu().clone() for k, v in backend.state_dict().items()}
    initial_hashes = state_hashes(backend)
    modes = {module: module.training for module in backend.modules()}
    reports = []
    device = backend.device
    backend.set_random_seed(seed)
    with _preserve_rng():
        try:
            for label, step, component in audit_cases(branch):
                backend.zero_grad(set_to_none=True)
                backend.load_state_dict(initial, strict=True)
                backend.train()
                _seed_validation(seed)
                report: dict[str, Any] = {"case": label, "component": component, "status": "error"}
                reports.append(report)
                result = loss = None
                try:
                    _sync(device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    backend.begin_optimizer_step(1)
                    with _count_vista_paths(backend) as calls:
                        with _autocast(device, precision):
                            result = backend(prepared, branch, step)
                            loss = result.loss if component == "total" else result.metrics[f"loss/{component}"]
                    _sync(device)
                    report["forward_seconds"] = time.perf_counter() - start
                    report["path_calls"] = dict(calls)
                    # Outside model timing: identify explicit EMA/temp updates.
                    after_forward = state_hashes(backend)
                    report["forward_state_changes"] = _changed(initial_hashes, after_forward)
                    report["unexpected_forward_changes"] = [n for n in report["forward_state_changes"]
                                                              if not _allowed_forward_change(backend.model_name, n)]
                    report["losses"] = {
                        k: float(v.detach().float().cpu()) if bool(torch.isfinite(v.detach()).all()) else None
                        for k, v in result.metrics.items()
                    }
                    report["nonfinite_losses"] = [k for k, v in report["losses"].items() if v is None]
                    report["relation_audit"] = dataclasses.asdict(result.audit) if result.audit is not None else None
                    if not bool(torch.isfinite(loss.detach()).all()):
                        raise FloatingPointError("Selected loss is nonfinite; no backward was attempted.")
                    start = time.perf_counter()
                    loss.backward()
                    _sync(device)
                    report["backward_seconds"] = time.perf_counter() - start
                    # Capture peaks before gradient statistics allocate temporaries.
                    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                    after_backward = state_hashes(backend)
                    report["backward_state_changes"] = _changed(after_forward, after_backward)
                    report["gradients"] = inspect_gradients(backend, backend.model_name, component)
                    path_ok = True
                    if backend.model_name == "vista":
                        expected = (1, 1, 1) if branch in {"standard", "fixed_2m"} else (1, 1, 2)
                        path_ok = tuple(calls[k] for k in ("encode_image", "encode_text", "encode_mm")) == expected
                    buffers = dict(backend.named_buffers())
                    report["queue_buffers"] = {
                        name: {"requires_grad": value.requires_grad, "has_gradient": value.grad is not None,
                               "unchanged": initial_hashes.get(name) == after_backward.get(name)}
                        for name, value in buffers.items() if name in {"model.image_queue", "model.text_queue", "model.queue_ptr"}
                    }
                    queue_ok = True
                    if backend.model_name == "albef":
                        queue_ok = len(report["queue_buffers"]) == 3 and all(
                            not v["requires_grad"] and not v["has_gradient"] and v["unchanged"]
                            for v in report["queue_buffers"].values())
                    passed = (report["gradients"]["status"] == "pass" and path_ok and queue_ok
                              and not report["unexpected_forward_changes"] and not report["backward_state_changes"]
                              and not report["nonfinite_losses"])
                    report["path_status"] = "pass" if path_ok else "fail"
                    report["status"] = "pass" if passed else "fail"
                except Exception as error:
                    report["error"] = {"type": type(error).__name__, "message": str(error)}
                    if device.type == "cuda":
                        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                finally:
                    # Never flush: that would enqueue. Discard collected features.
                    if backend.model_name == "albef":
                        backend._deferred_state_updates.abort()
                    backend.zero_grad(set_to_none=True)
                    result = loss = None
                if report["status"] == "error":
                    break
        finally:
            backend.load_state_dict(initial, strict=True)
            backend.zero_grad(set_to_none=True)
            for module, mode in modes.items():
                module.training = mode
            if state_hashes(backend) != initial_hashes:
                raise RuntimeError("Audit failed to restore the initial model tensor state.")
    return reports


def _audit_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Audit device must be cpu or cuda.")
    if device.type == "cuda":
        # set_device rejects an index-free torch.device('cuda'). Resolve the
        # current visible device explicitly; respect a supplied cuda:N index.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def run_gradient_audit(config: AuditConfig, device_name: str = "cuda") -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1
    ):
        raise RuntimeError("Gradient audit requires a single process/GPU.")
    device = _audit_device(device_name)
    _autocast(device, config.precision)  # Reject unsupported BF16 before loading.
    _seed_everything(config.seed, deterministic=True)
    report: dict[str, Any] = {
        "schema_version": 1, "kind": "gradient_audit", "run_id": config.run_id,
        "model": config.model_name, "branch": config.branch, "status": "error",
        "identities": config.identities, "code_commit": _current_git_commit(),
        "settings": {"seed": config.seed, "batch_size": config.batch_size, "precision": config.precision,
                     "augmentation": config.augmentation, "model_options": config.options},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "policy": {"optimizer_steps": 0, "checkpoints_written": False, "training_config_modified": False,
                   "same_prepared_batch_per_case": True, "tensor_state_restored_per_case": True,
                   "albef_momentum_update": "once before each forward, then restored",
                   "albef_queue_update": "collected features discarded; no flush/enqueue",
                   "timing_excludes": "state hashing/restoration and gradient statistics",
                   "scope": "one physical batch; no accumulation, optimizer-memory or resume validation"},
        "cases": [],
    }
    try:
        start = time.perf_counter()
        pairs = load_training_pairs(config.data.train_manifest, config.data.image_root)
        if len(pairs) < config.batch_size:
            raise ValueError("Train manifest has fewer samples than the requested batch size.")
        dataset = PairedTrainingDataset(pairs)
        sampler = iter(DeterministicEpochSampler(pairs, config.seed))
        indices = [next(sampler) for _ in range(config.batch_size)]
        raw = collate_raw_training_batch([dataset[index] for index in indices])
        report["batch"] = {"indices": indices, "sample_ids": raw.sample_ids, "semantic_ids": raw.semantic_ids}
        report["data_load_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        backend = create_training_backend(config.model_name, config.checkpoint, config.resources,
                                          device, config.augmentation, config.options)
        _sync(device)
        report["model_load_seconds"] = time.perf_counter() - start
        root = Path(__file__).resolve().parents[2]
        sources = [root / name for name in (
            "src/training/audit_config.py", "src/training/audit_groups.py",
            "src/training/gradient_audit.py", "src/training/backends.py",
            "src/training/albef_accumulation.py", "src/training/engine.py",
            "src/training/validation.py", "src/training/data_control.py",
            "src/datasets/training_pairs.py", "src/objectives/contrastive.py",
            "scripts/training/audit_gradients.py",
        )]
        sources.append(Path(inspect.getfile(type(backend.model))))
        report["source_sha256"] = {str(path): _sha256_file(path) for path in sources}
        report["checkpoint_load"] = backend.checkpoint_load_report
        if any(backend.checkpoint_load_report.values()):
            raise RuntimeError("M0 load has missing/unexpected keys; inspect checkpoint_load before proceeding.")
        start = time.perf_counter()
        prepared = backend.prepare_batch(raw, config.seed)
        _sync(device)
        report["prepare_batch_seconds"] = time.perf_counter() - start
        report["cases"] = audit_backend(backend, prepared, branch=config.branch,
                                         seed=config.seed, precision=config.precision)
        report["status"] = "pass" if report["cases"] and all(c["status"] == "pass" for c in report["cases"]) else "fail"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report
