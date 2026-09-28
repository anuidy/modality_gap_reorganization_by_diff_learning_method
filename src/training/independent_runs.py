"""Schedule independent single-GPU jobs without importing a model or torch."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class TrainingJob:
    run_id: str
    output_dir: Path
    command: tuple[str, ...]
    validation_command: tuple[str, ...]
    config_run_id: str | None = None
    seed: int | None = None


def child_environment(
    gpu_id: str, parent: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if parent is None else parent)
    distributed_names = {
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    }
    for name in tuple(environment):
        if name in distributed_names or name.startswith(("TORCHELASTIC_", "OMPI_", "PMI_", "PMIX_")):
            environment.pop(name)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["WORLD_SIZE"] = "1"
    return environment


def validate_job_selection(jobs: Sequence[TrainingJob]) -> None:
    if not jobs:
        raise ValueError("Select at least one run explicitly.")
    if len({job.run_id for job in jobs}) != len(jobs):
        raise ValueError("Duplicate run IDs are not allowed.")
    directories = [job.output_dir.resolve() for job in jobs]
    for index, directory in enumerate(directories):
        for previous in directories[:index]:
            if directory == previous or directory in previous.parents or previous in directory.parents:
                raise ValueError("Run output directories must be distinct and cannot be nested.")


def validate_gpu_ids(gpu_ids: Sequence[str]) -> None:
    if len(gpu_ids) not in (1, 2) or len(set(gpu_ids)) != len(gpu_ids) or any(
        not gpu.isascii() or not gpu.isdecimal() or gpu != str(int(gpu))
        for gpu in gpu_ids
    ):
        raise ValueError("Specify one or two distinct non-negative GPU indices, e.g. --gpus 0 or --gpus 0 1.")


def build_jobs(
    config_path: Path, run_ids: Sequence[str], project_root: Path,
    seeds: Sequence[int] | None = None, *, stop_after_step: int | None = None,
) -> tuple[TrainingJob, ...]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        raise ValueError("The config must contain a runs mapping.")
    selected_seeds = list(seeds) if seeds is not None else [payload.get("controls", {}).get("seed")]
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seed selection must be nonempty and contain no duplicates.")
    if any(seed is not None and (type(seed) is not int or seed < 0) for seed in selected_seeds):
        raise ValueError("Seeds must be non-negative integers.")
    jobs = []
    for run_id in run_ids:
        run = runs.get(run_id)
        if not isinstance(run, dict) or not isinstance(run.get("output_dir"), str) or not run["output_dir"]:
            raise ValueError(f"Unknown run or missing output_dir: {run_id}")
        output_dir = (project_root / run["output_dir"]).resolve()
        if output_dir == project_root or output_dir in project_root.parents:
            raise ValueError("A run output directory cannot contain the project root.")
        command = (
            sys.executable, "-u", str(project_root / "scripts/training/train.py"),
            "--config", str(config_path), "--run", run_id, "--device", "cuda:0",
        )
        for seed in selected_seeds:
            instance_id = run_id if seed is None else f"{run_id}_seed_{seed}"
            instance_output = output_dir if seed is None else output_dir / f"seed_{seed}"
            instance_command = command if seed is None else (*command, "--seed", str(seed))
            if stop_after_step is not None:
                instance_command = (*instance_command, "--stop-after-step", str(stop_after_step))
            jobs.append(TrainingJob(instance_id, instance_output, instance_command,
                                    (*instance_command, "--validate-only"), run_id, seed))
    validate_job_selection(jobs)
    return tuple(jobs)


@contextlib.contextmanager
def training_run_lock(output_dir: Path) -> Iterator[None]:
    """Atomic CLI ownership lock; never remove a lock belonging to another process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".training.lock"
    with path.open("x", encoding="utf-8") as handle:
        try:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
            handle.flush()
            yield
        finally:
            # Close before unlink for Windows; the context manager's second close is harmless.
            handle.close()
            path.unlink()


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_jobs(
    jobs: Sequence[TrainingJob], gpu_ids: Sequence[str], project_root: Path,
    poll_interval: float = 0.1,
) -> dict[str, int | None]:
    validate_job_selection(jobs)
    validate_gpu_ids(gpu_ids)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Start this launcher as one ordinary process, not with torchrun.")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive.")
    for job in jobs:
        if job.output_dir.exists() and (not job.output_dir.is_dir() or any(job.output_dir.iterdir())):
            raise FileExistsError(f"Fresh launch requires an empty output directory: {job.output_dir}")

    results: dict[str, int | None] = {job.run_id: None for job in jobs}
    active: dict[str, tuple[TrainingJob, subprocess.Popen, object]] = {}
    with contextlib.ExitStack() as locks:
        # Prevent two launchers in this checkout from occupying the same GPU slot.
        lock_root = project_root / "outputs/.independent_gpu_locks" / socket.gethostname()
        for gpu in sorted(gpu_ids):
            locks.enter_context(training_run_lock(lock_root / f"gpu_{gpu}"))
        try:
            # Validate every selected task before any training process is started.
            for job in jobs:
                if not job.validation_command:
                    continue
                process = subprocess.Popen(
                    job.validation_command, cwd=project_root,
                    env=child_environment(gpu_ids[0]), start_new_session=os.name == "posix",
                )
                try:
                    code = process.wait()
                except BaseException:
                    _stop_process(process)
                    raise
                if code:
                    results[job.run_id] = code
                    return results

            pending = iter(jobs)
            exhausted = False
            failed = False
            while active or not exhausted:
                # Poll all active tasks before filling any free slot.
                for gpu, (job, process, log) in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        results[job.run_id] = code
                        failed = failed or code != 0
                        log.close()
                        del active[gpu]
                        print(json.dumps({"run": job.run_id, "gpu": gpu, "exit_code": code}), flush=True)
                if failed:
                    exhausted = True
                if not exhausted:
                    for gpu in gpu_ids:
                        if gpu in active:
                            continue
                        job = next(pending, None)
                        if job is None:
                            exhausted = True
                            break
                        job.output_dir.mkdir(parents=True, exist_ok=True)
                        log = (job.output_dir / "launcher.log").open("x", encoding="utf-8")
                        try:
                            process = subprocess.Popen(
                                job.command, cwd=project_root, env=child_environment(gpu),
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name == "posix",
                            )
                        except BaseException:
                            log.close()
                            raise
                        active[gpu] = (job, process, log)
                        print(json.dumps({"run": job.run_id, "gpu": gpu, "pid": process.pid}), flush=True)
                if active:
                    time.sleep(poll_interval)
        finally:
            for _, process, log in active.values():
                try:
                    _stop_process(process)
                finally:
                    log.close()
    return results
