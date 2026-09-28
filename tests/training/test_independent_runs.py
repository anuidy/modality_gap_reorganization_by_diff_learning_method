import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.independent_runs import (  # noqa: E402
    TrainingJob,
    build_jobs,
    child_environment,
    run_jobs,
    training_run_lock,
)


WORKER = r"""
import json
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
name = sys.argv[2]
mode = sys.argv[3]
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
gpu_lock = root / ("gpu-" + gpu + ".lock")
gpu_lock.mkdir()
try:
    started = time.monotonic_ns()
    (root / (name + ".started")).write_text("started", encoding="utf-8")
    if mode in {"barrier", "fail", "finish_after_failure"}:
        deadline = time.monotonic() + 8
        while not all((root / (peer + ".started")).exists() for peer in ("first", "second")):
            if time.monotonic() >= deadline:
                raise RuntimeError("The first two independent jobs did not overlap.")
            time.sleep(0.01)
    if mode == "fail":
        (root / "failure_observed").write_text("failed", encoding="utf-8")
        sys.exit(7)
    if mode == "finish_after_failure":
        deadline = time.monotonic() + 8
        while not (root / "failure_observed").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("The other worker never reached its failure.")
            time.sleep(0.01)
        time.sleep(0.1)
    else:
        time.sleep(0.03)
    record = {
        "gpu": gpu,
        "world_size": os.environ.get("WORLD_SIZE"),
        "rank": os.environ.get("RANK"),
        "started": started,
        "finished": time.monotonic_ns(),
    }
    (root / (name + ".json")).write_text(json.dumps(record), encoding="utf-8")
    print("completed " + name, flush=True)
finally:
    gpu_lock.rmdir()
"""


class IndependentRunsTest(unittest.TestCase):
    def test_single_gpu_workers_run_sequentially(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [self._job(root, name) for name in ("first", "second", "third")]
            result = run_jobs(jobs, ("0",), root, poll_interval=0.01)
            self.assertEqual(result, {job.run_id: 0 for job in jobs})
            records = [json.loads((root / (job.run_id + ".json")).read_text()) for job in jobs]
            self.assertTrue(all(record["gpu"] == "0" and record["world_size"] == "1" for record in records))
            for previous, following in zip(records, records[1:]):
                self.assertLessEqual(previous["finished"], following["started"])

    def _write_config(self, root, first_output="outputs/first", second_output="outputs/second"):
        config = root / "runs.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "runs": {
                        "clip_standard": {
                            "model": "clip",
                            "branch": "standard",
                            "output_dir": first_output,
                        },
                        "clip_mixed_3m_fn_off": {
                            "model": "clip",
                            "branch": "mixed_3m_fn_off",
                            "output_dir": second_output,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return config

    def _job(self, root, name, mode="normal", validation_command=()):
        return TrainingJob(
            run_id=name,
            output_dir=root / "outputs" / name,
            command=(sys.executable, "-c", WORKER, str(root), name, mode),
            validation_command=validation_command,
        )

    def test_child_environment_is_independent_and_removes_distributed_state(self):
        parent = {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
            "TORCHELASTIC_RUN_ID": "previous-job",
            "PYTHONPATH": "keep-this",
        }
        original = parent.copy()
        child = child_environment("1", parent)
        self.assertEqual(parent, original)
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(child["WORLD_SIZE"], "1")
        self.assertEqual(child["PYTHONPATH"], "keep-this")
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
            self.assertNotIn(name, child)

    def test_plan_can_be_built_without_frozen_training_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root)
            jobs = build_jobs(config, ("clip_standard", "clip_mixed_3m_fn_off"), root)
            self.assertEqual([job.run_id for job in jobs], ["clip_standard", "clip_mixed_3m_fn_off"])
            self.assertEqual(jobs[0].output_dir, (root / "outputs" / "first").resolve())
            self.assertFalse((root / "outputs").exists())

    def test_plan_rejects_duplicate_runs_and_unknown_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root)
            for runs in (("clip_standard", "clip_standard"), ("unknown_run",)):
                with self.subTest(runs=runs), self.assertRaises(ValueError):
                    build_jobs(config, runs, root)

    def test_plan_rejects_identical_and_nested_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for other in ("outputs/first", "outputs/other/../first", "outputs/first/child", "outputs"):
                with self.subTest(other=other):
                    config = self._write_config(root, second_output=other)
                    with self.assertRaises(ValueError):
                        build_jobs(config, ("clip_standard", "clip_mixed_3m_fn_off"), root)

    def test_run_lock_rejects_second_writer_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with training_run_lock(output):
                    self.assertTrue((output / ".training.lock").exists())
                    with self.assertRaises(FileExistsError):
                        with training_run_lock(output):
                            self.fail("The same run was locked twice.")
                    self.assertTrue((output / ".training.lock").exists())
                    raise RuntimeError("intentional")
            self.assertFalse((output / ".training.lock").exists())
            with training_run_lock(output):
                pass

    def test_real_cpu_workers_overlap_with_one_job_per_visible_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [self._job(root, name, "barrier" if name in {"first", "second"} else "normal")
                    for name in ("first", "second", "third", "fourth")]
            with mock.patch.dict(os.environ, {"RANK": "9", "LOCAL_RANK": "9", "WORLD_SIZE": "1"}):
                result = run_jobs(jobs, ("0", "1"), root, poll_interval=0.01)
            self.assertEqual(result, {job.run_id: 0 for job in jobs})
            records = [json.loads((root / (job.run_id + ".json")).read_text(encoding="utf-8")) for job in jobs]
            self.assertEqual({record["gpu"] for record in records}, {"0", "1"})
            self.assertTrue(all(record["world_size"] == "1" and record["rank"] is None for record in records))
            self.assertLess(max(record["started"] for record in records[:2]), min(record["finished"] for record in records[:2]))
            for gpu in ("0", "1"):
                intervals = sorted((record["started"], record["finished"]) for record in records if record["gpu"] == gpu)
                self.assertTrue(all(previous[1] <= following[0] for previous, following in zip(intervals, intervals[1:])))
            for job in jobs:
                self.assertIn("completed " + job.run_id, (job.output_dir / "launcher.log").read_text(encoding="utf-8"))

    def test_failed_worker_stops_queue_but_allows_running_peer_to_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [self._job(root, "first", "fail"), self._job(root, "second", "finish_after_failure"), self._job(root, "third")]
            result = run_jobs(jobs, ("0", "1"), root, poll_interval=0.01)
            self.assertEqual(result, {"first": 7, "second": 0, "third": None})
            self.assertTrue((root / "second.json").is_file())
            self.assertFalse((root / "third.started").exists())

    def test_failed_prevalidation_starts_no_training_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [
                self._job(root, "first", validation_command=(sys.executable, "-c", "pass")),
                self._job(root, "second", validation_command=(sys.executable, "-c", "raise SystemExit(3)")),
            ]
            result = run_jobs(jobs, ("0", "1"), root, poll_interval=0.01)
            self.assertEqual(result["second"], 3)
            self.assertFalse(any(root.glob("*.started")))

    def test_duplicate_gpu_ids_are_rejected_before_any_worker_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for gpu_ids in (("0", "0"), ("0", "00"), ("0", "０")):
                with self.subTest(gpu_ids=gpu_ids), self.assertRaises(ValueError):
                    run_jobs([self._job(root, "first"), self._job(root, "second")], gpu_ids, root)
            self.assertFalse(any(root.glob("*.started")))

    def test_spawn_error_reaps_started_worker_and_releases_gpu_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [
                TrainingJob(name, root / "outputs" / name,
                            (sys.executable, "-c", "import time; time.sleep(30)"), ())
                for name in ("first", "second")
            ]
            real_popen = subprocess.Popen
            started = []

            def launch(*args, **kwargs):
                if started:
                    raise OSError("intentional spawn failure")
                process = real_popen(*args, **kwargs)
                started.append(process)
                return process

            with mock.patch("training.independent_runs.subprocess.Popen", side_effect=launch):
                with self.assertRaisesRegex(OSError, "intentional spawn failure"):
                    run_jobs(jobs, ("0", "1"), root, poll_interval=0.01)
            self.assertEqual(len(started), 1)
            self.assertIsNotNone(started[0].poll())
            self.assertFalse(list((root / "outputs" / ".independent_gpu_locks").rglob(".training.lock")))


if __name__ == "__main__":
    unittest.main()
