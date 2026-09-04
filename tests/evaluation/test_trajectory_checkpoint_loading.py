import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.artifact import sha256_file  # noqa: E402
from evaluation.trajectory import (  # noqa: E402
    load_completed_run_manifest,
    load_snapshot_into_adapter,
    mark_snapshot_evaluated,
    resolve_trajectory_snapshots,
)
from models.base import EmbeddingAdapter  # noqa: E402


class FakeAdapter(EmbeddingAdapter):
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)

    def encode_image(self, images):
        raise NotImplementedError

    def encode_text(self, texts):
        raise NotImplementedError

    def metadata(self):
        return {"model_name": "fake"}


def provenance(step: int, fraction: float) -> dict[str, object]:
    return {
        "run_id": "clip_standard",
        "model_name": "clip",
        "branch": "standard",
        "optimizer_step": step,
        "progress_fraction": fraction,
        "m0_checkpoint_sha256": "m0-sha",
        "config_sha256": "config-sha",
        "probe_manifests": {"lcs": "lcs-sha", "coco": "coco-sha"},
        "code_commit": "commit",
        "evaluation": {"status": "pending"},
    }


class TrajectoryCheckpointLoadingTest(unittest.TestCase):
    def test_resolves_model_and_final_snapshots_and_loads_backend_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            trajectory_directory = run_directory / "checkpoints" / "trajectory"
            resume_directory = run_directory / "checkpoints" / "resume"
            trajectory_directory.mkdir(parents=True)
            resume_directory.mkdir(parents=True)
            source_model = torch.nn.Linear(2, 1)
            with torch.no_grad():
                source_model.weight.fill_(3.0)
                source_model.bias.fill_(4.0)
            backend_state = {f"model.{name}": value for name, value in source_model.state_dict().items()}

            trajectory_path = trajectory_directory / "step_00000005_p050_model.pt"
            torch.save(
                {
                    "schema_version": 2,
                    "checkpoint_kind": "trajectory_model",
                    "run_id": "clip_standard",
                    "config_sha256": "config-sha",
                    "completed_steps": 5,
                    "model": backend_state,
                },
                trajectory_path,
            )
            trajectory_metadata = {
                "checkpoint_kind": "trajectory_model",
                "path": trajectory_path.name,
                "artifact_sha256": sha256_file(trajectory_path),
                "provenance": provenance(5, 0.5),
            }
            (trajectory_path.with_suffix(".json")).write_text(
                json.dumps(trajectory_metadata), encoding="utf-8"
            )

            final_path = resume_directory / "step_00000010.pt"
            torch.save(
                {
                    "schema_version": 2,
                    "checkpoint_kind": "full_resume",
                    "run_id": "clip_standard",
                    "config_sha256": "config-sha",
                    "completed_steps": 10,
                    "model": backend_state,
                    "optimizer": {},
                    "rng": {},
                    "stream": {},
                },
                final_path,
            )
            (run_directory / "checkpoints" / "final.json").write_text(
                json.dumps(
                    {
                        "checkpoint_kind": "final_full_resume",
                        "path": "checkpoints/resume/step_00000010.pt",
                        "artifact_sha256": sha256_file(final_path),
                        "provenance": provenance(10, 1.0),
                    }
                ),
                encoding="utf-8",
            )
            run_manifest = {
                "status": "complete",
                "config_sha256": "config-sha",
                "config": {
                    "run_id": "clip_standard",
                    "model_name": "clip",
                    "branch": "standard",
                    "checkpoint_sha256": "m0-sha",
                    "lcs_probe_manifest_sha256": "lcs-sha",
                    "coco_probe_manifest_sha256": "coco-sha",
                },
                "checkpoint_policy": {
                    "trajectory_points": [
                        {"label": "p050", "optimizer_step": 5, "progress_fraction": 0.5},
                        {"label": "p100", "optimizer_step": 10, "progress_fraction": 1.0},
                    ]
                },
            }
            (run_directory / "run_manifest.json").write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )

            loaded_manifest = load_completed_run_manifest(run_directory)
            snapshots = resolve_trajectory_snapshots(run_directory, loaded_manifest)
            adapter = FakeAdapter()
            load_snapshot_into_adapter(adapter, snapshots[0], loaded_manifest)

            self.assertTrue(torch.equal(adapter.model.weight, source_model.weight))
            self.assertTrue(torch.equal(adapter.model.bias, source_model.bias))
            self.assertEqual([snapshot.checkpoint_kind for snapshot in snapshots], ["trajectory_model", "full_resume"])

            mark_snapshot_evaluated(
                snapshots[0],
                [{"probe": "coco", "status": "complete"}],
                run_directory=run_directory,
            )
            updated = json.loads(snapshots[0].metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["provenance"]["evaluation"]["status"], "complete")
            index_event = json.loads(
                (run_directory / "checkpoint_index.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(index_event["event"], "trajectory_evaluation_completed")

    def test_refuses_model_only_snapshot_with_unexpected_backend_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pt"
            torch.save(
                {
                    "schema_version": 2,
                    "checkpoint_kind": "trajectory_model",
                    "run_id": "clip_standard",
                    "config_sha256": "config-sha",
                    "model": {"unexpected.weight": torch.ones(1)},
                },
                path,
            )
            snapshot = type(
                "Snapshot",
                (),
                {
                    "checkpoint_path": path,
                    "artifact_sha256": sha256_file(path),
                    "label": "p050",
                    "checkpoint_kind": "trajectory_model",
                },
            )()
            manifest = {
                "config_sha256": "config-sha",
                "config": {"run_id": "clip_standard"},
            }
            with self.assertRaisesRegex(ValueError, "outside the backend model namespace"):
                load_snapshot_into_adapter(FakeAdapter(), snapshot, manifest)


if __name__ == "__main__":
    unittest.main()
