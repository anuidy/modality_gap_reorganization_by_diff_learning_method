import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.config import RunConfig
from training.engine import run_training, sha256_file
from evaluation.trajectory import resolve_trajectory_snapshots
from tests.training.test_formal_runtime import SmallEncoder


def fixture(root: Path) -> RunConfig:
    image_root = root / "images"
    image_root.mkdir()
    rows = []
    for index in range(12):
        Image.new("RGB", (4, 4), (index, 0, 0)).save(image_root / f"{index}.png")
        rows.append({"sample_id": str(index), "semantic_id": str(index),
                     "image": f"{index}.png", "text": str(index)})
    train, validation, checkpoint = root / "train.json", root / "validation.json", root / "m0.pt"
    train.write_text(json.dumps({"samples": rows}))
    validation.write_text(json.dumps({"samples": rows[:7]}))
    checkpoint.write_bytes(b"toy initialization identity")
    return RunConfig(
        run_id="clip_mixed_3m_fn_off", model_name="clip", branch="mixed_3m_fn_off",
        checkpoint=checkpoint, checkpoint_sha256=sha256_file(checkpoint), resources={},
        train_manifest=train, image_root=image_root, output_dir=root / "trajectory_only",
        seed=42, deterministic=True, precision="fp32", num_workers=0,
        micro_batch_size=6, gradient_accumulation=1, optimizer_type="adamw",
        learning_rate=1e-3, weight_decay=.05, beta1=.9, beta2=.999, epsilon=1e-8,
        scheduler_type="cosine", warmup_steps=3, min_lr_ratio=.1, max_steps=100,
        trajectory_progress_fractions=(.01, .05, .20, .50, 1.), resume_progress_interval=.2,
        resume_retention=2, log_interval=1, gradient_clip_norm=None,
        augmentation={}, model_options={}, save_resume_checkpoints=False,
        validation_progress_interval=.2, validation_manifest=validation,
        validation_sample_count=7, validation_manifest_sha256=sha256_file(validation),
    )


class TrajectoryOnlyTest(unittest.TestCase):
    def run_mocked(self, config, **kwargs):
        with patch("training.engine.create_training_backend", side_effect=lambda **kw: SmallEncoder()), \
             patch("training.engine.validate_formal_data_identity", return_value={"mode": "test"}), \
             contextlib.redirect_stdout(io.StringIO()):
            return run_training(config, "cpu", **kwargs)

    def test_gate_writes_three_model_snapshots_and_no_recovery_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            with patch("training.engine._save_full_resume_checkpoint", side_effect=AssertionError("Full state must not be saved")):
                output = self.run_mocked(config, stop_after_step=20)
            paths = sorted((config.output_dir / "checkpoints/trajectory").glob("*.pt"))
            self.assertEqual([p.name for p in paths], [
                "step_00000001_p001_model.pt", "step_00000005_p005_model.pt", "step_00000020_p020_model.pt",
            ])
            for path in paths:
                payload = torch.load(path, weights_only=False)
                self.assertEqual(payload["checkpoint_kind"], "trajectory_model")
                self.assertFalse({"optimizer", "rng", "stream"} & set(payload))
            self.assertEqual(output, paths[-1])
            self.assertFalse((config.output_dir / "checkpoints/resume").exists())
            self.assertFalse((config.output_dir / "checkpoints/final.json").exists())
            manifest = json.loads((config.output_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "paused")
            self.assertEqual(manifest["config"]["max_steps"], 100)
            self.assertEqual(manifest["completed_steps"], 20)
            self.assertEqual(manifest["checkpoint_policy"]["resume_steps"], [])
            self.assertFalse(manifest["exact_resume_supported"])
            self.assertNotIn("resume_checkpoint", manifest)
            with self.assertRaisesRegex(ValueError, "Exact resume is disabled"):
                self.run_mocked(config, resume_checkpoint=output)

    def test_model_only_preserves_updates_and_validation_without_saved_resume_files(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            model_path = self.run_mocked(config)
            full_config = replace(config, output_dir=Path(directory) / "with_full_state", save_resume_checkpoints=True)
            full_path = self.run_mocked(full_config)
            actual = torch.load(model_path, weights_only=False)
            reference = torch.load(full_path, weights_only=False)
            for name, tensor in actual["model"].items():
                torch.testing.assert_close(tensor, reference["model"][name], rtol=0, atol=0)
            manifest = json.loads((config.output_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            final = json.loads((config.output_dir / "checkpoints/final.json").read_text())
            self.assertEqual(final["checkpoint_kind"], "final_trajectory_model")
            snapshots = resolve_trajectory_snapshots(config.output_dir, manifest)
            self.assertEqual(len(snapshots), 5)
            self.assertTrue(all(s.checkpoint_kind == "trajectory_model" for s in snapshots))
            self.assertEqual(snapshots[-1].checkpoint_path, model_path)
            self.assertFalse((config.output_dir / "checkpoints/resume").exists())
            validation = [json.loads(line) for line in (config.output_dir / "validation_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([v["completed_steps"] for v in validation], [20, 40, 60, 80, 100])
            self.assertTrue(all(v["checkpoint"] is None for v in validation if v["completed_steps"] in [40, 60, 80]))
            self.assertTrue(all(v["sample_count"] == 6 and v["dropped_sample_count"] == 1 for v in validation))


if __name__ == "__main__":
    unittest.main()
