import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.training_pairs import RawTrainingBatch  # noqa: E402
from training.backends import PreparedBatch, TrainingBackend, TrainingStepResult  # noqa: E402
from training.config import RunConfig  # noqa: E402
from training.engine import run_training, sha256_file  # noqa: E402


class ValidationIntegrationBackend(TrainingBackend):
    model_name = "fake"

    def __init__(self):
        super().__init__(torch.device("cpu"))
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.validation_calls = 0

    def prepare_batch(self, batch: RawTrainingBatch, augmentation_seed: int) -> PreparedBatch:
        del augmentation_seed
        return PreparedBatch(
            semantic_ids=batch.semantic_ids,
            images=torch.ones(len(batch.semantic_ids), 1),
            text_tokens=None,
        )

    def forward(self, batch, branch, optimizer_step):
        del batch, branch, optimizer_step
        loss = self.weight.square()
        return TrainingStepResult(loss=loss, metrics={"loss": loss}, audit=None)

    def validation_metrics(self, batch, branch, optimizer_step):
        del batch, branch, optimizer_step
        self.validation_calls += 1
        return {"common/loss": self.weight.square()}


class ValidationEngineIntegrationTest(unittest.TestCase):
    def test_checkpoint_interval_and_final_step_write_read_only_validation_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "m0.pt"
            checkpoint.write_bytes(b"fake")
            image_root = root / "images"
            image_root.mkdir()
            records = []
            for index in range(4):
                Image.new("RGB", (4, 4)).save(image_root / f"{index}.png")
                records.append(
                    {
                        "sample_id": str(index),
                        "semantic_id": str(index),
                        "image": f"{index}.png",
                        "text": f"caption {index}",
                    }
                )
            train_manifest = root / "train.json"
            validation_manifest = root / "validation.json"
            payload = json.dumps({"samples": records})
            train_manifest.write_text(payload, encoding="utf-8")
            validation_manifest.write_text(payload, encoding="utf-8")
            config = RunConfig(
                run_id="clip_standard",
                model_name="clip",
                branch="standard",
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                resources={},
                train_manifest=train_manifest,
                image_root=image_root,
                output_dir=root / "output",
                seed=7,
                deterministic=True,
                precision="fp32",
                num_workers=0,
                micro_batch_size=2,
                gradient_accumulation=1,
                optimizer_type="adamw",
                learning_rate=0.01,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                scheduler_type="cosine",
                warmup_steps=1,
                min_lr_ratio=0.1,
                max_steps=3,
                checkpoint_interval=2,
                log_interval=1,
                gradient_clip_norm=1.0,
                augmentation={},
                model_options={},
                validation_manifest=validation_manifest,
                validation_manifest_sha256=sha256_file(validation_manifest),
            )
            backend = ValidationIntegrationBackend()
            with (
                mock.patch("training.engine.create_training_backend", return_value=backend),
                mock.patch(
                    "training.engine.validate_formal_data_identity",
                    return_value={"mode": "test"},
                ),
            ):
                run_training(config, device_name="cpu")

            records = [
                json.loads(line)
                for line in (config.output_dir / "validation_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([record["completed_steps"] for record in records], [2, 3])
            self.assertTrue(all(record["sample_count"] == 4 for record in records))
            self.assertTrue(
                all(record["policy"]["checkpoint_selection"] is False for record in records)
            )
            self.assertEqual(backend.validation_calls, 4)


if __name__ == "__main__":
    unittest.main()
