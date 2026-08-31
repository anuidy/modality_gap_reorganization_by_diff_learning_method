import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectives.contrastive import RelationAudit, relation_for_optimizer_step  # noqa: E402
from training.backends import PreparedBatch, TrainingBackend, TrainingStepResult  # noqa: E402
from training.config import RunConfig  # noqa: E402
from training.engine import run_training, sha256_file  # noqa: E402


class FakeBackend(TrainingBackend):
    model_name = "fake"

    def __init__(self):
        super().__init__(torch.device("cpu"))
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def prepare_batch(self, batch, augmentation_seed):
        del augmentation_seed
        return PreparedBatch(
            semantic_ids=batch.semantic_ids,
            images=torch.ones(len(batch.semantic_ids), 1),
            text_tokens=None,
        )

    def forward(self, batch, branch, optimizer_step):
        self.assert_branch = branch
        loss = self.weight.square()
        relation = relation_for_optimizer_step(optimizer_step)
        batch_size = len(batch.semantic_ids)
        return TrainingStepResult(
            loss=loss,
            metrics={"loss": loss},
            audit=RelationAudit(
                relation=relation,
                batch_size=batch_size,
                positive_terms=2 * batch_size,
                candidates_per_query=batch_size,
                negatives_per_query=batch_size - 1,
            ),
        )


class TrainingEngineTest(unittest.TestCase):
    def test_gradient_accumulation_keeps_one_relation_per_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "m0.pt"
            checkpoint.write_bytes(b"fake-m0")
            image_root = root / "images"
            image_root.mkdir()
            samples = []
            for index in range(4):
                Image.new("RGB", (4, 4)).save(image_root / f"{index}.png")
                samples.append(
                    {
                        "sample_id": str(index),
                        "semantic_id": str(index),
                        "image": f"{index}.png",
                        "text": f"caption {index}",
                    }
                )
            manifest = root / "train.json"
            manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
            config = RunConfig(
                run_id="clip_count_matched_mixed",
                model_name="clip",
                branch="count_matched_mixed",
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                resources={},
                train_manifest=manifest,
                image_root=image_root,
                output_dir=root / "output",
                seed=123,
                deterministic=True,
                precision="fp32",
                num_workers=0,
                micro_batch_size=2,
                gradient_accumulation=2,
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
            )
            backend = FakeBackend()
            with mock.patch("training.engine.create_training_backend", return_value=backend):
                final_checkpoint = run_training(config, device_name="cpu")

            self.assertTrue(final_checkpoint.is_file())
            records = [
                json.loads(line)
                for line in (config.output_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["relation_audit"]["relation"] for record in records],
                ["I<->T", "I<->IT", "T<->IT"],
            )
            self.assertTrue(
                all(
                    record["relation_audit"]["positive_terms_per_optimizer_step"] == 8
                    for record in records
                )
            )
            self.assertLess(float(backend.weight.detach()), 1.0)

            periodic_checkpoint = config.output_dir / "checkpoints" / "step_00000002.pt"
            with mock.patch(
                "training.engine.create_training_backend", return_value=FakeBackend()
            ):
                resumed_checkpoint = run_training(
                    config, device_name="cpu", resume_checkpoint=periodic_checkpoint
                )
            self.assertTrue(resumed_checkpoint.is_file())

            changed_config = replace(config, learning_rate=0.02)
            with mock.patch(
                "training.engine.create_training_backend", return_value=FakeBackend()
            ):
                with self.assertRaisesRegex(ValueError, "different controlled configuration"):
                    run_training(
                        changed_config,
                        device_name="cpu",
                        resume_checkpoint=periodic_checkpoint,
                    )


if __name__ == "__main__":
    unittest.main()
