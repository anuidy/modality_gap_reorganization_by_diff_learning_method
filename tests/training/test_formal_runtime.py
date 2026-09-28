import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from dataclasses import replace

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.backends import TrainingBackend, PreparedBatch, _relation_step, _circular_albef_enqueue
from training.config import RunConfig, load_run_config, EXPECTED_RUNS
from training.engine import run_training, sha256_file
from training.independent_runs import build_jobs


class SmallEncoder(TrainingBackend):
    model_name = "clip"

    def __init__(self):
        super().__init__(torch.device("cpu"))
        self.image = torch.nn.Linear(8, 8)
        self.text = torch.nn.Linear(8, 8)
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def prepare_batch(self, batch, augmentation_seed):
        generator = torch.Generator().manual_seed(augmentation_seed)
        views = torch.rand(len(batch.semantic_ids), 8, generator=generator)
        return PreparedBatch(batch.semantic_ids, views, views.roll(1, dims=1))

    def forward(self, batch, branch, optimizer_step):
        image = torch.nn.functional.dropout(self.image(batch.images), 0.1, training=self.training)
        text = self.text(batch.text_tokens)
        return _relation_step(batch.semantic_ids, image, text, self.scale.exp(), branch,
                              optimizer_step, None, master_seed=self.random_seed)

    def validation_metrics(self, batch, branch, optimizer_step):
        return {"common/loss": self.forward(batch, "standard", optimizer_step).loss}


class FormalRuntimeTest(unittest.TestCase):
    def test_circular_albef_queue_preserves_fifo_across_wrap(self):
        model = SimpleNamespace(image_queue=torch.zeros(2, 8), text_queue=torch.zeros(2, 8),
                                queue_ptr=torch.tensor([6]))
        images = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        texts = images + 100
        _circular_albef_enqueue(model, images, texts)
        self.assertEqual(model.queue_ptr.item(), 3)
        for index, slot in enumerate([6, 7, 0, 1, 2]):
            torch.testing.assert_close(model.image_queue[:, slot], images[index])
            torch.testing.assert_close(model.text_queue[:, slot], texts[index])
        self.assertTrue(torch.equal(model.image_queue[:, 3:6], torch.zeros(2, 3)))
        self.assertFalse(model.image_queue.requires_grad)

    def test_all_formal_configs_and_seed_output_paths(self):
        path = ROOT / "configs/training/train_runs.yaml"
        self.assertEqual(len(EXPECTED_RUNS), 29)
        for run in EXPECTED_RUNS:
            config = load_run_config(path, run, ROOT, {"seed": 42})
            self.assertEqual(config.micro_batch_size, 36)
            self.assertEqual(config.gradient_accumulation, 1)
            self.assertIsNone(config.gradient_clip_norm)
            self.assertFalse(config.save_resume_checkpoints)
            self.assertEqual(config.validation_progress_interval, .2)
            self.assertEqual(config.max_steps, 15003)
            self.assertEqual(config.output_dir.name, "seed_42")
            self.assertEqual(config.run_id, run + "_seed_42")
        jobs = build_jobs(path, ["clip_standard"], ROOT, [42, 43])
        self.assertNotEqual(jobs[0].output_dir, jobs[1].output_dir)
        self.assertIn("--seed", jobs[0].command)

    def test_mixed_updates_without_clipping_resume_exactly_and_validation_drops_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_root = root / "images"
            image_root.mkdir()
            rows = []
            for index in range(12):
                Image.new("RGB", (4, 4), (index, 0, 0)).save(image_root / f"{index}.png")
                rows.append({"sample_id": str(index), "semantic_id": str(index),
                             "image": f"{index}.png", "text": str(index)})
            train = root / "train.json"
            validation = root / "validation.json"
            train.write_text(json.dumps({"samples": rows}))
            validation.write_text(json.dumps({"samples": rows[:7]}))
            checkpoint = root / "m0.pt"
            checkpoint.write_bytes(b"toy initialization identity")
            base = RunConfig(
                run_id="clip_mixed_3m_fn_off", model_name="clip", branch="mixed_3m_fn_off",
                checkpoint=checkpoint, checkpoint_sha256=sha256_file(checkpoint), resources={},
                train_manifest=train, image_root=image_root, output_dir=root / "continuous",
                seed=42, deterministic=True, precision="fp32", num_workers=0,
                micro_batch_size=6, gradient_accumulation=1, optimizer_type="adamw",
                learning_rate=1e-3, weight_decay=.05, beta1=.9, beta2=.999, epsilon=1e-8,
                scheduler_type="cosine", warmup_steps=1, min_lr_ratio=.1, max_steps=4,
                trajectory_progress_fractions=(.25, .5, 1.), resume_progress_interval=.5,
                resume_retention=2, log_interval=1, gradient_clip_norm=None,
                augmentation={}, model_options={}, validation_manifest=validation,
                validation_sample_count=7, validation_manifest_sha256=sha256_file(validation),
            )
            split = replace(base, output_dir=root / "resumed")
            with patch("training.engine.create_training_backend", side_effect=lambda **kw: SmallEncoder()), \
                 patch("training.engine.validate_formal_data_identity", return_value={"mode": "test"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                whole = run_training(base, "cpu")
                paused = run_training(split, "cpu", stop_after_step=2)
                state = json.loads((split.output_dir / "run_manifest.json").read_text())
                self.assertEqual(state["status"], "paused")
                self.assertEqual(state["config"]["max_steps"], 4)
                self.assertFalse((split.output_dir / "checkpoints/final.json").exists())
                resumed = run_training(split, "cpu", resume_checkpoint=paused)
            left = torch.load(whole, weights_only=False)
            right = torch.load(resumed, weights_only=False)
            for name in left["model"]:
                torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
            logs = [json.loads(x) for x in (split.output_dir / "train_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([x["completed_steps"] for x in logs], [1, 2, 3, 4])
            self.assertTrue(all(x["gradient_norm"] > 0 for x in logs))
            self.assertTrue(all(x["relation_audit"]["positive_terms_per_optimizer_step"] == 12 for x in logs))
            val = [json.loads(x) for x in (split.output_dir / "validation_metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(x["sample_count"] == 6 and x["dropped_sample_count"] == 1 for x in val))
