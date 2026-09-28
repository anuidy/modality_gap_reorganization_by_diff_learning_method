import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.training_pairs import RawTrainingBatch  # noqa: E402
from training.backends import PreparedBatch, TrainingBackend, TrainingStepResult  # noqa: E402
from training.validation import relation_validation_metrics, run_validation  # noqa: E402


class ValidationFakeBackend(TrainingBackend):
    model_name = "fake"

    def __init__(self):
        super().__init__(torch.device("cpu"))
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def prepare_batch(self, batch, augmentation_seed):
        del augmentation_seed
        return PreparedBatch(
            semantic_ids=batch.semantic_ids,
            images=torch.ones(len(batch.semantic_ids), 1),
            text_tokens=None,
        )

    def forward(self, batch, branch, optimizer_step):
        del batch, branch, optimizer_step
        return TrainingStepResult(self.weight.square(), {}, None)

    def validation_metrics(self, batch, branch, optimizer_step):
        del branch, optimizer_step
        return {
            "common/loss": self.weight + torch.rand((), device=batch.images.device),
        }


class ValidationTest(unittest.TestCase):
    def test_mixed_reports_common_exam_and_two_relation_diagnostics(self):
        semantic_ids = tuple(f"sample-{index}" for index in range(6))
        image = torch.randn(6, 8)
        text = torch.randn(6, 8)
        standard = relation_validation_metrics(
            semantic_ids=semantic_ids,
            image=image,
            text=text,
            logit_scale=10.0,
            branch="standard",
        )
        mixed = relation_validation_metrics(
            semantic_ids=semantic_ids,
            image=image,
            text=text,
            logit_scale=10.0,
            branch="mixed_3m_fn_off",
        )
        self.assertAlmostEqual(
            float(standard["common/I<->T/loss"]),
            float(mixed["common/I<->T/loss"]),
        )
        self.assertIn("diagnostic/mixed_3m_fn_off/loss", mixed)
        self.assertIn("diagnostic/mixed_3m_fn_off/T->IT", mixed)
        self.assertNotIn("diagnostic/mixed_3m_fn_off/loss", standard)

    def test_validation_restores_rng_mode_and_parameters(self):
        backend = ValidationFakeBackend()
        backend.train()
        before_weight = backend.weight.detach().clone()
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        expected_python = random.random()
        expected_numpy = float(np.random.rand())
        expected_torch = float(torch.rand(()))
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        batch = RawTrainingBatch(
            sample_ids=("a", "b"),
            semantic_ids=("a", "b"),
            images=(),
            texts=("a", "b"),
        )
        result = run_validation(
            backend=backend,
            loader=[batch],  # type: ignore[arg-type]
            branch="standard",
            optimizer_step=12,
            seed=101,
            device=torch.device("cpu"),
            precision="fp32",
        )
        self.assertTrue(backend.training)
        self.assertTrue(torch.equal(before_weight, backend.weight.detach()))
        self.assertEqual(result["sample_count"], 2)
        self.assertFalse(result["policy"]["checkpoint_selection"])
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.rand()), expected_numpy)
        self.assertEqual(float(torch.rand(())), expected_torch)


if __name__ == "__main__":
    unittest.main()
