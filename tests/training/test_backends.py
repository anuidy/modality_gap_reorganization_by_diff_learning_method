import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.backends import _relation_step  # noqa: E402


class RelationBackendTest(unittest.TestCase):
    def test_vista_joint_encoder_runs_only_for_relations_that_need_it(self):
        calls = 0

        def joint_encoder():
            nonlocal calls
            calls += 1
            return torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)

        image = torch.randn(6, 8, requires_grad=True)
        text = torch.randn(6, 8, requires_grad=True)
        semantic_ids = tuple(f"sample-{index}" for index in range(6))

        first = _relation_step(
            semantic_ids, image, text, 10.0, "fixed_2m", 0, joint_encoder
        )
        self.assertEqual(first.audit.relation, "I<->T")
        self.assertEqual(calls, 0)
        second = _relation_step(
            semantic_ids, image, text, 10.0, "mixed_3m_fn_off", 1, joint_encoder
        )
        self.assertEqual(second.audit.relation, "I<->T+I<->IT+T<->IT")
        self.assertEqual(calls, 1)

    def test_learnable_scale_keeps_gradient_in_the_objective(self):
        raw_scale = torch.tensor(4.7, requires_grad=True)
        image = torch.randn(6, 8, requires_grad=True)
        text = torch.randn(6, 8, requires_grad=True)
        result = _relation_step(
            tuple(str(index) for index in range(6)),
            image,
            text,
            raw_scale.exp(),
            "standard",
            0,
            None,
        )
        result.loss.backward()
        self.assertIsNotNone(raw_scale.grad)
        self.assertNotEqual(float(raw_scale.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
