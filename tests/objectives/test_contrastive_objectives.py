import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectives.contrastive import RepresentationBatch, standard_objective


class ContrastiveObjectiveTest(unittest.TestCase):
    def test_standard_reports_count_matched_supervision_and_backpropagates(self):
        batch_size = 4
        image = torch.randn(batch_size, 8, requires_grad=True)
        text = torch.randn(batch_size, 8, requires_grad=True)
        batch = RepresentationBatch(
            semantic_ids=torch.arange(batch_size),
            representations={"I": image, "T": text},
        )

        result = standard_objective(logit_scale=10.0)(batch)

        self.assertEqual(result.audit.relation, "I<->T")
        self.assertEqual(result.audit.positive_terms, 2 * batch_size)
        self.assertEqual(result.audit.negatives_per_query, batch_size - 1)
        result.loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertIsNotNone(text.grad)


if __name__ == "__main__":
    unittest.main()
