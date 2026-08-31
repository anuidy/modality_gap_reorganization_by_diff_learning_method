import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectives.contrastive import (  # noqa: E402
    RepresentationBatch,
    additive_multimodal_embedding,
    count_matched_mixed_objective,
    relation_for_optimizer_step,
    standard_objective,
)


class CountMatchedMixedObjectiveTest(unittest.TestCase):
    def test_relation_cycle_advances_by_optimizer_step(self):
        self.assertEqual(
            [relation_for_optimizer_step(step) for step in range(7)],
            ["I<->T", "I<->IT", "T<->IT", "I<->T", "I<->IT", "T<->IT", "I<->T"],
        )

    def test_each_relation_has_identical_supervision_and_candidates(self):
        batch_size = 5
        image = torch.nn.functional.normalize(torch.randn(batch_size, 8), dim=-1)
        text = torch.nn.functional.normalize(torch.randn(batch_size, 8), dim=-1)
        batch = RepresentationBatch(
            semantic_ids=[f"sample-{index}" for index in range(batch_size)],
            representations={"I": image, "T": text, "IT": additive_multimodal_embedding(image, text)},
        )

        for relation in ("I<->T", "I<->IT", "T<->IT"):
            with self.subTest(relation=relation):
                audit = count_matched_mixed_objective(10.0, relation)(batch).audit
                self.assertEqual(audit.relation, relation)
                self.assertEqual(audit.positive_terms, 2 * batch_size)
                self.assertEqual(audit.candidates_per_query, batch_size)
                self.assertEqual(audit.negatives_per_query, batch_size - 1)

    def test_duplicate_semantic_ids_are_rejected_as_false_negatives(self):
        batch = RepresentationBatch(
            semantic_ids=("same", "same"),
            representations={"I": torch.randn(2, 4), "T": torch.randn(2, 4)},
        )
        with self.assertRaisesRegex(ValueError, "unique semantic_ids"):
            standard_objective(10.0)(batch)

    def test_additive_embedding_is_unit_normalized(self):
        image = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
        text = torch.tensor([[0.0, 4.0], [5.0, 0.0]])
        multimodal = additive_multimodal_embedding(image, text)
        torch.testing.assert_close(multimodal.norm(dim=-1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
