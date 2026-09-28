import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from objectives.contrastive import (BRANCH_DEFINITIONS, RepresentationBatch, additive_multimodal_embedding,
    balanced_relation_groups, standard_objective, training_objective)
from training.randomness import stream_seed


class FormalObjectiveTest(unittest.TestCase):
    def batch(self, n=36):
        g = torch.Generator().manual_seed(91)
        image = (torch.randn(n, 8, generator=g) * 3).requires_grad_()
        text = torch.randn(n, 8, generator=g).requires_grad_()
        return RepresentationBatch(tuple(map(str, range(n))), {"I": image, "T": text, "IT": additive_multimodal_embedding(image, text)})

    def test_nine_branches_have_protocol_counts_and_finite_backward(self):
        expected = {
            "standard": (72, 36, 35), "fixed_2m": (72, 71, 70),
            "fixed_3m_fn_off": (72, 106, 105), "fixed_3m_fn_on": (72, 107, 106),
            "mixed_2m": (72, 71, 70), "mixed_3m_fn_off": (72, 106, 105),
            "mixed_3m_fn_on": (72, 107, 106),
            "full_3m_fn_off": (216, 106, 105), "full_3m_fn_on": (216, 107, 106),
        }
        self.assertEqual(set(expected), set(BRANCH_DEFINITIONS))
        for branch, counts in expected.items():
            with self.subTest(branch=branch):
                batch = self.batch()
                scale = torch.tensor(2.0, requires_grad=True)
                result = training_objective(scale.exp(), branch, group_seed=17)(batch)
                a = result.audit
                self.assertEqual((a.positive_terms, a.candidates_per_query, a.negatives_per_query), counts)
                if branch.startswith("mixed_"):
                    self.assertEqual(sorted(map(len, a.group_indices.values())), [12, 12, 12])
                    self.assertEqual(sorted(sum(a.group_indices.values(), [])), list(range(36)))
                    self.assertEqual(set(a.direction_query_counts.values()), {12})
                result.loss.backward()
                for tensor in [batch.representations["I"], batch.representations["T"], scale]:
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(float(tensor.grad.abs().sum()), 0)

    def test_raw_sum_keeps_magnitudes_and_is_not_normalized_early(self):
        image = torch.tensor([[3., 0.], [0., 2.]], requires_grad=True)
        text = torch.tensor([[0., 4.], [5., 0.]], requires_grad=True)
        joint = additive_multimodal_embedding(image, text)
        torch.testing.assert_close(joint, torch.tensor([[3., 4.], [5., 2.]]))
        self.assertFalse(torch.allclose(F.normalize(joint, dim=-1), F.normalize(F.normalize(image, dim=-1)+F.normalize(text, dim=-1),dim=-1)))
        joint.sum().backward()
        torch.testing.assert_close(image.grad, torch.ones_like(image))
        torch.testing.assert_close(text.grad, torch.ones_like(text))

    def test_directional_loss_matches_explicit_candidate_lists(self):
        batch = self.batch(6)
        z = {k: F.normalize(v.detach(), dim=-1) for k, v in batch.representations.items()}
        for branch in BRANCH_DEFINITIONS:
            result = training_objective(2.5, branch, group_seed=13)(batch)
            reference = []
            for direction, measured in result.directional_losses.items():
                query, target = direction.split("->")
                if branch.startswith("mixed_"):
                    relation = next(k for k in result.audit.group_indices if set(k.split("<->")) == {query, target})
                    query_ids = result.audit.group_indices[relation]
                else:
                    query_ids = range(6)
                modalities = [target] if branch == "standard" else ([query, target] if "2m" in branch else ["I", "T", "IT"])
                direction_reference = []
                for i in query_ids:
                    scores = []
                    for modality in modalities:
                        for j in range(6):
                            if j == i and modality == query:
                                continue
                            if j == i and branch.endswith("fn_off") and modality != target:
                                continue
                            scores.append(float(2.5 * torch.dot(z[query][i], z[modality][j])))
                    positive = float(2.5 * torch.dot(z[query][i], z[target][i]))
                    loss = math.log(sum(math.exp(s) for s in scores)) - positive
                    direction_reference.append(loss)
                self.assertAlmostEqual(float(measured.detach()), sum(direction_reference)/len(direction_reference), places=5)
                reference.extend(direction_reference)
            self.assertAlmostEqual(float(result.loss.detach()), sum(reference)/len(reference), places=5)

    def test_group_rng_does_not_change_global_random_sequence(self):
        torch.manual_seed(88)
        before = torch.random.get_rng_state().clone()
        seed = stream_seed(42, "group", 100)
        first = balanced_relation_groups(36, seed)
        balanced_relation_groups(36, stream_seed(42, "group", 101))
        second = balanced_relation_groups(36, seed)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))
        self.assertNotEqual(stream_seed(42, "group", 100), stream_seed(42, "augmentation", 100))

    def test_queries_outside_relation_group_still_receive_candidate_gradients(self):
        batch = self.batch()
        result = training_objective(2., "mixed_2m", group_seed=17)(batch)
        selected = set(result.audit.group_indices["I<->T"])
        outside = next(i for i in range(36) if i not in selected)
        result.directional_losses["I->T"].backward()
        self.assertGreater(float(batch.representations["T"].grad[outside].abs().sum()), 0)
        self.assertGreater(float(batch.representations["I"].grad[outside].abs().sum()), 0)

    def test_standard_is_unchanged_and_two_directions_are_distinct(self):
        batch = self.batch()
        result = training_objective(3., "standard")(batch)
        torch.testing.assert_close(result.loss, standard_objective(3.)(batch).loss)
        self.assertNotAlmostEqual(float(result.directional_losses["I->T"].detach()), float(result.directional_losses["T->I"].detach()), places=7)

    def test_rejects_nondivisible_mixed_batch_and_old_rotating_branch(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            training_objective(3., "mixed_3m_fn_off")(self.batch(8))
        with self.assertRaisesRegex(ValueError, "Unknown formal"):
            training_objective(3., "rotating")

    def test_duplicate_semantic_ids_are_rejected(self):
        batch = self.batch(6)
        duplicate = RepresentationBatch(("same",)*6, batch.representations)
        with self.assertRaisesRegex(ValueError, "unique semantic_ids"):
            training_objective(3., "mixed_3m_fn_off")(duplicate)


if __name__ == "__main__":
    unittest.main()
