import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.optim import build_weight_decay_parameter_groups  # noqa: E402


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(3)
        self.pos_embed = nn.Parameter(torch.ones(2, 3))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.frozen = nn.Parameter(torch.ones(3, 3), requires_grad=False)


class OptimizerParameterGroupsTest(unittest.TestCase):
    def test_groups_are_complete_disjoint_and_apply_decay_selectively(self):
        model = ToyModel()
        groups = build_weight_decay_parameter_groups(
            model,
            weight_decay=0.2,
            no_weight_decay_names={"pos_embed"},
        )
        by_name = {str(group["group_name"]): group for group in groups}

        self.assertEqual(set(by_name), {"decay", "no_decay"})
        self.assertEqual(float(by_name["decay"]["weight_decay"]), 0.2)
        self.assertEqual(float(by_name["no_decay"]["weight_decay"]), 0.0)

        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        decay_names = {parameter_names[id(parameter)] for parameter in by_name["decay"]["params"]}
        no_decay_names = {
            parameter_names[id(parameter)] for parameter in by_name["no_decay"]["params"]
        }

        self.assertEqual(decay_names, {"projection.weight"})
        self.assertEqual(
            no_decay_names,
            {
                "projection.bias",
                "norm.weight",
                "norm.bias",
                "pos_embed",
                "logit_scale",
            },
        )
        self.assertTrue(decay_names.isdisjoint(no_decay_names))
        self.assertNotIn("frozen", decay_names | no_decay_names)
        self.assertEqual(
            decay_names | no_decay_names,
            {name for name, parameter in model.named_parameters() if parameter.requires_grad},
        )

    def test_rejects_negative_weight_decay(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_weight_decay_parameter_groups(ToyModel(), weight_decay=-0.1)


if __name__ == "__main__":
    unittest.main()
