import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.optim import build_weight_decay_parameter_groups  # noqa: E402


class OptimizerWeightDecayEffectTest(unittest.TestCase):
    def test_adamw_changes_only_the_decay_group_when_gradients_are_zero(self):
        model = nn.Linear(2, 2)
        before_weight = model.weight.detach().clone()
        before_bias = model.bias.detach().clone()
        groups = build_weight_decay_parameter_groups(model, weight_decay=0.2)
        optimizer = torch.optim.AdamW(groups, lr=0.1, weight_decay=0.0)

        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()

        self.assertTrue(torch.allclose(model.weight, before_weight * 0.98))
        self.assertTrue(torch.equal(model.bias, before_bias))


if __name__ == "__main__":
    unittest.main()
