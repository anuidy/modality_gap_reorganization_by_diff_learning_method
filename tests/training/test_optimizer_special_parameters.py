import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.optim import build_weight_decay_parameter_groups  # noqa: E402


class SpecialParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.regular_weight = nn.Parameter(torch.ones(2, 2))
        self.positional_embedding = nn.Parameter(torch.ones(4, 2))
        self.cls_token = nn.Parameter(torch.ones(1, 1, 2))


class OptimizerSpecialParametersTest(unittest.TestCase):
    def test_named_special_parameters_are_excluded_without_manual_overrides(self):
        model = SpecialParameterModel()
        groups = build_weight_decay_parameter_groups(model, weight_decay=0.2)
        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        group_names = {
            str(group["group_name"]): {
                parameter_names[id(parameter)] for parameter in group["params"]
            }
            for group in groups
        }

        self.assertEqual(group_names["decay"], {"regular_weight"})
        self.assertEqual(
            group_names["no_decay"],
            {"positional_embedding", "cls_token"},
        )


if __name__ == "__main__":
    unittest.main()
