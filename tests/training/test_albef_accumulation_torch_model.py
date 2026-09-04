import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.albef_accumulation import DeferredAlbefStateUpdates  # noqa: E402


class FakeTorchAlbefModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.momentum_update_calls = 0
        self.enqueue_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _momentum_update(self) -> None:
        self.momentum_update_calls += 1

    def _dequeue_and_enqueue(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> None:
        self.enqueue_calls.append((image_features.clone(), text_features.clone()))


class DeferredAlbefTorchModelTest(unittest.TestCase):
    def test_nn_module_methods_are_intercepted_and_restored(self):
        model = FakeTorchAlbefModel()
        controller = DeferredAlbefStateUpdates()
        controller.begin(model, expected_microbatches=2)

        for index in range(2):
            with controller.intercept_forward():
                model._momentum_update()
                feature = torch.tensor([[float(index)]])
                model._dequeue_and_enqueue(feature, feature + 10)
        controller.flush(lambda chunks: torch.cat(chunks, dim=0))

        self.assertEqual(model.momentum_update_calls, 1)
        self.assertEqual(len(model.enqueue_calls), 1)
        self.assertTrue(
            torch.equal(model.enqueue_calls[0][0], torch.tensor([[0.0], [1.0]]))
        )
        self.assertTrue(
            torch.equal(model.enqueue_calls[0][1], torch.tensor([[10.0], [11.0]]))
        )

        model._momentum_update()
        self.assertEqual(model.momentum_update_calls, 2)


if __name__ == "__main__":
    unittest.main()
