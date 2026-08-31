import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.backends import PreparedBatch  # noqa: E402
from training.validation import _albef_itc_context  # noqa: E402


class FakeVision(torch.nn.Module):
    def forward(self, image):
        return torch.stack([image, image * 0.5], dim=1)


class FakeBert(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, input_ids, attention_mask, return_dict, mode):
        del attention_mask, return_dict, mode
        hidden = torch.nn.functional.one_hot(
            input_ids.remainder(self.dimension), num_classes=self.dimension
        ).float()
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.bert = FakeBert(dimension)


class FakeAlbefModel(torch.nn.Module):
    def __init__(self, dimension: int = 4):
        super().__init__()
        self.visual_encoder = FakeVision()
        self.visual_encoder_m = FakeVision()
        self.vision_proj = torch.nn.Identity()
        self.vision_proj_m = torch.nn.Identity()
        self.text_encoder = FakeTextEncoder(dimension)
        self.text_encoder_m = FakeTextEncoder(dimension)
        self.text_proj = torch.nn.Identity()
        self.text_proj_m = torch.nn.Identity()
        self.temp = torch.nn.Parameter(torch.tensor(0.07))
        self.register_buffer(
            "image_queue", torch.nn.functional.normalize(torch.randn(dimension, 8), dim=0)
        )
        self.register_buffer(
            "text_queue", torch.nn.functional.normalize(torch.randn(dimension, 8), dim=0)
        )
        self.register_buffer("queue_ptr", torch.tensor([3], dtype=torch.long))

    def _momentum_update(self):
        raise AssertionError("Validation must not update momentum encoders.")

    def _dequeue_and_enqueue(self, image, text):
        del image, text
        raise AssertionError("Validation must not mutate ALBEF queues.")


class AlbefValidationIsolationTest(unittest.TestCase):
    def test_native_itc_validation_does_not_mutate_temperature_or_queues(self):
        model = FakeAlbefModel()
        backend = SimpleNamespace(model=model)
        batch = PreparedBatch(
            semantic_ids=("a", "b"),
            images=torch.randn(2, 4),
            text_tokens=SimpleNamespace(
                input_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
                attention_mask=torch.ones(2, 2, dtype=torch.long),
            ),
        )
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}

        loss, _ = _albef_itc_context(backend, batch, alpha=0.4)  # type: ignore[arg-type]

        self.assertTrue(torch.isfinite(loss))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()
