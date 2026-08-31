import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.backends import PreparedBatch, TrainingBackend, VistaTrainingBackend  # noqa: E402


class FakeVistaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.02
        self.calls = {"I": 0, "T": 0, "IT": 0}

    def encode_image(self, images):
        self.calls["I"] += 1
        return images

    def encode_text(self, tokens):
        self.calls["T"] += 1
        return tokens

    def encode_mm(self, images, tokens):
        self.calls["IT"] += 1
        return images + tokens


def fake_backend() -> VistaTrainingBackend:
    backend = VistaTrainingBackend.__new__(VistaTrainingBackend)
    TrainingBackend.__init__(backend, torch.device("cpu"))
    backend.model = FakeVistaModel()
    return backend


class VistaRoutingTest(unittest.TestCase):
    def setUp(self):
        self.batch = PreparedBatch(
            semantic_ids=("0", "1", "2", "3"),
            images=torch.randn(4, 8),
            text_tokens=torch.randn(4, 8),
        )

    def test_i_to_it_omits_unused_standalone_text_forward(self):
        backend = fake_backend()
        result = backend(self.batch, "count_matched_mixed", 1)
        self.assertEqual(result.audit.relation, "I<->IT")
        self.assertEqual(backend.model.calls, {"I": 1, "T": 0, "IT": 1})

    def test_t_to_it_omits_unused_standalone_image_forward(self):
        backend = fake_backend()
        result = backend(self.batch, "count_matched_mixed", 2)
        self.assertEqual(result.audit.relation, "T<->IT")
        self.assertEqual(backend.model.calls, {"I": 0, "T": 1, "IT": 1})


if __name__ == "__main__":
    unittest.main()
