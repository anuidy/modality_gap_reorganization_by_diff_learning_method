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
            semantic_ids=tuple(str(i) for i in range(6)),
            images=torch.randn(6, 8),
            text_tokens=torch.randn(6, 8),
        )

    def test_mixed_encodes_all_modalities_for_full_batch_candidates(self):
        backend = fake_backend()
        result = backend(self.batch, "mixed_3m_fn_off", 1)
        self.assertEqual(result.audit.relation, "I<->T+I<->IT+T<->IT")
        self.assertEqual(backend.model.calls, {"I": 1, "T": 1, "IT": 1})

    def test_fixed_2m_omits_joint_forward(self):
        backend = fake_backend()
        result = backend(self.batch, "fixed_2m", 2)
        self.assertEqual(result.audit.relation, "I<->T")
        self.assertEqual(backend.model.calls, {"I": 1, "T": 1, "IT": 0})


if __name__ == "__main__":
    unittest.main()
