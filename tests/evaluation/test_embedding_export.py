import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import ProbeManifest, ProbeSample  # noqa: E402
from evaluation.embedding_export import export_raw_embeddings  # noqa: E402
from models.base import EmbeddingAdapter  # noqa: E402


class FakeAdapter(EmbeddingAdapter):
    def encode_image(self, images):
        return torch.tensor([[float(image.size[0]), 1.0] for image in images], dtype=torch.float32)

    def encode_text(self, texts):
        return torch.tensor([[float(len(text)), 2.0] for text in texts], dtype=torch.float32)

    def metadata(self):
        return {"model_name": "fake"}


class EmbeddingExportTest(unittest.TestCase):
    def test_exports_raw_float32_embeddings_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            for index, width in enumerate((3, 5, 7)):
                image_path = root / f"{index}.png"
                Image.new("RGB", (width, 2)).save(image_path)
                samples.append(
                    ProbeSample(
                        sample_id=f"sample-{index}",
                        semantic_id=f"semantic-{index}",
                        image_path=image_path,
                        text="x" * (index + 1),
                    )
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            manifest = ProbeManifest(
                name="fake_probe",
                path=manifest_path,
                sha256="manifest",
                samples=tuple(samples),
                metadata={},
            )
            progress = []

            image, text = export_raw_embeddings(
                FakeAdapter(), manifest, batch_size=2, progress=lambda completed, total: progress.append((completed, total))
            )

        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(text.dtype, np.float32)
        np.testing.assert_array_equal(image[:, 0], np.asarray([3.0, 5.0, 7.0]))
        np.testing.assert_array_equal(text[:, 0], np.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(progress, [(2, 3), (3, 3)])


if __name__ == "__main__":
    unittest.main()
