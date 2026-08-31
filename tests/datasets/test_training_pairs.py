import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.training_pairs import (  # noqa: E402
    DeterministicEpochSampler,
    PairedTrainingDataset,
    collate_raw_training_batch,
    load_training_pairs,
)


class TrainingPairsTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        image_root = root / "images"
        image_root.mkdir()
        records = []
        for index in range(4):
            Image.new("RGB", (8, 8), color=(index, 0, 0)).save(image_root / f"{index}.png")
            records.append(
                {
                    "sample_id": f"sample-{index}",
                    "semantic_id": f"semantic-{index}",
                    "image": f"{index}.png",
                    "text": f"caption {index}",
                }
            )
        manifest = root / "train.json"
        manifest.write_text(json.dumps({"samples": records}), encoding="utf-8")
        return manifest, image_root

    def test_load_dataset_and_collate_rgb_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, image_root = self._fixture(Path(temporary))
            dataset = PairedTrainingDataset(load_training_pairs(manifest, image_root))
            batch = collate_raw_training_batch([dataset[0], dataset[1]])
            self.assertEqual(batch.semantic_ids, ("semantic-0", "semantic-1"))
            self.assertTrue(all(image.mode == "RGB" for image in batch.images))

    def test_duplicate_semantic_instances_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, image_root = self._fixture(Path(temporary))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["samples"][1]["semantic_id"] = payload["samples"][0]["semantic_id"]
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate semantic_id"):
                load_training_pairs(manifest, image_root)

    def test_image_path_cannot_escape_image_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, image_root = self._fixture(Path(temporary))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["samples"][0]["image"] = "../outside.png"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes image_root"):
                load_training_pairs(manifest, image_root)

    def test_epoch_sampler_is_reproducible_and_changes_by_epoch(self):
        source = list(range(20))
        left = DeterministicEpochSampler(source, seed=123)
        right = DeterministicEpochSampler(source, seed=123)
        self.assertEqual(list(left), list(right))
        left.set_epoch(1)
        right.set_epoch(1)
        self.assertEqual(list(left), list(right))
        self.assertNotEqual(list(DeterministicEpochSampler(source, seed=123)), list(left))


if __name__ == "__main__":
    unittest.main()
