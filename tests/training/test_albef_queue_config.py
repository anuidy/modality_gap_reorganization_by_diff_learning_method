from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import load_run_config  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "train_runs.yaml"


def overrides(gradient_accumulation: int) -> dict[str, object]:
    return {
        "train_manifest": "data/train.jsonl",
        "image_root": "data/images",
        "seed": 7,
        "micro_batch_size": 64,
        "gradient_accumulation": gradient_accumulation,
        "optimizer_type": "adamw",
        "learning_rate": 1e-5,
        "weight_decay": 0.02,
        "scheduler_type": "cosine",
        "warmup_steps": 10,
        "min_lr_ratio": 0.1,
        "max_steps": 100,
        "augmentation_name": "random_resized_crop",
        "augmentation_scale_min": 0.2,
        "augmentation_scale_max": 1.0,
        "augmentation_hflip": 0.5,
    }


class AlbefQueueConfigTest(unittest.TestCase):
    def test_effective_batch_must_divide_native_queue_size(self):
        with self.assertRaisesRegex(ValueError, "effective_batch_size"):
            load_run_config(
                CONFIG_PATH,
                "albef_itc_only",
                PROJECT_ROOT,
                overrides(gradient_accumulation=3),
            )

    def test_effective_batch_that_divides_queue_is_accepted(self):
        config = load_run_config(
            CONFIG_PATH,
            "albef_itc_only",
            PROJECT_ROOT,
            overrides(gradient_accumulation=8),
        )
        self.assertEqual(config.effective_batch_size, 512)


if __name__ == "__main__":
    unittest.main()
