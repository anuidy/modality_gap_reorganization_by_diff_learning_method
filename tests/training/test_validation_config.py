import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import load_run_config  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "train_runs.yaml"


class ValidationConfigTest(unittest.TestCase):
    def test_micro_batch_must_divide_locked_validation_count(self):
        overrides = {
            "seed": 7,
            "micro_batch_size": 128,
            "gradient_accumulation": 1,
            "optimizer_type": "adamw",
            "learning_rate": 1e-5,
            "weight_decay": 0.02,
            "scheduler_type": "cosine",
            "warmup_steps": 10,
            "min_lr_ratio": 0.1,
            "max_steps": 100,
            "augmentation_name": "random_resized_crop",
            "augmentation_scale_min": 0.8,
            "augmentation_scale_max": 1.0,
            "augmentation_hflip": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "divide validation_sample_count"):
            load_run_config(CONFIG_PATH, "clip_standard", PROJECT_ROOT, overrides)


if __name__ == "__main__":
    unittest.main()
