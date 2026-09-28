import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import load_run_config  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "train_runs.yaml"


class ValidationConfigTest(unittest.TestCase):
    def test_validation_accepts_training_batch_and_drops_partial_tail(self):
        overrides = {
            "seed": 7,
            "micro_batch_size": 36,
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
        config = load_run_config(CONFIG_PATH, "clip_standard", PROJECT_ROOT, overrides)
        self.assertEqual(config.validation_sample_count // config.micro_batch_size, 222)
        self.assertEqual(config.validation_sample_count % config.micro_batch_size, 8)



if __name__ == "__main__":
    unittest.main()
