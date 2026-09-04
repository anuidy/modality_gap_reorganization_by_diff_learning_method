import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import load_run_config  # noqa: E402


class ModelControlConfigTest(unittest.TestCase):
    def test_model_level_optimizer_is_shared_by_its_two_branches(self):
        source = PROJECT_ROOT / "configs" / "training" / "train_runs.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["models"]["clip"]["optimizer"] = {
            "type": "adamw",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
        }
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "runs.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            common = {
                "train_manifest": "data/train.jsonl",
                "image_root": "data/images",
                "seed": 7,
                "micro_batch_size": 8,
                "gradient_accumulation": 1,
                "scheduler_type": "cosine",
                "warmup_steps": 10,
                "min_lr_ratio": 0.1,
                "max_steps": 100,
                "augmentation_name": "random_resized_crop",
                "augmentation_scale_min": 0.8,
                "augmentation_scale_max": 1.0,
                "augmentation_hflip": 0.5,
            }
            standard = load_run_config(config_path, "clip_standard", PROJECT_ROOT, common)
            mixed = load_run_config(config_path, "clip_count_matched_mixed", PROJECT_ROOT, common)
        self.assertEqual(standard.learning_rate, 2e-5)
        self.assertEqual(standard.learning_rate, mixed.learning_rate)
        self.assertEqual(standard.weight_decay, mixed.weight_decay)


if __name__ == "__main__":
    unittest.main()
