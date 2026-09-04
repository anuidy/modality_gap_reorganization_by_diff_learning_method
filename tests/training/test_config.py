import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import EXPECTED_RUNS, load_run_config, validate_experiment_matrix  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "train_runs.yaml"


class TrainingConfigTest(unittest.TestCase):
    def test_formal_matrix_contains_exactly_eight_locked_runs(self):
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        validate_experiment_matrix(payload)
        self.assertEqual(set(payload["runs"]), set(EXPECTED_RUNS))

    def test_unfrozen_hyperparameters_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "not frozen"):
            load_run_config(CONFIG_PATH, "clip_standard", PROJECT_ROOT)

    def test_programmatic_overrides_resolve_a_run(self):
        overrides = {
            "train_manifest": "data/train.jsonl",
            "image_root": "data/images",
            "seed": 7,
            "micro_batch_size": 8,
            "gradient_accumulation": 2,
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
        config = load_run_config(CONFIG_PATH, "clip_count_matched_mixed", PROJECT_ROOT, overrides)
        self.assertEqual(config.model_name, "clip")
        self.assertEqual(config.branch, "count_matched_mixed")
        self.assertEqual(config.effective_batch_size, 16)
        self.assertEqual(config.trajectory_progress_fractions, (0.01, 0.05, 0.20, 0.50, 1.00))
        self.assertEqual(config.resume_progress_interval, 0.20)
        self.assertEqual(config.resume_retention, 2)

    def test_run_cannot_override_a_controlled_field(self):
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["runs"]["clip_standard"]["learning_rate"] = 1e-4
        with self.assertRaisesRegex(ValueError, "overrides controlled fields"):
            validate_experiment_matrix(payload)


if __name__ == "__main__":
    unittest.main()
