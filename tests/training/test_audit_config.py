from pathlib import Path
import sys
import tempfile
import unittest

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.training import test_data_control
from training.audit_config import load_audit_config
from training.data_control import _sha256_file


class AuditConfigTest(unittest.TestCase):
    def fixture(self, root):
        identity = test_data_control.FormalDataControlTest()._config(root)
        payload = yaml.safe_load((PROJECT_ROOT / "configs/training/train_runs.yaml").read_text(encoding="utf-8"))
        payload["controls"]["data"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(identity).items()}
        payload["controls"]["data"]["image_root"] = str(root)
        checkpoint = root / "m0.pt"
        checkpoint.write_bytes(b"test checkpoint identity")
        for model in payload["models"].values():
            model["checkpoint"] = str(checkpoint)
            model["checkpoint_sha256"] = _sha256_file(checkpoint)
            model["resources"] = {}
        config = root / "audit_source.yaml"
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return config

    def load(self, path, run="clip_standard", **kwargs):
        args = dict(seed=4, batch_size=4, augmentation="resize_center_crop", flip_probability=0.0)
        args.update(kwargs)
        return load_audit_config(path, run, path.parent, **args)

    def test_unfrozen_training_settings_are_untouched_and_not_required(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            original = path.read_bytes()
            config = self.load(path)
            self.assertEqual(config.batch_size, 4)
            self.assertFalse(hasattr(config, "learning_rate"))
            self.assertFalse(hasattr(config, "max_steps"))
            self.assertEqual(path.read_bytes(), original)
            self.assertIsNone(yaml.safe_load(original)["controls"]["seed"])

    def test_probe_cannot_be_substituted_for_train(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["controls"]["data"]["train_manifest"] = payload["controls"]["data"]["lcs_probe_manifest"]
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Train manifest SHA-256 mismatch"):
                self.load(path)

    def test_checkpoint_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            (path.parent / "m0.pt").write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
                self.load(path)

    def test_albef_requires_explicit_diagnostic_alpha(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "explicit --alpha"):
                self.load(path, "albef_full")
            config = self.load(path, "albef_full", alpha=0.3)
            self.assertEqual(config.options["alpha"], 0.3)
            self.assertEqual(config.options["alpha_warmup_steps"], 0)

    def test_invalid_batch_and_crop_are_rejected_before_io(self):
        for kwargs in ({"batch_size": 1}, {"seed": -1}, {"flip_probability": float("nan")},
                       {"augmentation": "random_resized_crop"}, {"crop_scale": (0.8, 1.0)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load(Path("nonexistent.yaml"), **kwargs)


if __name__ == "__main__":
    unittest.main()
