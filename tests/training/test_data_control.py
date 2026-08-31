import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.data_control import _sha256_file, validate_formal_data_identity  # noqa: E402


class FormalDataControlTest(unittest.TestCase):
    def _config(self, root: Path):
        dataset_spec = root / "formal_datasets.yaml"
        dataset_spec.write_text("schema_version: 1\n", encoding="utf-8")
        train = root / "train.jsonl"
        validation = root / "validation.jsonl"
        lcs_probe = root / "lcs_probe.json"
        coco_probe = root / "coco_probe.json"
        for path, content in (
            (train, "train\n"),
            (validation, "validation\n"),
            (lcs_probe, "lcs probe\n"),
            (coco_probe, "coco probe\n"),
        ):
            path.write_text(content, encoding="utf-8")
        lock = root / "lock.json"
        lock.write_text(
            json.dumps(
                {
                    "dataset": "lcs_558k",
                    "splits": {
                        "train": {
                            "sha256": _sha256_file(train),
                            "role": "parameter_updates_only",
                        },
                        "validation": {
                            "sha256": _sha256_file(validation),
                            "role": "training_monitoring_only_no_checkpoint_selection",
                            "sample_count": 1,
                        },
                        "in_domain_probe": {
                            "sha256": _sha256_file(lcs_probe),
                            "role": "representation_metrics_only",
                        },
                    },
                    "invariants": {"branch_resampling_forbidden": True},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            dataset_name="lcs_558k",
            dataset_spec=dataset_spec,
            split_lock=lock,
            split_lock_sha256=_sha256_file(lock),
            train_manifest=train,
            train_manifest_sha256=_sha256_file(train),
            validation_manifest=validation,
            validation_manifest_sha256=_sha256_file(validation),
            validation_sample_count=1,
            lcs_probe_manifest=lcs_probe,
            lcs_probe_manifest_sha256=_sha256_file(lcs_probe),
            coco_probe_manifest=coco_probe,
            coco_probe_manifest_sha256=_sha256_file(coco_probe),
        )

    def test_locked_identity_is_accepted_and_records_probe_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            result = validate_formal_data_identity(config, config.train_manifest_sha256)
            self.assertEqual(result["mode"], "formal_locked_lcs_558k")
            self.assertFalse(result["probes_used_for_training_or_checkpoint_selection"])

    def test_changed_validation_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            config.validation_manifest.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation SHA-256 mismatch"):
                validate_formal_data_identity(config, config.train_manifest_sha256)


if __name__ == "__main__":
    unittest.main()
