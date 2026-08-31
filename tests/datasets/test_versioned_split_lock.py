import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.formal_data import prepare_lcs_splits, sha256_file  # noqa: E402


class VersionedSplitLockTest(unittest.TestCase):
    def test_preexisting_lock_bootstraps_missing_generated_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "source.json"
            source = [
                {
                    "id": str(index),
                    "image": f"{index}.jpg",
                    "conversations": [{"from": "gpt", "value": f"caption {index}"}],
                }
                for index in range(7)
            ]
            annotation.write_text(json.dumps(source), encoding="utf-8")
            source_sha256 = sha256_file(annotation)
            probe = root / "probe.json"
            probe.write_text(
                json.dumps(
                    {
                        "source_annotation_sha256": source_sha256,
                        "sampling": {"seed": 41, "count": 2},
                        "samples": [
                            {
                                "source_index": index,
                                "id": str(index),
                                "image": f"{index}.jpg",
                                "caption": f"caption {index}",
                            }
                            for index in (0, 5)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            train = root / "generated" / "train.jsonl"
            validation = root / "generated" / "validation.jsonl"
            lock = root / "versioned_lock.json"
            arguments = {
                "project_root": root,
                "annotation_path": annotation,
                "probe_manifest_path": probe,
                "train_manifest_path": train,
                "validation_manifest_path": validation,
                "lock_path": lock,
                "expected_source_sha256": source_sha256,
                "expected_source_count": 7,
                "expected_probe_count": 2,
                "probe_seed": 41,
                "validation_count": 1,
                "validation_seed": 43,
            }
            prepare_lcs_splits(**arguments)
            expected = (sha256_file(train), sha256_file(validation), sha256_file(lock))
            train.unlink()
            validation.unlink()

            prepare_lcs_splits(**arguments)

            self.assertEqual(
                expected,
                (sha256_file(train), sha256_file(validation), sha256_file(lock)),
            )


if __name__ == "__main__":
    unittest.main()
