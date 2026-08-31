import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.formal_data import (  # noqa: E402
    prepare_lcs_splits,
    scan_training_manifest,
    sha256_file,
    validate_coco_probe,
    validate_lcs_splits,
)


class FormalDatasetTest(unittest.TestCase):
    def _lcs_fixture(self, root: Path) -> dict[str, Path | str]:
        annotation = root / "lcs.json"
        source = [
            {
                "id": f"{index:03d}",
                "image": f"folder/{index:03d}.jpg",
                "conversations": [
                    {"from": "human", "value": "describe"},
                    {"from": "gpt", "value": f"caption {index}"},
                ],
            }
            for index in range(8)
        ]
        annotation.write_text(json.dumps(source), encoding="utf-8")
        source_sha256 = sha256_file(annotation)
        probe = root / "probe.json"
        probe.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_annotation_sha256": source_sha256,
                    "sampling": {"method": "random.sample", "seed": 17, "count": 2},
                    "samples": [
                        {
                            "source_index": index,
                            "id": source[index]["id"],
                            "image": source[index]["image"],
                            "caption": f"caption {index}",
                        }
                        for index in (1, 6)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {
            "annotation_path": annotation,
            "probe_manifest_path": probe,
            "train_manifest_path": root / "processed" / "train.jsonl",
            "validation_manifest_path": root / "processed" / "validation.jsonl",
            "lock_path": root / "processed" / "lock.json",
            "expected_source_sha256": source_sha256,
        }

    def test_lcs_split_is_disjoint_exhaustive_and_reusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._lcs_fixture(root)
            lock = prepare_lcs_splits(
                project_root=root,
                **paths,
                expected_source_count=8,
                expected_probe_count=2,
                probe_seed=17,
                validation_count=2,
                validation_seed=23,
            )
            self.assertEqual(lock["splits"]["train"]["sample_count"], 4)
            self.assertEqual(lock["splits"]["validation"]["sample_count"], 2)
            train = scan_training_manifest(paths["train_manifest_path"])
            validation = scan_training_manifest(paths["validation_manifest_path"])
            self.assertFalse(train.source_indices & validation.source_indices)
            hashes = (train.sha256, validation.sha256, sha256_file(paths["lock_path"]))

            prepare_lcs_splits(
                project_root=root,
                **paths,
                expected_source_count=8,
                expected_probe_count=2,
                probe_seed=17,
                validation_count=2,
                validation_seed=23,
            )
            self.assertEqual(
                hashes,
                (
                    sha256_file(paths["train_manifest_path"]),
                    sha256_file(paths["validation_manifest_path"]),
                    sha256_file(paths["lock_path"]),
                ),
            )

    def test_lcs_source_change_is_rejected_by_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._lcs_fixture(root)
            prepare_lcs_splits(
                project_root=root,
                **paths,
                expected_source_count=8,
                expected_probe_count=2,
                probe_seed=17,
                validation_count=2,
                validation_seed=23,
            )
            paths["annotation_path"].write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_lcs_splits(
                    project_root=root,
                    **paths,
                    expected_source_count=8,
                    expected_probe_count=2,
                    expected_probe_seed=17,
                    expected_validation_count=2,
                )

    def test_coco_requires_minimum_caption_id_per_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = []
            annotations = []
            samples = []
            for image_id in (3, 8, 11):
                image_relpath = f"images/{image_id}.jpg"
                images.append({"id": image_id, "file_name": f"{image_id}.jpg"})
                annotations.extend(
                    [
                        {"id": image_id * 100 + 9, "image_id": image_id, "caption": "later"},
                        {"id": image_id * 100 + 2, "image_id": image_id, "caption": "selected"},
                    ]
                )
                instance_id = f"coco_2017_val:{image_id}"
                samples.append(
                    {
                        "sample_id": instance_id,
                        "semantic_id": instance_id,
                        "image_id": image_id,
                        "caption_id": image_id * 100 + 2,
                        "image_relpath": image_relpath,
                        "caption": "selected",
                    }
                )
            captions = root / "captions.json"
            captions.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")
            manifest = root / "coco.json"
            manifest.write_text(
                json.dumps(
                    {
                        "caption_selection": {"rule": "minimum_caption_id_per_image"},
                        "samples": samples,
                    }
                ),
                encoding="utf-8",
            )
            summary = validate_coco_probe(
                project_root=root,
                captions_path=captions,
                manifest_path=manifest,
                expected_captions_sha256=sha256_file(captions),
                expected_manifest_sha256=sha256_file(manifest),
                expected_count=3,
            )
            self.assertEqual(summary["sample_count"], 3)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["samples"][0]["caption_id"] += 7
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "minimum caption_id"):
                validate_coco_probe(
                    project_root=root,
                    captions_path=captions,
                    manifest_path=manifest,
                    expected_captions_sha256=sha256_file(captions),
                    expected_manifest_sha256=sha256_file(manifest),
                    expected_count=3,
                )


if __name__ == "__main__":
    unittest.main()
