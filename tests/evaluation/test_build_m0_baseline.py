"""Building an M0 baseline for a probe: the step that feeds every delta.

The baseline is what ``m0 -> checkpoint`` deltas subtract, so a new probe needs
one before its checkpoints can be compared. These tests exercise that path with
synthetic data (no model is loaded): a canonical probe manifest, an embedding
export, and the baseline document that results.
"""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import load_probe_manifest  # noqa: E402
from embeddings.artifact import save_embedding_artifact  # noqa: E402
from scripts.evaluation import build_m0_baseline as builder  # noqa: E402


def write_probe_manifest(root: Path, name: str, count: int) -> Path:
    path = root / "data" / "splits" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "probe_name": name,
                "sample_count": count,
                "samples": [
                    {
                        "sample_id": f"sample_{index}",
                        "semantic_id": f"instance_{index}",
                        "image_relpath": f"data/raw/images/{index}.jpg",
                        "caption": f"caption {index}",
                    }
                    for index in range(count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class BuildM0BaselineTest(unittest.TestCase):
    SAMPLE_COUNT = 16  # the geometry reference needs more samples than kNN's k

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.embeddings_root = self.root / "outputs" / "embeddings" / "m0"
        self.metrics_root = self.root / "outputs" / "metrics" / "m0"
        self.pair_index_root = self.root / "data" / "metadata" / "geometry_references"
        self.anchor_list = self.root / "data" / "metadata" / "m0_sha256.txt"
        self.manifest_path = write_probe_manifest(self.root, "synthetic_probe", self.SAMPLE_COUNT)
        self.manifest = load_probe_manifest(self.manifest_path, self.root)

        artifact_model = builder.MODEL_ARTIFACT_NAMES["clip"]
        self.stem = f"m0_{artifact_model}_{self.manifest.name}"
        embedding_directory = self.embeddings_root / artifact_model / self.manifest.name
        generator = np.random.default_rng(20_260_915)
        save_embedding_artifact(
            embedding_directory,
            self.stem,
            [sample.sample_id for sample in self.manifest.samples],
            generator.standard_normal((self.SAMPLE_COUNT, 4)).astype(np.float32) * 2.0,
            generator.standard_normal((self.SAMPLE_COUNT, 4)).astype(np.float32) * 3.0,
            {
                "probe_name": self.manifest.name,
                "probe_manifest_sha256": self.manifest.sha256,
                "checkpoint": "synthetic-checkpoint",
            },
        )
        self.args = argparse.Namespace(
            model=["clip"], dataset=None, split=None, score_block_size=4, force=False, write_anchor=False
        )
        self._patchers = [
            patch.object(builder, "PROJECT_ROOT", self.root),
            patch.object(builder, "M0_EMBEDDINGS_ROOT", self.embeddings_root),
            patch.object(builder, "M0_METRICS_ROOT", self.metrics_root),
            patch.object(builder, "PAIR_INDEX_ROOT", self.pair_index_root),
            patch.object(builder, "ANCHOR_LIST", self.anchor_list),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temporary.cleanup)

    def metrics_path(self) -> Path:
        return self.metrics_root / builder.MODEL_ARTIFACT_NAMES["clip"] / self.manifest.name / f"{self.stem}_metrics.json"

    def test_baseline_document_matches_the_shipped_schema(self):
        metrics_path, geometry_path = builder.build_one("clip", self.manifest, self.args)

        self.assertTrue(metrics_path.is_file())
        self.assertTrue(geometry_path.is_file())
        document = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("centroid_gap_raw", "norm_imbalance", "anisotropy_gap", "intra_geometry_text"):
            self.assertIn(key, document)
        # M0 is the reference compared with itself.
        self.assertAlmostEqual(document["intra_geometry_image"], 1.0)
        self.assertAlmostEqual(document["intra_geometry_text"], 1.0)
        self.assertEqual(document["m0_baseline"]["probe_manifest_sha256"], self.manifest.sha256)
        self.assertEqual(document["m0_baseline"]["role"], "baseline")
        self.assertEqual(document["training_progress"], 0.0)
        # The probe's fixed pair index is created by the first baseline.
        pair_index = (
            self.pair_index_root
            / f"{self.manifest.name}_{self.manifest.sha256[:12]}_upper_triangle_pairs_v1.npz"
        )
        self.assertTrue(pair_index.is_file())

    def test_existing_baseline_is_frozen(self):
        builder.build_one("clip", self.manifest, self.args)
        with self.assertRaisesRegex(FileExistsError, "frozen data"):
            builder.build_one("clip", self.manifest, self.args)
        self.args.force = True
        builder.build_one("clip", self.manifest, self.args)

    def test_export_for_another_probe_is_refused(self):
        other = write_probe_manifest(self.root, "another_probe", self.SAMPLE_COUNT)
        other_manifest = load_probe_manifest(other, self.root)
        artifact_model = builder.MODEL_ARTIFACT_NAMES["clip"]
        other_stem = f"m0_{artifact_model}_{other_manifest.name}"
        generator = np.random.default_rng(20_260_915)
        save_embedding_artifact(
            self.embeddings_root / artifact_model / other_manifest.name,
            other_stem,
            [sample.sample_id for sample in other_manifest.samples],
            generator.standard_normal((self.SAMPLE_COUNT, 4)).astype(np.float32),
            generator.standard_normal((self.SAMPLE_COUNT, 4)).astype(np.float32),
            {
                "probe_name": other_manifest.name,
                # exported for the *other* probe: the builder must refuse it
                "probe_manifest_sha256": self.manifest.sha256,
            },
        )
        with self.assertRaisesRegex(ValueError, "was exported for probe manifest"):
            builder.build_one("clip", other_manifest, self.args)

    def test_anchor_covers_every_baseline_file(self):
        builder.build_one("clip", self.manifest, self.args)
        count = builder.write_anchor()
        self.assertEqual(count, 4)
        for line in self.anchor_list.read_text(encoding="utf-8").splitlines():
            digest, _, relative = line.partition("  ")
            self.assertEqual(len(digest), 64)
            self.assertTrue((self.root / relative).is_file())


if __name__ == "__main__":
    unittest.main()
