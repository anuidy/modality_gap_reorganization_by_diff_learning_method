"""The evaluator reads the M0 baseline from one place, and only one place.

``outputs/metrics/m0/<model>/<probe>/`` holds the frozen metrics document and the
geometry reference of that model/probe; the raw embeddings live under
``outputs/embeddings/m0/``. There is no recomputation path and no second copy of
the M0 metrics, so an m0 -> checkpoint delta always uses the same source.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.artifact import save_embedding_artifact  # noqa: E402
from scripts.evaluation import evaluate_trajectory as evaluator  # noqa: E402


class M0BaselineResolutionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.embeddings_root = self.root / "outputs" / "embeddings" / "m0"
        self.metrics_root = self.root / "outputs" / "metrics" / "m0"
        self.manifest = SimpleNamespace(name="coco_2017_val_5k", sha256="a" * 64)
        self.artifact_model = evaluator.MODEL_ARTIFACT_NAMES["clip"]
        self.stem = f"m0_{self.artifact_model}_{self.manifest.name}"
        self.embedding_directory = self.embeddings_root / self.artifact_model / self.manifest.name
        self.metrics_directory = self.metrics_root / self.artifact_model / self.manifest.name
        self.metrics_directory.mkdir(parents=True)

        generator = np.random.default_rng(20_260_915)
        save_embedding_artifact(
            self.embedding_directory,
            self.stem,
            [f"sample_{index}" for index in range(4)],
            generator.standard_normal((4, 3)).astype(np.float32),
            generator.standard_normal((4, 3)).astype(np.float32),
            {
                "probe_name": self.manifest.name,
                "probe_manifest_sha256": self.manifest.sha256,
                "checkpoint_sha256": "0" * 64,
            },
        )
        self.metrics_path = self.metrics_directory / f"{self.stem}_metrics.json"
        self.metrics_path.write_text("{}", encoding="utf-8")
        self.geometry_path = self.metrics_directory / f"{self.stem}_geometry_reference.npz"
        self.geometry_path.write_bytes(b"geometry reference")

        self._patchers = [
            patch.object(evaluator, "PROJECT_ROOT", self.root),
            patch.object(evaluator, "M0_EMBEDDINGS_ROOT", self.embeddings_root),
            patch.object(evaluator, "M0_METRICS_ROOT", self.metrics_root),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temporary.cleanup)

    def test_m0_point_reads_the_canonical_baseline_files(self):
        point = evaluator._m0_point("clip", self.manifest, score_block_size=8)
        self.assertEqual(point.metrics_path, self.metrics_path)
        self.assertEqual(point.geometry_state_path, self.geometry_path)
        self.assertEqual(
            point.artifact_path, self.embedding_directory / f"{self.stem}.npz"
        )

    def test_missing_baseline_metrics_fails_loudly(self):
        self.metrics_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "M0 baseline artifact"):
            evaluator._m0_point("clip", self.manifest, score_block_size=8)

    def test_embedding_checksum_is_verified_against_its_metadata(self):
        (self.embedding_directory / f"{self.stem}.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "failed SHA-256 validation"):
            evaluator._m0_point("clip", self.manifest, score_block_size=8)


if __name__ == "__main__":
    unittest.main()
