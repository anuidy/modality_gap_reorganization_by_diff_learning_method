import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics.representation_metrics import (  # noqa: E402
    compare_geometry_states,
    floating_metric_deltas,
    neighbor_overlap,
    spearman_correlation,
)


class TrajectoryMetricTest(unittest.TestCase):
    def test_spearman_uses_average_tie_ranks(self):
        source = np.asarray([1.0, 2.0, 2.0, 4.0], dtype=np.float32)
        target = np.asarray([4.0, 3.0, 3.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(spearman_correlation(source, target), -1.0)

    def test_neighbor_overlap_is_mean_fraction_not_jaccard(self):
        source = np.asarray([[1, 2], [0, 2]], dtype=np.int32)
        target = np.asarray([[1, 3], [0, 2]], dtype=np.int32)
        self.assertEqual(neighbor_overlap(source, target), 0.75)

    def test_geometry_state_comparison_checks_manifest_and_modalities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.npz"
            target_path = root / "target.npz"
            common = {
                "manifest_sha256": np.asarray("manifest"),
                "knn_k": np.asarray(2, dtype=np.int64),
            }
            np.savez(
                source_path,
                **common,
                image_pair_cosines=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                text_pair_cosines=np.asarray([0.3, 0.2, 0.1], dtype=np.float32),
                image_knn_indices=np.asarray([[1, 2], [0, 2]], dtype=np.int32),
                text_knn_indices=np.asarray([[2, 1], [2, 0]], dtype=np.int32),
            )
            np.savez(
                target_path,
                **common,
                image_pair_cosines=np.asarray([0.2, 0.4, 0.6], dtype=np.float32),
                text_pair_cosines=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                image_knn_indices=np.asarray([[1, 3], [0, 2]], dtype=np.int32),
                text_knn_indices=np.asarray([[2, 1], [0, 3]], dtype=np.int32),
            )

            result = compare_geometry_states(source_path, target_path)

        self.assertAlmostEqual(result["image"]["spearman"], 1.0)
        self.assertAlmostEqual(result["text"]["spearman"], -1.0)
        self.assertEqual(result["image"]["neighbor_overlap"], 0.75)
        self.assertEqual(result["text"]["neighbor_overlap"], 0.75)

    def test_metric_deltas_only_include_matching_float_leaves(self):
        source = {"metric": {"raw": 1.5, "count": 10}, "protocol": "v1"}
        target = {"metric": {"raw": 2.0, "count": 10}, "protocol": "v1"}
        self.assertEqual(floating_metric_deltas(source, target), {"metric": {"raw": 0.5}})


if __name__ == "__main__":
    unittest.main()
