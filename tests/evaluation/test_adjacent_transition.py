import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.artifact import save_embedding_artifact  # noqa: E402
from evaluation.trajectory import PointOutput, compute_adjacent_transition  # noqa: E402


def geometry_state(path: Path, scale: float) -> None:
    np.savez(
        path,
        image_pair_cosines=np.asarray([1.0, 2.0, 3.0], dtype=np.float32) * scale,
        text_pair_cosines=np.asarray([3.0, 2.0, 1.0], dtype=np.float32) * scale,
        image_knn_indices=np.asarray([[1, 2], [0, 2]], dtype=np.int32),
        text_knn_indices=np.asarray([[1, 2], [0, 2]], dtype=np.int32),
        manifest_sha256=np.asarray("manifest"),
        knn_k=np.asarray(2, dtype=np.int64),
    )


class AdjacentTransitionTest(unittest.TestCase):
    def test_transition_uses_saved_raw_artifacts_and_target_minus_source_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            points = []
            for label, value in (("m0", 1.0), ("p001", 1.5)):
                directory = root / label
                artifact_path, metadata_path = save_embedding_artifact(
                    output_directory=directory,
                    stem=label,
                    sample_ids=["a", "b"],
                    image_embeddings_raw=np.asarray([[value, 0.0], [0.0, value]], dtype=np.float32),
                    text_embeddings_raw=np.asarray([[0.0, value], [value, 0.0]], dtype=np.float32),
                    metadata={
                        "probe_name": "probe",
                        "probe_manifest_sha256": "manifest",
                    },
                )
                metrics_path = directory / "metrics.json"
                metrics_path.write_text(
                    json.dumps({"centroid_gap": {"raw": value}, "candidate_count": 2}),
                    encoding="utf-8",
                )
                state_path = directory / "geometry.npz"
                geometry_state(state_path, value)
                points.append(
                    PointOutput(
                        label=label,
                        artifact_path=artifact_path,
                        metadata_path=metadata_path,
                        metrics_path=metrics_path,
                        geometry_state_path=state_path,
                    )
                )
            output_path = root / "transition.json"

            transition = compute_adjacent_transition(points[0], points[1], output_path)

            self.assertEqual(
                transition["point_metric_delta_target_minus_source"],
                {"centroid_gap": {"raw": 0.5}},
            )
            self.assertAlmostEqual(
                transition["intra_modal_geometry_preservation"]["image"]["spearman"],
                1.0,
            )
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
