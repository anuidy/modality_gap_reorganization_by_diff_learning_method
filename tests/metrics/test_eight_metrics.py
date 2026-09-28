import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics.representation_metrics import (  # noqa: E402
    COVARIANCE_GAP_EPSILON,
    EFFECTIVE_RANK_EPSILON,
    SCORE_GAP_CATEGORY,
    anisotropy,
    centroid_gap,
    compute_point_metrics,
    covariance_gap,
    cross_modal_alignment,
    effective_rank,
    l2_normalize,
    matched_pair_cosine_summary,
    norm_imbalance,
    score_gap,
)


def _raw(rows: int = 8, dim: int = 6, seed: int = 0, scale: float = 1.0):
    generator = np.random.default_rng(seed)
    return (generator.standard_normal((rows, dim)) * scale).astype(np.float32)


class NormImbalanceTest(unittest.TestCase):
    def test_formula_matches_abs_log_of_mean_norm_ratio(self):
        image = _raw(16, 8, seed=1, scale=3.0)
        text = _raw(16, 8, seed=2, scale=1.0)
        stats = norm_imbalance(image, text)
        image_mean = float(np.linalg.norm(image, axis=1).mean())
        text_mean = float(np.linalg.norm(text, axis=1).mean())
        self.assertAlmostEqual(stats["image_norm_mean"], image_mean, places=6)
        self.assertAlmostEqual(stats["image_text_norm_ratio"], image_mean / text_mean, places=6)
        self.assertAlmostEqual(stats["norm_imbalance"], abs(np.log(image_mean / text_mean)), places=6)

    def test_ratio_keeps_the_direction_that_the_absolute_value_removes(self):
        # Same matrices with the roles swapped: the ratio inverts, |log| does not.
        image = _raw(16, 8, 3, 5.0)
        text = _raw(16, 8, 4, 1.0)
        dominant_image = norm_imbalance(image, text)
        dominant_text = norm_imbalance(text, image)
        self.assertGreater(dominant_image["image_text_norm_ratio"], 1.0)
        self.assertLess(dominant_text["image_text_norm_ratio"], 1.0)
        self.assertAlmostEqual(
            dominant_image["norm_imbalance"], dominant_text["norm_imbalance"], places=9
        )

    def test_l2_rows_have_unit_norm_so_the_metric_is_not_defined_there(self):
        image = _raw(16, 8, 5, 2.0)
        text = _raw(16, 8, 6, 2.0)
        for matrix in (l2_normalize(image), l2_normalize(text)):
            norms = np.linalg.norm(matrix, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        point = compute_point_metrics(image, text)
        self.assertNotIn("norm_imbalance_l2", point)
        self.assertIn("norm_imbalance", point)


class AnisotropyTest(unittest.TestCase):
    def test_matches_brute_force_and_excludes_the_diagonal(self):
        embeddings = l2_normalize(_raw(37, 11, seed=7))
        similarities = embeddings @ embeddings.T
        off_diagonal = ~np.eye(embeddings.shape[0], dtype=bool)
        expected = float(similarities[off_diagonal].mean(dtype=np.float64))
        self.assertAlmostEqual(anisotropy(embeddings, block_size=8), expected, places=10)
        with_diagonal = float(similarities.mean(dtype=np.float64))
        self.assertNotAlmostEqual(anisotropy(embeddings, block_size=8), with_diagonal, places=6)

    def test_block_size_does_not_change_the_value(self):
        # float32 matmul accumulates in a different order per block size, so the
        # agreement is at float32 tolerance, not bit-exact.
        embeddings = l2_normalize(_raw(64, 16, seed=8))
        values = [anisotropy(embeddings, block_size=size) for size in (1, 7, 64, 512)]
        for value in values[1:]:
            self.assertAlmostEqual(value, values[0], places=6)

    def test_cosine_makes_raw_and_l2_equivalent(self):
        raw = _raw(24, 9, seed=9, scale=4.0)
        self.assertAlmostEqual(
            anisotropy(l2_normalize(raw), block_size=5),
            anisotropy(l2_normalize(l2_normalize(raw)), block_size=5),
            places=6,
        )


class RawVersusL2Test(unittest.TestCase):
    def test_centroid_covariance_and_rank_report_both_spaces(self):
        image = _raw(32, 10, seed=11, scale=3.0)
        text = _raw(32, 10, seed=12, scale=1.0)
        point = compute_point_metrics(image, text)
        for key in (
            "centroid_gap_raw",
            "centroid_gap_l2",
            "covariance_gap_raw",
            "covariance_gap_l2",
            "effective_rank_image_raw",
            "effective_rank_text_raw",
            "effective_rank_image_l2",
            "effective_rank_text_l2",
        ):
            with self.subTest(key=key):
                self.assertIsInstance(point[key], float)
                self.assertTrue(np.isfinite(point[key]))
        self.assertAlmostEqual(point["centroid_gap_raw"], centroid_gap(image, text), places=9)
        self.assertNotAlmostEqual(point["centroid_gap_raw"], point["centroid_gap_l2"], places=4)
        # Legacy nested keys stay in place for existing artifacts.
        self.assertAlmostEqual(point["centroid_gap"]["raw"], point["centroid_gap_raw"], places=12)
        self.assertAlmostEqual(
            point["effective_rank"]["l2_normalized"]["image"], point["effective_rank_image_l2"], places=12
        )

    def test_covariance_gap_is_relative_frobenius_and_guarded(self):
        image = _raw(24, 7, seed=13, scale=2.0)
        text = _raw(24, 7, seed=14, scale=2.0)
        image_covariance = np.cov(image.astype(np.float64), rowvar=False, ddof=1)
        text_covariance = np.cov(text.astype(np.float64), rowvar=False, ddof=1)
        expected = float(
            np.linalg.norm(image_covariance - text_covariance, ord="fro")
            / (np.linalg.norm(image_covariance, ord="fro") + np.linalg.norm(text_covariance, ord="fro"))
        )
        self.assertAlmostEqual(covariance_gap(image, text), expected, places=6)
        self.assertLessEqual(covariance_gap(image, text), 1.0)
        # A degenerate denominator is floored, not turned into inf/nan.
        degenerate = np.zeros((4, 3), dtype=np.float32)
        self.assertTrue(np.isfinite(covariance_gap(degenerate, degenerate)))
        self.assertGreater(COVARIANCE_GAP_EPSILON, 0.0)

    def test_effective_rank_uses_entropy_of_normalized_eigenvalues(self):
        embeddings = _raw(40, 12, seed=15)
        covariance = np.cov(embeddings.astype(np.float64), rowvar=False, ddof=1)
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        probabilities = eigenvalues[eigenvalues > EFFECTIVE_RANK_EPSILON] / eigenvalues.sum()
        expected = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        self.assertAlmostEqual(effective_rank(embeddings), expected, places=9)
        self.assertLessEqual(effective_rank(embeddings), embeddings.shape[1] + 1e-9)


class AlignmentAndScoreTest(unittest.TestCase):
    def test_alignment_equals_two_minus_two_cosine_identity(self):
        image = l2_normalize(_raw(20, 8, seed=16))
        text = l2_normalize(_raw(20, 8, seed=17))
        alignment = cross_modal_alignment(image, text)
        cosines = np.sum(image * text, axis=1)
        expected = float(np.mean(np.sqrt(np.clip(2.0 - 2.0 * cosines, 0.0, None))))
        self.assertAlmostEqual(alignment["mean"], expected, places=6)
        summary = matched_pair_cosine_summary(image, text)
        self.assertAlmostEqual(summary["mean"], float(cosines.mean()), places=6)

    def test_score_gap_excludes_self_and_matched_positive(self):
        rows = 12
        image = l2_normalize(_raw(rows, 5, seed=18))
        text = l2_normalize(_raw(rows, 5, seed=19))
        scores = score_gap(image, text, block_size=4)
        expected_pairs = rows * (rows - 1)
        self.assertEqual(scores["image_query"]["candidate_pair_count"], expected_pairs)
        self.assertEqual(scores["text_query"]["candidate_pair_count"], expected_pairs)
        bruteforce = image @ image.T
        mask = ~np.eye(rows, dtype=bool)
        self.assertAlmostEqual(
            scores["image_query"]["same_modality_distribution"]["mean"],
            float(bruteforce[mask].mean(dtype=np.float64)),
            places=5,
        )
        cross = image @ text.T
        self.assertAlmostEqual(
            scores["image_query"]["cross_modality_distribution"]["mean"],
            float(cross[mask].mean(dtype=np.float64)),
            places=5,
        )

    def test_score_gap_is_marked_as_score_level(self):
        image = l2_normalize(_raw(10, 4, seed=20))
        text = l2_normalize(_raw(10, 4, seed=21))
        point = compute_point_metrics(image, text)
        self.assertEqual(point["score_gap"]["category"], SCORE_GAP_CATEGORY)
        self.assertEqual(
            point["metric_protocol"]["categories"]["score_gap"], "score_level"
        )
        self.assertEqual(
            point["metric_protocol"]["categories"]["centroid_gap"], "representation_geometry"
        )
        self.assertAlmostEqual(point["score_gap_mean"], point["score_gap"]["overall_wasserstein_1"], places=12)


class SchemaTest(unittest.TestCase):
    def test_eight_metric_flat_keys_and_auxiliary_placement(self):
        image = _raw(16, 6, seed=22, scale=2.5)
        text = _raw(16, 6, seed=23, scale=0.5)
        point = compute_point_metrics(
            image, text, representation_boundary="encoder_output_pre_l2", raw_available=True
        )
        required = (
            "centroid_gap_raw",
            "centroid_gap_l2",
            "covariance_gap_raw",
            "covariance_gap_l2",
            "effective_rank_image_raw",
            "effective_rank_text_raw",
            "effective_rank_image_l2",
            "effective_rank_text_l2",
            "cross_modal_alignment",
            "matched_pair_cosine_mean",
            "matched_pair_cosine_std",
            "score_gap_image",
            "score_gap_text",
            "score_gap_mean",
            "image_norm_mean",
            "image_norm_std",
            "text_norm_mean",
            "text_norm_std",
            "image_text_norm_ratio",
            "norm_imbalance",
            "anisotropy_image",
            "anisotropy_text",
            "anisotropy_gap",
        )
        for key in required:
            with self.subTest(key=key):
                self.assertIn(key, point)
                self.assertTrue(np.isfinite(point[key]))
        self.assertEqual(point["representation_boundary"], "encoder_output_pre_l2")
        self.assertTrue(point["raw_available"])
        self.assertIn("auxiliary_metrics", point)
        self.assertIn("cross_modal_alignment_distribution", point["auxiliary_metrics"])
        self.assertEqual(
            point["anisotropy_gap"], abs(point["anisotropy_image"] - point["anisotropy_text"])
        )

    def test_identity_fields_are_merged_when_supplied(self):
        identity = {
            "model": "clip",
            "training_regime": "rotating",
            "global_step": 3376,
            "training_progress": 0.2,
            "sample_count": 16,
        }
        point = compute_point_metrics(
            _raw(16, 6, seed=24), _raw(16, 6, seed=25), identity=identity
        )
        for key, value in identity.items():
            self.assertEqual(point[key], value)


if __name__ == "__main__":
    unittest.main()
