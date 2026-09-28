import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.checkpoint_plan import build_checkpoint_plan  # noqa: E402


class CheckpointPlanTest(unittest.TestCase):
    def test_default_protocol_resolves_to_fixed_optimizer_steps(self):
        plan = build_checkpoint_plan(max_steps=20_000)

        self.assertEqual(
            [(point.progress_fraction, point.optimizer_step, point.label) for point in plan.trajectory_points],
            [
                (0.01, 200, "p001"),
                (0.05, 1_000, "p005"),
                (0.20, 4_000, "p020"),
                (0.50, 10_000, "p050"),
                (1.00, 20_000, "p100"),
            ],
        )
        self.assertEqual(plan.resume_steps, frozenset({4_000, 8_000, 12_000, 16_000, 20_000}))
        self.assertEqual(plan.resume_retention, 2)

    def test_fractional_targets_round_half_up_and_final_is_always_included(self):
        plan = build_checkpoint_plan(
            max_steps=101,
            trajectory_progress_fractions=(0.01, 0.05, 0.20, 0.50, 1.00),
            resume_progress_interval=0.20,
        )

        self.assertEqual(
            [point.optimizer_step for point in plan.trajectory_points],
            [1, 5, 20, 51, 101],
        )
        self.assertEqual(plan.resume_steps, frozenset({20, 40, 61, 81, 101}))

    def test_formal_batch_36_one_epoch_schedule(self):
        plan = build_checkpoint_plan(max_steps=540128 // 36)
        self.assertEqual([p.optimizer_step for p in plan.trajectory_points], [150, 750, 3001, 7502, 15003])
        self.assertEqual(plan.resume_steps, frozenset({3001, 6001, 9002, 12002, 15003}))

    def test_rejects_trajectory_schedule_that_collides_at_small_step_count(self):
        with self.assertRaisesRegex(ValueError, "distinct trajectory"):
            build_checkpoint_plan(max_steps=3)

    def test_model_only_does_not_couple_validation_to_resume_storage(self):
        plan = build_checkpoint_plan(15003, save_resume_checkpoints=False, validation_progress_interval=.2)
        self.assertEqual(plan.resume_steps, frozenset())
        self.assertEqual(plan.resume_retention, 0)
        self.assertEqual(plan.validation_steps, frozenset({3001, 6001, 9002, 12002, 15003}))
        self.assertEqual([p.optimizer_step for p in plan.trajectory_points], [150, 750, 3001, 7502, 15003])

    def test_rejects_invalid_final_and_retention_settings(self):
        with self.assertRaisesRegex(ValueError, "end at 1.0"):
            build_checkpoint_plan(max_steps=100, trajectory_progress_fractions=(0.1, 0.5))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_checkpoint_plan(max_steps=100, resume_retention=1)


if __name__ == "__main__":
    unittest.main()
