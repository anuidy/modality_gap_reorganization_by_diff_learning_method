import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AlbefAccumulationWiringTest(unittest.TestCase):
    def test_engine_opens_and_flushes_window_around_all_microbatches(self):
        source = (PROJECT_ROOT / "src" / "training" / "engine.py").read_text(encoding="utf-8")
        begin = source.index("backend.begin_optimizer_step(config.gradient_accumulation)")
        forward = source.index("result = backend(prepared, config.branch, completed_steps)")
        flush = source.index("backend.before_optimizer_step()")
        optimizer_step = source.index("optimizer.step()", flush)
        after = source.index("backend.after_optimizer_step()", optimizer_step)

        self.assertLess(begin, forward)
        self.assertLess(forward, flush)
        self.assertLess(flush, optimizer_step)
        self.assertLess(optimizer_step, after)

    def test_both_albef_training_branches_intercept_native_state_updates(self):
        source = (PROJECT_ROOT / "src" / "training" / "backends.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            source.count("with self._deferred_state_updates.intercept_forward():"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
