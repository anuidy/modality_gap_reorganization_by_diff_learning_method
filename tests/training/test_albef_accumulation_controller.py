import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.albef_accumulation import DeferredAlbefStateUpdates  # noqa: E402


class FakeAlbefModel:
    def __init__(self) -> None:
        self.momentum_update_calls = 0
        self.enqueue_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _momentum_update(self) -> None:
        self.momentum_update_calls += 1

    def _dequeue_and_enqueue(self, image_features, text_features) -> None:
        self.enqueue_calls.append((tuple(image_features), tuple(text_features)))


def concatenate(chunks):
    return tuple(item for chunk in chunks for item in chunk)


class DeferredAlbefStateUpdatesTest(unittest.TestCase):
    def test_one_momentum_update_and_one_merged_enqueue_per_optimizer_step(self):
        model = FakeAlbefModel()
        controller = DeferredAlbefStateUpdates()

        controller.begin(model, expected_microbatches=3)
        for index in range(3):
            with controller.intercept_forward():
                model._momentum_update()
                model._dequeue_and_enqueue([index], [index + 10])

        self.assertEqual(model.momentum_update_calls, 1)
        self.assertEqual(model.enqueue_calls, [])

        controller.flush(concatenate)

        self.assertEqual(model.momentum_update_calls, 1)
        self.assertEqual(model.enqueue_calls, [((0, 1, 2), (10, 11, 12))])
        self.assertFalse(controller.active)

    def test_incomplete_accumulation_window_is_rejected(self):
        model = FakeAlbefModel()
        controller = DeferredAlbefStateUpdates()
        controller.begin(model, expected_microbatches=2)
        with controller.intercept_forward():
            model._momentum_update()
            model._dequeue_and_enqueue([0], [10])

        with self.assertRaisesRegex(RuntimeError, "expected 2 micro-batches"):
            controller.flush(concatenate)
        controller.abort()
        self.assertFalse(controller.active)

    def test_native_methods_are_restored_after_interception(self):
        model = FakeAlbefModel()
        controller = DeferredAlbefStateUpdates()
        controller.begin(model, expected_microbatches=1)
        with controller.intercept_forward():
            model._momentum_update()
            model._dequeue_and_enqueue([0], [10])
        controller.flush(concatenate)

        model._momentum_update()
        model._dequeue_and_enqueue([1], [11])

        self.assertEqual(model.momentum_update_calls, 2)
        self.assertEqual(
            model.enqueue_calls,
            [((0,), (10,)), ((1,), (11,))],
        )


if __name__ == "__main__":
    unittest.main()
