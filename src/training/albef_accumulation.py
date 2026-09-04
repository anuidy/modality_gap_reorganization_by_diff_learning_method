from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any


class DeferredAlbefStateUpdates:
    """Defer native ALBEF queue mutation across one accumulation window."""

    def __init__(self) -> None:
        self._reset()

    @property
    def active(self) -> bool:
        return self._model is not None

    def begin(self, model: Any, expected_microbatches: int) -> None:
        if self.active:
            raise RuntimeError("An ALBEF accumulation window is already active.")
        if expected_microbatches <= 0:
            raise ValueError("expected_microbatches must be positive.")

        momentum_update = getattr(model, "_momentum_update", None)
        enqueue = getattr(model, "_dequeue_and_enqueue", None)
        if not callable(momentum_update) or not callable(enqueue):
            raise TypeError("ALBEF model must provide momentum-update and queue-enqueue methods.")

        self._model = model
        self._expected_microbatches = int(expected_microbatches)
        self._original_enqueue = enqueue
        try:
            momentum_update()
        except BaseException:
            self._reset()
            raise

    @contextlib.contextmanager
    def intercept_forward(self) -> Iterator[None]:
        if not self.active:
            raise RuntimeError("ALBEF forward ran outside an optimizer-step accumulation window.")
        if self._intercepting:
            raise RuntimeError("Nested ALBEF state-update interception is unsupported.")

        model = self._model
        sentinel = object()
        previous_momentum = model.__dict__.get("_momentum_update", sentinel)
        previous_enqueue = model.__dict__.get("_dequeue_and_enqueue", sentinel)

        def defer_momentum_update() -> None:
            self._momentum_update_requests += 1

        def collect_for_enqueue(image_features: Any, text_features: Any) -> None:
            self._pending_image_features.append(image_features)
            self._pending_text_features.append(text_features)

        self._intercepting = True
        object.__setattr__(model, "_momentum_update", defer_momentum_update)
        object.__setattr__(model, "_dequeue_and_enqueue", collect_for_enqueue)
        try:
            yield
        finally:
            if previous_momentum is sentinel:
                object.__delattr__(model, "_momentum_update")
            else:
                object.__setattr__(model, "_momentum_update", previous_momentum)
            if previous_enqueue is sentinel:
                object.__delattr__(model, "_dequeue_and_enqueue")
            else:
                object.__setattr__(model, "_dequeue_and_enqueue", previous_enqueue)
            self._intercepting = False

    def flush(self, concatenate: Callable[[tuple[Any, ...]], Any]) -> None:
        if not self.active:
            raise RuntimeError("No ALBEF accumulation window is active.")
        observed_microbatches = len(self._pending_image_features)
        if (
            self._momentum_update_requests != self._expected_microbatches
            or observed_microbatches != self._expected_microbatches
            or len(self._pending_text_features) != self._expected_microbatches
        ):
            raise RuntimeError(
                "ALBEF accumulation expected "
                f"{self._expected_microbatches} micro-batches but observed "
                f"{observed_microbatches} queue updates and "
                f"{self._momentum_update_requests} momentum-update requests."
            )

        image_features = concatenate(tuple(self._pending_image_features))
        text_features = concatenate(tuple(self._pending_text_features))
        enqueue = self._original_enqueue
        try:
            enqueue(image_features, text_features)
        except BaseException:
            raise
        else:
            self._reset()

    def abort(self) -> None:
        if self._intercepting:
            raise RuntimeError("Cannot abort ALBEF state updates during a model forward.")
        self._reset()

    def _reset(self) -> None:
        self._model: Any | None = None
        self._expected_microbatches = 0
        self._original_enqueue: Callable[[Any, Any], None] | None = None
        self._momentum_update_requests = 0
        self._pending_image_features: list[Any] = []
        self._pending_text_features: list[Any] = []
        self._intercepting = False
