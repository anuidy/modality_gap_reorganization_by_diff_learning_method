from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


DEFAULT_TRAJECTORY_PROGRESS_FRACTIONS = (0.01, 0.05, 0.20, 0.50, 1.00)
DEFAULT_RESUME_PROGRESS_INTERVAL = 0.20
DEFAULT_RESUME_RETENTION = 2


@dataclass(frozen=True)
class TrajectoryPoint:
    progress_fraction: float
    optimizer_step: int
    final_progress_fraction: float = 1.0

    @property
    def label(self) -> str:
        return f"p{round(self.progress_fraction * 100):03d}"

    @property
    def is_final(self) -> bool:
        return self.progress_fraction == self.final_progress_fraction


@dataclass(frozen=True)
class CheckpointPlan:
    max_steps: int
    trajectory_points: tuple[TrajectoryPoint, ...]
    resume_steps: frozenset[int]
    resume_retention: int
    validation_steps: frozenset[int]

    def trajectory_point_at(self, optimizer_step: int) -> TrajectoryPoint | None:
        for point in self.trajectory_points:
            if point.optimizer_step == optimizer_step:
                return point
        return None

    def is_resume_step(self, optimizer_step: int) -> bool:
        return optimizer_step in self.resume_steps

    def is_validation_step(self, optimizer_step: int) -> bool:
        return optimizer_step in self.validation_steps


def _validate_trajectory_fractions(fractions: tuple[float, ...], final_fraction: float = 1.0) -> None:
    if not fractions:
        raise ValueError("trajectory_progress_fractions must not be empty.")
    if fractions[-1] != final_fraction:
        raise ValueError(f"trajectory_progress_fractions must end at {final_fraction} for the final checkpoint.")
    if any(not 0.0 < fraction <= final_fraction for fraction in fractions):
        raise ValueError(f"trajectory_progress_fractions must be in (0, {final_fraction}].")
    if tuple(sorted(fractions)) != fractions or len(set(fractions)) != len(fractions):
        raise ValueError("trajectory_progress_fractions must be strictly increasing.")


def _round_progress_steps(progress_fraction: float, max_steps: int) -> int:
    value = Decimal(str(progress_fraction)) * Decimal(max_steps)
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def build_checkpoint_plan(
    max_steps: int,
    trajectory_progress_fractions: tuple[float, ...] = DEFAULT_TRAJECTORY_PROGRESS_FRACTIONS,
    resume_progress_interval: float = DEFAULT_RESUME_PROGRESS_INTERVAL,
    resume_retention: int = DEFAULT_RESUME_RETENTION,
    save_resume_checkpoints: bool = True,
    validation_progress_interval: float | None = None,
    progress_reference_steps: int | None = None,
) -> CheckpointPlan:
    """Resolve the frozen progress-based checkpoint protocol into optimizer steps."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if progress_reference_steps is not None and (type(progress_reference_steps) is not int or progress_reference_steps <= 0):
        raise ValueError("progress_reference_steps must be a positive integer.")
    reference = max_steps if progress_reference_steps is None else progress_reference_steps
    final_fraction = max_steps / reference
    fractions = tuple(float(fraction) for fraction in trajectory_progress_fractions)
    _validate_trajectory_fractions(fractions, final_fraction)
    if not 0.0 < resume_progress_interval <= 1.0:
        raise ValueError("resume_progress_interval must be in (0, 1].")
    if resume_retention < 2:
        raise ValueError("resume_retention must be at least 2.")
    validation_interval = resume_progress_interval if validation_progress_interval is None else validation_progress_interval
    if not 0.0 < validation_interval <= 1.0:
        raise ValueError("validation_progress_interval must be in (0, 1].")

    trajectory_points = tuple(
        TrajectoryPoint(
            progress_fraction=fraction,
            optimizer_step=_round_progress_steps(fraction, reference),
            final_progress_fraction=final_fraction,
        )
        for fraction in fractions
    )
    trajectory_steps = tuple(point.optimizer_step for point in trajectory_points)
    if min(trajectory_steps) < 1 or len(set(trajectory_steps)) != len(trajectory_steps):
        raise ValueError(
            "max_steps is too small to realize distinct trajectory progress checkpoints."
        )

    def periodic_steps(progress_interval: float) -> frozenset[int]:
        steps = {max_steps}
        multiplier = 1
        interval = Decimal(str(progress_interval))
        while Decimal(multiplier) * interval * Decimal(reference) < Decimal(max_steps):
            value = Decimal(multiplier) * interval * Decimal(reference)
            steps.add(max(1, int(value.to_integral_value(rounding=ROUND_HALF_UP))))
            multiplier += 1
        return frozenset(steps)

    return CheckpointPlan(
        max_steps=max_steps,
        trajectory_points=trajectory_points,
        resume_steps=periodic_steps(resume_progress_interval) if save_resume_checkpoints else frozenset(),
        resume_retention=resume_retention if save_resume_checkpoints else 0,
        validation_steps=periodic_steps(validation_interval),
    )
