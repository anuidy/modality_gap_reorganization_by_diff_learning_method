from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING


DEFAULT_TRAJECTORY_PROGRESS_FRACTIONS = (0.01, 0.05, 0.20, 0.50, 1.00)
DEFAULT_RESUME_PROGRESS_INTERVAL = 0.20
DEFAULT_RESUME_RETENTION = 2


@dataclass(frozen=True)
class TrajectoryPoint:
    progress_fraction: float
    optimizer_step: int

    @property
    def label(self) -> str:
        return f"p{round(self.progress_fraction * 100):03d}"

    @property
    def is_final(self) -> bool:
        return self.progress_fraction == 1.0


@dataclass(frozen=True)
class CheckpointPlan:
    max_steps: int
    trajectory_points: tuple[TrajectoryPoint, ...]
    resume_steps: frozenset[int]
    resume_retention: int

    def trajectory_point_at(self, optimizer_step: int) -> TrajectoryPoint | None:
        for point in self.trajectory_points:
            if point.optimizer_step == optimizer_step:
                return point
        return None

    def is_resume_step(self, optimizer_step: int) -> bool:
        return optimizer_step in self.resume_steps


def _validate_trajectory_fractions(fractions: tuple[float, ...]) -> None:
    if not fractions:
        raise ValueError("trajectory_progress_fractions must not be empty.")
    if fractions[-1] != 1.0:
        raise ValueError("trajectory_progress_fractions must end at 1.0 for the final checkpoint.")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("trajectory_progress_fractions must be in (0, 1].")
    if tuple(sorted(fractions)) != fractions or len(set(fractions)) != len(fractions):
        raise ValueError("trajectory_progress_fractions must be strictly increasing.")


def _ceil_progress_steps(progress_fraction: float, max_steps: int) -> int:
    value = Decimal(str(progress_fraction)) * Decimal(max_steps)
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def build_checkpoint_plan(
    max_steps: int,
    trajectory_progress_fractions: tuple[float, ...] = DEFAULT_TRAJECTORY_PROGRESS_FRACTIONS,
    resume_progress_interval: float = DEFAULT_RESUME_PROGRESS_INTERVAL,
    resume_retention: int = DEFAULT_RESUME_RETENTION,
) -> CheckpointPlan:
    """Resolve the frozen progress-based checkpoint protocol into optimizer steps."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    fractions = tuple(float(fraction) for fraction in trajectory_progress_fractions)
    _validate_trajectory_fractions(fractions)
    if not 0.0 < resume_progress_interval <= 1.0:
        raise ValueError("resume_progress_interval must be in (0, 1].")
    if resume_retention < 2:
        raise ValueError("resume_retention must be at least 2.")

    trajectory_points = tuple(
        TrajectoryPoint(
            progress_fraction=fraction,
            optimizer_step=_ceil_progress_steps(fraction, max_steps),
        )
        for fraction in fractions
    )
    trajectory_steps = tuple(point.optimizer_step for point in trajectory_points)
    if len(set(trajectory_steps)) != len(trajectory_steps):
        raise ValueError(
            "max_steps is too small to realize distinct trajectory progress checkpoints."
        )

    resume_steps: set[int] = set()
    multiplier = 1
    interval = Decimal(str(resume_progress_interval))
    while Decimal(multiplier) * interval < Decimal("1"):
        progress_fraction = Decimal(multiplier) * interval
        resume_steps.add(
            int((progress_fraction * Decimal(max_steps)).to_integral_value(rounding=ROUND_CEILING))
        )
        multiplier += 1
    resume_steps.add(max_steps)

    return CheckpointPlan(
        max_steps=max_steps,
        trajectory_points=trajectory_points,
        resume_steps=frozenset(resume_steps),
        resume_retention=resume_retention,
    )
