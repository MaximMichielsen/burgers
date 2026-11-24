"""Problem definitions for the 1D Burgers pipeline."""

from dataclasses import dataclass


STANDARD_REYNOLDS = 180


@dataclass(frozen=True)
class Problem:
    """Immutable problem definition with derived viscosity."""

    name: str
    domain_length: float
    t_start: float
    t_end: float
    reynolds: float = STANDARD_REYNOLDS

    @property
    def domain_timespan(self) -> float:
        if self.t_end <= self.t_start:
            raise ValueError(
                f"t_end ({self.t_end}) must be greater than t_start ({self.t_start})"
            )
        return self.t_end - self.t_start

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity derived from Re and domain length."""
        return self.domain_length / 2 / self.reynolds
