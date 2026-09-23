"""Scheduler for mean profile and instantaneous snapshots for reward signaling."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# TODO: adjust name to something better


class ReferenceTrajectory:
    """Manages reference target profiles and projected snapshots for RL rewards."""

    def __init__(
        self,
        target_profile_path: Path,
        projected_w_path: Path,
        n_les_nodes: int,
    ) -> None:

        self.target_profile_path = Path(target_profile_path)
        self.projected_w_path = Path(projected_w_path)
        self.n_les_nodes = n_les_nodes

        self._current_step_idx: int = 0
        self._target_profile: NDArray | None = None
        self._projected_w_data: NDArray | None = None

        self.load_data()

    def load_data(self) -> None:
        """Loads reference datasets into memory with memory-mapped array access."""
        if not self.target_profile_path.exists():
            raise FileNotFoundError(
                f"Target profile missing at {self.target_profile_path}"
            )
        if not self.projected_w_path.exists():
            raise FileNotFoundError(
                f"Projected w field missing at {self.projected_w_path}"
            )

        # Static 1D target mean profile <u_DNS>_T
        self._target_profile = np.load(self.target_profile_path).astype(np.float64)

        # Time-resolved projected field (Memory-mapped query interface)
        self._projected_w_data = np.load(self.projected_w_path, mmap_mode="r").astype(
            np.float64
        )

        # Basic shape validation
        if self._target_profile.shape[-1] != self.n_les_nodes:
            raise ValueError(
                f"Target profile node count ({self._target_profile.shape[-1]}) "
                f"mismatches LES grid ({self.n_les_nodes})."
            )

    def query(self, step_idx: int) -> NDArray:
        """Query projected solution field at a specific timestep index."""
        if self._projected_w_data is None:
            raise RuntimeError("Reference data is not loaded.")

        # Safeguard index boundaries
        safe_idx = min(max(0, step_idx), len(self._projected_w_data) - 1)
        return np.asarray(self._projected_w_data[safe_idx], dtype=np.float64)

    def reset(self) -> None:
        """Reset episode index to zero."""
        self._current_step_idx = 0

    def set_step_index(self, step_idx: int) -> None:
        """Set the active step index."""
        self._current_step_idx = step_idx

    @property
    def target_profile(self) -> NDArray:
        if self._target_profile is None:
            raise RuntimeError("Target profile is not loaded.")
        return self._target_profile

    @property
    def projected_solution(self) -> NDArray:
        """Returns instantaneous projected snapshot for the current step."""
        return self.query(self._current_step_idx)
