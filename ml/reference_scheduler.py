"""Scheduler for mean profile and instantaneous snapshots for reward signaling."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class ReferenceTrajectory:
    """Manages reference target profiles and projected snapshots for RL rewards."""

    def __init__(
        self,
        target_profile_path: Path,
        projected_w_path: Path,
        n_les_nodes: int,
        target_time_index: int,
    ) -> None:

        self.target_profile_path = Path(target_profile_path)
        self.projected_w_path = Path(projected_w_path)
        self.n_les_nodes = n_les_nodes
        self.target_time_index = target_time_index

        self._current_step_idx: int = 0
        self._target_profile: NDArray | None = None
        self._projected_w_data: NDArray | None = None

        self.load_data()

    def load_data(self) -> None:
        """Loads reference datasets into memory and project target profile if needed."""
        if not self.target_profile_path.exists():
            raise FileNotFoundError(
                f"Target profile missing at {self.target_profile_path}"
            )
        if not self.projected_w_path.exists():
            raise FileNotFoundError(
                f"Projected w field missing at {self.projected_w_path}"
            )

        # Static 1D target mean profile <u_DNS>_T
        raw_target_profile = np.load(self.target_profile_path).astype(np.float64)[
            self.target_time_index
        ]

        # Nodal projection (linear interpolation) if DNS grid size != LES node count
        n_dns_nodes = raw_target_profile.shape[-1]
        if n_dns_nodes != self.n_les_nodes:
            dns_grid = np.linspace(0.0, 1.0, n_dns_nodes)
            les_grid = np.linspace(0.0, 1.0, self.n_les_nodes)
            self._target_profile = np.interp(les_grid, dns_grid, raw_target_profile)
        else:
            self._target_profile = raw_target_profile

        # Time-resolved projected field (Memory-mapped query interface)
        self._projected_w_data = np.load(self.projected_w_path, mmap_mode="r").astype(
            np.float64
        )

        # Shape validation
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
