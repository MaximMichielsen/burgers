"""Scheduler for mean profile and instantaneous snapshots for reward signaling."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class ReferenceTrajectory:
    """Manages reference target profiles and projected snapshots for RL rewards."""

    def __init__(
        self,
        target_profile_dir: Path,
        projected_w_path: Path,
        times_path: Path,
        n_les_nodes: int,
        t_start: float,
        t_end: float,
        t_start_test: float | None = None,
        t_end_test: float | None = None,
        test_mode: bool = False,
    ) -> None:

        self.target_mean_profile_dir = Path(target_profile_dir)
        self.projected_w_path = Path(projected_w_path)
        self.n_les_nodes = n_les_nodes
        self.t_start = t_start
        self.t_end = t_end

        self.t_start_test = t_start_test
        self.t_end_test = t_end_test

        self.mean_w_path = (
            self.target_mean_profile_dir / f"w_mean_{self.t_start}_{self.t_end}.npy"
        )
        self.reynolds_path = (
            self.target_mean_profile_dir / f"ww_{self.t_start}_{self.t_end}.npy"
        )
        self.w_field_times_path = times_path

        self._test_mode = test_mode

        self._start_idx: int = 0
        self._train_start_idx: int = 0
        self._start_test_idx: int = 0
        self._current_step_idx: int = 0
        self._target_profile: NDArray | None = None
        self._reynolds_profile: NDArray | None = None
        self._projected_w_data: NDArray | None = None

        self.load_data()

    @property
    def test_mode(self) -> bool:
        return self._test_mode

    @test_mode.setter
    def test_mode(self, value: bool) -> None:
        self._test_mode = value
        if self._test_mode:
            self._start_idx = self._start_test_idx
        else:
            self._start_idx = self._train_start_idx
        # Reload profiles when switching modes to use test intervals if applicable
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Helper to load correct mean and Reynolds profiles depending on training vs testing mode."""
        if (
            self._test_mode
            and self.t_start_test is not None
            and self.t_end_test is not None
        ):
            mean_w_path = (
                self.target_mean_profile_dir
                / f"w_mean_{self.t_start_test}_{self.t_end_test}.npy"
            )
            reynolds_path = (
                self.target_mean_profile_dir
                / f"ww_{self.t_start_test}_{self.t_end_test}.npy"
            )
        else:
            mean_w_path = (
                self.target_mean_profile_dir / f"w_mean_{self.t_start}_{self.t_end}.npy"
            )
            reynolds_path = (
                self.target_mean_profile_dir / f"ww_{self.t_start}_{self.t_end}.npy"
            )

        if not mean_w_path.exists():
            raise FileNotFoundError(f"Target profile missing at {mean_w_path}")
        if not reynolds_path.exists():
            raise FileNotFoundError(f"Reynolds stress field missing at {reynolds_path}")

        raw_target_profile = np.load(mean_w_path)
        raw_reynolds_profile = np.load(reynolds_path)

        n_dns_nodes = raw_target_profile.shape[-1]
        if n_dns_nodes != self.n_les_nodes:
            dns_grid = np.linspace(0.0, 1.0, n_dns_nodes)
            les_grid = np.linspace(0.0, 1.0, self.n_les_nodes)
            self._target_profile = np.interp(les_grid, dns_grid, raw_target_profile)
            self._reynolds_profile = np.interp(les_grid, dns_grid, raw_reynolds_profile)
        else:
            self._target_profile = raw_target_profile
            self._reynolds_profile = raw_reynolds_profile

    def load_data(self) -> None:
        """Loads reference datasets into memory and project target profile if needed."""
        if not self.projected_w_path.exists():
            raise FileNotFoundError(
                f"Projected w field missing at {self.projected_w_path}"
            )

        # Load appropriate profiles initially
        self._load_profiles()

        # Time-resolved projected field (Memory-mapped query interface)
        self._projected_w_data = np.load(self.projected_w_path, mmap_mode="r").astype(
            np.float64
        )

        # Shape validation
        if (
            self._target_profile is not None
            and self._target_profile.shape[-1] != self.n_les_nodes
        ):
            raise ValueError(
                f"Target profile node count ({self._target_profile.shape[-1]}) "
                f"mismatches LES grid ({self.n_les_nodes})."
            )

        snapshot_times = np.load(self.w_field_times_path)
        if len(snapshot_times) != len(self._projected_w_data):
            raise ValueError(
                f"Times ({len(snapshot_times)}) and projected w "
                f"({len(self._projected_w_data)}) have different lengths."
            )

        self._train_start_idx = int(np.argmin(np.abs(snapshot_times - self.t_start)))
        if self.t_start_test is not None:
            self._start_test_idx = int(
                np.argmin(np.abs(snapshot_times - self.t_start_test))
            )

        max_offset = np.median(np.diff(snapshot_times))
        if abs(snapshot_times[self._train_start_idx] - self.t_start) > max_offset:
            raise ValueError(
                f"t_start={self.t_start} is not covered by the projected field "
                f"(range {snapshot_times[0]} to {snapshot_times[-1]})."
            )
        if (
            self.t_start_test is not None
            and abs(snapshot_times[self._start_test_idx] - self.t_start_test)
            > max_offset
        ):
            raise ValueError(
                f"t_start_test={self.t_start_test} is not covered by the projected field "
                f"(range {snapshot_times[0]} to {snapshot_times[-1]})."
            )

        self._start_idx = (
            self._start_test_idx if self._test_mode else self._train_start_idx
        )

    def query(self, step_idx: int) -> NDArray:
        """Query projected solution field at a specific timestep index."""
        if self._projected_w_data is None:
            raise RuntimeError("Reference data is not loaded.")

        # Safeguard index boundaries
        safe_idx = min(max(0, step_idx), len(self._projected_w_data) - 1)
        return np.asarray(self._projected_w_data[safe_idx], dtype=np.float64)

    def reset(self) -> None:
        """Reset episode index to start index."""
        self._current_step_idx = self._start_idx

    def set_step_index(self, step_idx: int) -> None:
        """Set the active step index."""
        self._current_step_idx = step_idx

    @property
    def initial_condition(self) -> NDArray:
        """Projected (nodal) w field at t_start."""
        return self.query(self._start_idx)

    @property
    def target_profile(self) -> NDArray:
        if self._target_profile is None:
            raise RuntimeError("Target profile is not loaded.")
        return self._target_profile

    @property
    def projected_solution(self) -> NDArray:
        """Returns instantaneous projected snapshot for the current step."""
        return self.query(self._current_step_idx)

    @property
    def reynolds_profile(self) -> NDArray:
        if self._reynolds_profile is None:
            raise RuntimeError("Reynolds profile is not loaded.")
        return self._reynolds_profile
