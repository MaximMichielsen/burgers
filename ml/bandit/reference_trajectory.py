"""Reference mean profiles and projected snapshots used for RL reward signaling."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from setup.constants import Z_LENGTH_DNS


class BanditReferenceTrajectory:
    """Manages reference target profiles and projected snapshots for RL rewards.

    Two intervals can be active: a training interval and a testing interval.
    The most recently set one determines where `reset()` starts an episode.
    """

    def __init__(
        self,
        mean_target_profile_dir: Path,
        projected_w_path: Path,
        times_path: Path,
        n_nodes_les: int,
    ) -> None:
        self.target_mean_profile_dir = Path(mean_target_profile_dir)
        self.projected_w_path = Path(projected_w_path)
        self.w_field_times_path = Path(times_path)
        self.n_les_nodes = n_nodes_les

        # Intervals (set via set_active_training_interval / set_testing_interval)
        self.t_start: float | None = None
        self.t_end: float | None = None
        self.t_start_test: float | None = None
        self.t_end_test: float | None = None

        # Mode and snapshot indices
        self._is_test_mode = False
        self._train_start_idx = 0
        self._test_start_idx = 0
        self._start_idx = 0  # start index of the active mode
        self._current_step_idx = 0

        # Reference data (filled by load_data / interval setters)
        self._projected_w_data: NDArray | None = None
        self._snapshot_times: NDArray | None = None
        self._target_profile: NDArray | None = None
        self._reynolds_profile: NDArray | None = None
        self._raw_target_profile: NDArray | None = None  # only set if resampled
        self._raw_reynolds_profile: NDArray | None = None  # only set if resampled

        self.load_data()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load_data(self) -> None:
        """Load the projected field and snapshot times, and validate them.

        If a training interval is already set, its profiles and start index
        are reloaded as well.
        """
        if not self.projected_w_path.exists():
            raise FileNotFoundError(
                f"Projected w field missing at {self.projected_w_path}"
            )

        # Note: astype() copies, so the whole field ends up in memory.
        self._projected_w_data = np.load(self.projected_w_path, mmap_mode="r").astype(
            np.float64
        )
        self._snapshot_times = np.load(self.w_field_times_path)

        if len(self._snapshot_times) != len(self._projected_w_data):
            raise ValueError(
                f"Times ({len(self._snapshot_times)}) and projected w "
                f"({len(self._projected_w_data)}) have different lengths."
            )

        if self.t_start is not None and self.t_end is not None:
            self._load_profiles(is_test=False)
            self._train_start_idx = self._nearest_index(self.t_start, validate=True)
            self._start_idx = self._train_start_idx

    def _load_profiles(self, is_test: bool) -> None:
        """Load mean and Reynolds profiles for the training or testing interval.

        Profiles are resampled onto the LES grid if their node count differs.
        """
        if is_test:
            t_start, t_end = self.t_start_test, self.t_end_test
        else:
            t_start, t_end = self.t_start, self.t_end
        if t_start is None or t_end is None:
            raise RuntimeError("Training or testing intervals are not properly set.")

        mean_path = self.target_mean_profile_dir / f"w_mean_{t_start}_{t_end}.npy"
        reynolds_path = self.target_mean_profile_dir / f"ww_{t_start}_{t_end}.npy"

        if not mean_path.exists():
            raise FileNotFoundError(f"Target profile missing at {mean_path}")
        if not reynolds_path.exists():
            raise FileNotFoundError(f"Reynolds stress field missing at {reynolds_path}")

        mean_profile = np.load(mean_path)
        reynolds_profile = np.load(reynolds_path)

        n_dns_nodes = mean_profile.shape[-1]
        if n_dns_nodes == self.n_les_nodes:
            self._target_profile = mean_profile
            self._reynolds_profile = reynolds_profile
            return

        dns_grid = np.linspace(0.0, Z_LENGTH_DNS, n_dns_nodes)
        les_grid = np.linspace(0.0, Z_LENGTH_DNS, self.n_les_nodes)
        self._target_profile = np.interp(les_grid, dns_grid, mean_profile)
        self._reynolds_profile = np.interp(les_grid, dns_grid, reynolds_profile)
        self._raw_target_profile = mean_profile
        self._raw_reynolds_profile = reynolds_profile

    def _nearest_index(self, t: float, validate: bool = False) -> int:
        """Index of the snapshot closest to time `t`.

        With `validate=True`, raise if `t` is further from that snapshot than
        the median snapshot spacing (i.e. not covered by the projected field).
        """
        times = self._snapshot_times
        if times is None:
            raise RuntimeError("Reference data is not loaded.")

        idx = int(np.argmin(np.abs(times - t)))
        if validate and abs(times[idx] - t) > np.median(np.diff(times)):
            raise ValueError(
                f"t_start={t} is not covered by the projected field "
                f"(range {times[0]} to {times[-1]})."
            )
        return idx

    # ------------------------------------------------------------------ #
    # Interval selection
    # ------------------------------------------------------------------ #

    def set_active_training_interval(self, t_start: float, t_end: float) -> None:
        """Switch to training mode and load the profiles for [t_start, t_end]."""
        self.t_start, self.t_end = t_start, t_end
        self._is_test_mode = False

        self._load_profiles(is_test=False)
        self._train_start_idx = self._nearest_index(t_start)
        self._start_idx = self._train_start_idx

    def set_testing_interval(self, t_start: float, t_end: float) -> None:
        """Switch to testing mode and load the profiles for [t_start, t_end]."""
        self.t_start_test, self.t_end_test = t_start, t_end
        self._is_test_mode = True

        self._load_profiles(is_test=True)
        self._test_start_idx = self._nearest_index(t_start)
        self._start_idx = self._test_start_idx

    # ------------------------------------------------------------------ #
    # Episode control and queries
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the episode index to the start of the active interval."""
        self._current_step_idx = self._start_idx

    def set_step_index(self, step_idx: int) -> None:
        """Set the active step index."""
        self._current_step_idx = step_idx

    def query(self, step_idx: int) -> NDArray:
        """Projected solution field at `step_idx` (clamped to the valid range)."""
        if self._projected_w_data is None:
            raise RuntimeError("Reference data is not loaded.")

        safe_idx = min(max(0, step_idx), len(self._projected_w_data) - 1)
        return np.asarray(self._projected_w_data[safe_idx], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def initial_condition(self) -> NDArray:
        """Projected (nodal) w field at the start of the active interval."""
        return self.query(self._start_idx)

    @property
    def projected_solution(self) -> NDArray:
        """Instantaneous projected snapshot for the current step."""
        return self.query(self._current_step_idx)

    @property
    def target_profile(self) -> NDArray:
        if self._target_profile is None:
            raise RuntimeError("Target profile is not loaded.")
        return self._target_profile

    @property
    def reynolds_profile(self) -> NDArray:
        if self._reynolds_profile is None:
            raise RuntimeError("Reynolds profile is not loaded.")
        return self._reynolds_profile
