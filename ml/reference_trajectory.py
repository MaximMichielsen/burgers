"""Reference mean profiles and projected snapshots used for RL reward signaling."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from setup.config_discretization import DiscretizationConfig
from setup.constants import Z_LENGTH_DNS

DT_DNS = 1e-4


class ReferenceTrajectory:
    """Manages reference target profiles and projected snapshots for RL rewards.

    Two intervals can be active: a training interval and a testing interval.
    The most recently set one determines where `reset()` starts an episode.
    """

    def __init__(
        self,
        disc_config: DiscretizationConfig,
        mean_target_profile_dir: Path,
        projected_w_path: Path,
        forcing_data_path: Path,
        times_path_velocity: Path,
        times_path_forcing: Path,
    ) -> None:
        self.target_mean_profile_dir = Path(mean_target_profile_dir)
        self.projected_w_path = Path(projected_w_path)
        self.forcing_data_path = Path(forcing_data_path)
        self.w_field_times_path = Path(times_path_velocity)
        self.forcing_times_path = Path(times_path_forcing)

        self.n_les_nodes = disc_config.n_nodes_les
        self.dt_les = disc_config.dt_les

        # Intervals (set via set_active_interval / set_testing_interval)
        self.t_start: float | None = None
        self.t_end: float | None = None
        self.t_start_test: float | None = None
        self.t_end_test: float | None = None

        # Mode and snapshot indices
        self._is_test_mode = False
        self._train_start_idx = 0
        self._test_start_idx = 0
        self._start_idx = 0  # local step start index of the active mode
        self._current_step_idx = 0

        # Reference data (filled by load_data / interval setters)
        self._projected_w_data: NDArray | None = None
        self._forcing_data: NDArray | None = None
        self._snapshot_velocity_times: NDArray | None = None
        self._snapshot_forcing_times: NDArray | None = None
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

        if not self.forcing_data_path.exists():
            raise FileNotFoundError(f"Forcing data missing at {self.forcing_data_path}")

        # Note: astype() copies, so the whole field ends up in memory.
        self._projected_w_data = np.load(self.projected_w_path, mmap_mode="r").astype(
            np.float64
        )
        self._snapshot_velocity_times = np.load(self.w_field_times_path)

        self._forcing_data = np.load(self.forcing_data_path, mmap_mode="r").astype(
            np.float64
        )
        self._snapshot_forcing_times = np.load(self.forcing_times_path)

        assert self._snapshot_velocity_times is not None
        assert self._projected_w_data is not None
        assert self._forcing_data is not None
        assert self._snapshot_forcing_times is not None

        if len(self._snapshot_velocity_times) != len(self._projected_w_data):
            raise ValueError(
                f"Times ({len(self._snapshot_velocity_times)}) and projected w "
                f"({len(self._projected_w_data)}) have different lengths."
            )

        if len(self._snapshot_forcing_times) != len(self._forcing_data):
            raise ValueError(
                f"Times ({len(self._snapshot_forcing_times)}) and forcing "
                f"({len(self._forcing_data)}) have different lengths."
            )

        if self.t_start is not None and self.t_end is not None:
            self._load_profiles(is_test=False)
            self._validate_time_coverage(self.t_start)
            self._start_idx = 0

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

    def _validate_time_coverage(self, t: float) -> None:
        """Ensure `t` falls within the loaded velocity and forcing time arrays."""
        if (
            self._snapshot_velocity_times is None
            or self._snapshot_forcing_times is None
        ):
            raise RuntimeError("Reference data is not loaded.")

        vel_idx = int(np.argmin(np.abs(self._snapshot_velocity_times - t)))
        if abs(self._snapshot_velocity_times[vel_idx] - t) > np.median(
            np.diff(self._snapshot_velocity_times)
        ):
            raise ValueError(
                f"t={t} is not covered by the projected velocity field "
                f"(range {self._snapshot_velocity_times[0]} to {self._snapshot_velocity_times[-1]})."
            )

        forcing_idx = int(np.argmin(np.abs(self._snapshot_forcing_times - t)))
        if abs(self._snapshot_forcing_times[forcing_idx] - t) > np.median(
            np.diff(self._snapshot_forcing_times)
        ):
            raise ValueError(
                f"t={t} is not covered by the forcing field "
                f"(range {self._snapshot_forcing_times[0]} to {self._snapshot_forcing_times[-1]})."
            )

    # ------------------------------------------------------------------ #
    # Interval selection
    # ------------------------------------------------------------------ #

    def set_active_interval(self, t_start: float, t_end: float) -> None:
        """Switch to training mode and load the profiles for [t_start, t_end]."""
        self.t_start, self.t_end = t_start, t_end
        self._is_test_mode = False

        self._load_profiles(is_test=False)
        self._validate_time_coverage(t_start)
        self._train_start_idx = 0
        self._start_idx = self._train_start_idx

    def set_testing_interval(self, t_start: float, t_end: float) -> None:
        """Switch to testing mode and load the profiles for [t_start, t_end]."""
        self.t_start_test, self.t_end_test = t_start, t_end
        self._is_test_mode = True

        self._load_profiles(is_test=True)
        self._validate_time_coverage(t_start)
        self._test_start_idx = 0
        self._start_idx = self._test_start_idx

    # ------------------------------------------------------------------ #
    # Episode control and queries
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the episode step index to the start of the active interval."""
        self._current_step_idx = self._start_idx

    def set_step_index(self, step_idx: int) -> None:
        """Set the active local LES step index."""
        self._current_step_idx = step_idx

    @property
    def _active_t_start(self) -> float:
        """Start time of the currently active (training or testing) interval."""
        t0 = self.t_start_test if self._is_test_mode else self.t_start
        if t0 is None:
            raise RuntimeError("Active interval is not set.")
        return t0

    def _step_to_time(self, les_step_idx: int) -> float:
        """Convert a local LES step index into physical simulation time."""
        return self._active_t_start + les_step_idx * self.dt_les

    def query_velocity(self, les_step_idx: int) -> NDArray:
        """Projected solution field at `les_step_idx` mapped via physical time."""
        if self._projected_w_data is None or self._snapshot_velocity_times is None:
            raise RuntimeError("Reference data is not loaded.")

        t = self._step_to_time(les_step_idx)
        vel_idx = int(np.argmin(np.abs(self._snapshot_velocity_times - t)))
        safe_idx = min(max(0, vel_idx), len(self._projected_w_data) - 1)
        return np.asarray(self._projected_w_data[safe_idx], dtype=np.float64)

    def query_forcing(self, les_step_idx: int) -> NDArray:
        """Forcing field at `les_step_idx` mapped via physical time."""
        if self._forcing_data is None or self._snapshot_forcing_times is None:
            raise RuntimeError("Forcing data is not loaded.")

        t = self._step_to_time(les_step_idx)
        forcing_idx = int(np.argmin(np.abs(self._snapshot_forcing_times - t)))
        safe_idx = min(max(0, forcing_idx), len(self._forcing_data) - 1)
        return np.asarray(self._forcing_data[safe_idx], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def initial_condition(self) -> NDArray:
        """Projected (nodal) w field at the start of the active interval."""
        return self.query_velocity(self._start_idx)

    @property
    def initial_forcing(self) -> NDArray:
        """Forcing field at the start of the active interval."""
        return self.query_forcing(self._start_idx)

    @property
    def projected_solution(self) -> NDArray:
        """Instantaneous projected snapshot for the current LES step."""
        return self.query_velocity(self._current_step_idx)

    @property
    def forcing(self) -> NDArray:
        """Forcing data for the current LES step."""
        return self.query_forcing(self._current_step_idx)

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
