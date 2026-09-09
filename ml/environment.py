import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.projection_schedule import ProjectionReferenceSchedule
from ml.tau_ann import TauANNConfig
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_coupled import SolverCoupled
from utils.io_utils import compute_adjusted_dt


class EnvironmentTauAnn:
    """MDP wrapper around SolverCoupled for the TauANN training."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        tau_ann_config: TauANNConfig,
        master_path: Path,
        proj_ref_schedule: ProjectionReferenceSchedule,
    ) -> None:

        self.problem = problem
        self.disc_config = disc_config
        self.tau_ann_config = tau_ann_config
        self.master_path = master_path
        self.proj_ref_schedule = proj_ref_schedule

        _, self._n_time_steps = compute_adjusted_dt(
            disc_config.dt_les, problem.domain_timespan
        )
        self._max_les_steps: int = self._n_time_steps
        self._total_les_steps: int = 0

        self.n_wavenumber_bins: int = tau_ann_config.n_wavenumber_bins
        self.state_dim: int = tau_ann_config.input_dimension
        self.action_dim: int = tau_ann_config.n_coefficients

        self.solver: SolverCoupled | None = None

    def reset(self) -> NDArray:
        """Instantiate a fresh BurgersAVC solver and return initial state sₙ."""
        self.solver = SolverCoupled(
            training_mode=True,
            problem=self.problem,
            disc_config=dataclasses.replace(
                self.disc_config, suppress_file_logging=True
            ),
            simulation_mode="tau_model",
            master_path=self.master_path,
            tau_model=self.tau_ann_config.tau_model,
            ann_path=None,
        )
        self._total_les_steps = 0
        if self.solver is None:
            raise ValueError("Something went wrong with setting the solver :(")

        return self.solver.create_input_stencil()

    def step(self, action: NDArray) -> tuple[NDArray, float, bool]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done)."""
        assert self.solver is not None, "Call reset() before step()."

        self.solver.correction_coefficients = action

        for _ in range(self.tau_ann_config.n_skip_steps):
            self.solver.advance_time_step()
            self._total_les_steps += 1

        reward_val = self.compute_reward()
        done_flag = self._total_les_steps >= self._max_les_steps
        next_state_array = self.solver.create_input_stencil()

        return next_state_array, reward_val, done_flag

    def compute_reward(self) -> float:
        assert self.solver is not None

        wavenumbers_all, raw_spectrum_all = self.solver._compute_energy_spectrum(
            self.solver.solution
        )
        _, positive_spectrum = self.solver.get_positive_spectrum(
            wavenumbers_all, raw_spectrum_all
        )

        spectrum_k = positive_spectrum.astype(np.float64)

        proj_spectrum_k = self.proj_ref_schedule.query(
            self.solver.simulation_time_elapsed
        )

        if len(spectrum_k) != len(proj_spectrum_k):
            raise ValueError(
                f"Spectrum length mismatch between live LES ({len(spectrum_k)}) "
                f"and reference schedule ({len(proj_spectrum_k)}). Check n_wavenumber_bins alignment."
            )

        w_energy = self.tau_ann_config.reward_weight_energy
        gamma_exp = self.tau_ann_config.reward_spectral_exponent
        wavenumber_indices = np.arange(1, len(spectrum_k) + 1, dtype=np.float64)

        spectral_penalty = float(
            np.sum(
                w_energy
                * wavenumber_indices**gamma_exp
                * ((spectrum_k - proj_spectrum_k) / (np.mean(proj_spectrum_k) + 1e-12))
                ** 2
            )
        )

        return -spectral_penalty
