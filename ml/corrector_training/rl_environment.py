import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.corrector_training.projection_schedule import (
    ProjectionReferenceSchedule,
)
from ml.ml_agents.corrector import AVCConfig, AVCTrainingConfig
from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problem
from solvers.explicit.avc_augment_rk2 import AVCSolverRK2
from utils.io_utils import compute_adjusted_dt


class AVCEnvironment:
    """MDP wrapper around BurgersAVC for the AV corrector control problem."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        avc_config: AVCConfig,
        avc_training_config: AVCTrainingConfig,
        simulation_mode: str,
        master_path: Path,
        proj_ref_schedule: ProjectionReferenceSchedule,
    ):
        self.problem = problem
        self.disc_config = disc_config
        self.avc_config = avc_config
        self.master_path = master_path
        self.simulation_mode = simulation_mode
        self.training_config = avc_training_config
        self.projection_ref_schedule = proj_ref_schedule

        _, self._n_time_steps = compute_adjusted_dt(
            disc_config.dt_les, problem.domain_timespan
        )
        self._max_les_steps: int = self._n_time_steps
        self._total_les_steps: int = 0

        self.n_wavenumber_bins = avc_config.n_wavenumber_bins

        self.correction_mode: str = avc_config.output_scope

        self._solver: AVCSolverRK2 | None = None

    def reset(self) -> NDArray:
        """Instantiate a fresh BurgersAVC solver and return initial state sₙ."""
        self._solver = AVCSolverRK2(
            problem=self.problem,
            disc_config=dataclasses.replace(self.disc_config, suppress_file_logging=True),
            simulation_mode=self.simulation_mode,
            master_path=self.master_path,
            avc_config=self.avc_config,
        )
        self._total_les_steps = 0
        if self._solver is None:
            raise ValueError("Something went wrong with setting the solver :(")

        return self._solver.create_avc_input_stencil()

    def step(self, alpha_action: float) -> tuple[NDArray, float, bool]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done)."""
        assert self._solver is not None, "Call reset() before step()."

        if self.correction_mode == "global":
            self._solver.av_correction = alpha_action
        else:
            raise ValueError(f"Wrong correction mode chosen {self.correction_mode}")

        for _ in range(self.avc_config.n_skip_steps):
            self._solver.advance_time_step()
            self._total_les_steps += 1

        reward_value = self.compute_reward()
        done_flag = self._total_les_steps >= self._max_les_steps
        next_state_array = self._solver.create_avc_input_stencil()

        return next_state_array, reward_value, done_flag

    def compute_reward(self) -> float:
        """Compute rₙ from eq. (2.10)."""

        assert self._solver is not None

        wavenumbers_all, raw_spectrum_all = self._solver.compute_energy_spectrum(
            self._solver.solution
        )
        _, positive_spectrum = self._solver.get_positive_spectrum(
            wavenumbers_all, raw_spectrum_all
        )
        spectrum_k = positive_spectrum.astype(np.float64)

        w_energy = self.training_config.reward_weight_energy
        gamma_exp = self.training_config.reward_spectral_exponent

        proj_spectrum_k = self.projection_ref_schedule.query(
            self._solver.simulation_time_elapsed
        )

        spectrum_k = spectrum_k[1:]
        proj_spectrum_k = proj_spectrum_k[1:]
        wavenumber_indices = np.arange(1, len(spectrum_k) + 1, dtype=np.float64)

        spectral_penalty = float(
            np.sum(
                w_energy
                * wavenumber_indices**gamma_exp
                * ((spectrum_k - proj_spectrum_k) / (np.mean(proj_spectrum_k) + 1e-12))
                ** 2
            )
        )

        return -np.log1p(spectral_penalty)
