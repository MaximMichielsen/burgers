"""Solver used for proofing runs."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.ann_config import SMOOTHING_FACTOR
from proofing.ann_config import ANNConfigProof
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode, TauModel
from solvers.solver_coupled import SolverCoupled


class SolverForProofs(SolverCoupled):
    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        master_path: Path,
        tau_model: TauModel,
        ann_config: ANNConfigProof,
        simulation_mode: SimulationMode = SimulationMode.TAU_BASED,
        snapshot_factor: int = 1,
        prescribed_action_trajectory: list | None = None,
    ):

        super().__init__(
            problem,
            disc_config,
            master_path,
            tau_model,
            ann_config,
            simulation_mode,
            snapshot_factor,
            prescribed_action_trajectory,
        )

        self.ann_config: ANNConfigProof = ann_config
        self.proof_mode = ann_config.proof_mode

        self.current_action_mean: NDArray | None = None
        self.action_mean_history: list[NDArray] = []

    def create_input_stencil(self, mean_profile: NDArray) -> NDArray:
        """Create input stencil based on the proof mode."""
        previous_coefficients = (
            self.correction_coefficients
            if self.correction_coefficients is not None
            else np.ones(self.ann_config.action_dimension)
        )

        if self.proof_mode == "b":
            return previous_coefficients

        elif self.proof_mode == "c":
            freq = self.ann_config.frequency
            omega = self.ann_config.omega
            phase = self.ann_config.phase
            phase_input = np.array(
                [
                    np.sin(freq * (omega * self.time)),
                    np.cos(freq * (omega * self.time)),
                    np.sin(freq * (self.time + phase)),
                    np.cos(freq * (self.time + phase)),
                ]
            )
            return np.concatenate([previous_coefficients, phase_input])

        elif self.proof_mode == "d":
            if self.correction_coefficients is None:
                self.update_running_action_mean(
                    action=np.ones(self.ann_config.action_dimension)
                )

            else:
                self.update_running_action_mean(action=self.correction_coefficients)

            return np.concatenate([previous_coefficients, self.current_action_mean])

        return super().create_input_stencil(mean_profile=self.mean_solution)

    def update_running_action_mean(self, action: NDArray) -> None:
        """Update the running action mean with an exponential moving average."""
        action = np.asarray(action, dtype=np.float64)
        smoothing_factor = float(
            getattr(self.ann_config, "smoothing_factor", SMOOTHING_FACTOR)
        )
        if self.current_action_mean is None:
            self.current_action_mean = action.copy()
        else:
            self.current_action_mean = (
                1.0 - smoothing_factor
            ) * self.current_action_mean + smoothing_factor * action
        self.action_mean_history.append(self.current_action_mean.copy())
