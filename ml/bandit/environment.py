import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ml.bandit.ann import BanditConfig, BanditHyperparameters
from ml.bandit.reference_trajectory import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode
from solvers.solver_coupled import SolverCoupled
from utils.diagnostics import compute_error_metrics


class BanditEnvironment:
    """Environment for Bandit training."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: BanditConfig,
        hyperparameters: BanditHyperparameters,
        reference_trajectory: ReferenceTrajectory,
        master_path: Path,
        baseline_path: Path | None,
    ) -> None:

        self.problem = problem
        self.disc_config = disc_config
        self.ann_config = ann_config
        self.hp = hyperparameters
        self.reference_trajectory = reference_trajectory
        self.master_path = master_path

        self.solver: SolverCoupled | None = None
        self.baseline_path = baseline_path

    def reset(self) -> float:
        self.solver = SolverCoupled(
            problem=self.problem,
            disc_config=dataclasses.replace(
                self.disc_config, suppress_file_logging=True
            ),
            ann_config=dataclasses.replace(
                self.ann_config,
                training_mode=True,
                ann_path=None,
            ),
            master_path=self.master_path,
            simulation_mode=SimulationMode.TAU_BASED,
            tau_model=self.ann_config.tau_model,
            baseline_mode=False,
        )
        if not getattr(self.ann_config, "is_contextual", False):
            state = 1
            if self.solver is not None:
                state = float(self.solver.create_input_stencil())
            return state

        return self.solver.create_input_stencil()

    def step(
        self, action: NDArray, is_baseline_run: bool = False
    ) -> tuple[float, float, Literal[True]]:
        if self.solver is None:
            raise RuntimeError("Call reset() before step().")

        self.solver.correction_coefficients = action
        for _ in range(self.disc_config.n_timesteps):
            self.solver.advance_time_step()

            if not np.all(np.isfinite(self.solver.solution)):
                raise FloatingPointError(
                    "NaN/Inf detected in intermediate solver step."
                )

            if self.solver.simulation_done:
                break

        reward_val = self.compute_reward(action, is_baseline_run)
        done_flag = True
        if not getattr(self.ann_config, "is_contextual", False):
            next_state = float(self.solver.create_input_stencil())
        else:
            next_state = self.solver.create_input_stencil()

        return next_state, reward_val, done_flag

    def compute_reward(self, action: NDArray, is_baseline_run: bool) -> float:
        if self.solver is None:
            raise RuntimeError("Solver not initialized.")

        mean_profile = np.mean(self.solver.snapshots_solution, axis=0)
        target_profile = self.reference_trajectory.target_profile

        mean_profile_error = self.compute_distance_error(mean_profile, target_profile)

        reynolds_error = 0
        if self.ann_config.include_reynolds_signal:
            reynolds_stress = self.solver.compute_reynolds_stress()
            target_reynolds = self.reference_trajectory.reynolds_profile
            reynolds_error = self.compute_distance_error(
                reynolds_stress, target_reynolds
            )

        penalty = (
            self.hp.weight_mean_profile * mean_profile_error
            + self.hp.weight_reynolds * reynolds_error
        )

        if not is_baseline_run:
            penalty_baseline = np.load(self.baseline_path / "reward_history.npy")

            # action regularization
            base_action = np.ones(self.ann_config.action_dimension)
            action_penalty = self.compute_distance_error(action, base_action)

            return penalty_baseline - penalty - action_penalty * 0.01

        return penalty

    @staticmethod
    def compute_distance_error(field: NDArray, target: NDArray) -> float:
        """Compute spatial L1 or RMS error (non-squared) between field and target vectors."""
        _, rms_error, _ = compute_error_metrics(field, target)
        return rms_error
