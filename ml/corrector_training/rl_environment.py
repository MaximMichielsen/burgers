import dataclasses
from pathlib import Path

from numpy.typing import NDArray

from ml.corrector_training.before_rk2.projection_schedule import ProjectionReferenceSchedule
from ml.ml_agents.corrector import AVCConfig
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
        simulation_mode: str,
        master_path: Path,
        proj_ref_schedule: ProjectionReferenceSchedule,
    ):
        self.problem = problem
        self.disc_config = disc_config
        self.avc_config = avc_config
        self.master_path = master_path
        self.simulation_mode = simulation_mode

        _, self._n_time_steps = compute_adjusted_dt(
            disc_config.dt_les, problem.domain_timespan
        )
        self._max_les_steps: int = self._n_time_steps
        self._total_les_steps: int = 0

        self.n_wavenumber_bins = avc_config.n_wavenumber_bins


    def reset(self) -> NDArray:
        """Instantiate a fresh BurgersAVC solver and return initial state sₙ."""
        self._solver = AVCSolverRK2(
            problem=self.problem,
            disc_cfg=dataclasses.replace(self.disc_config, suppress_file_logging=True),
            simulation_mode=self.simulation_mode,
            master_path=self.master_path,
            avc_cfg=self.avc_config,
        )
        self._total_les_steps = 0
        if self._solver is None:
            raise ValueError("Something went wrong with setting the solver :(")

        return self._solver.create_avc_input_stencil()
