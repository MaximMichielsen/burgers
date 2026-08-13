"""1D FEM Galerkin solver for the viscous Burgers equation.

Time-marching performed using RK2 and Mass-lumping.
SGS is modeled analytically.

Simulation modes
----------------
dns, no_model           - Pure Galerkin, no SGS model.
shakib             - Galerkin + analytic VMS/SGS stabilisation (τ-based).
"""

import csv
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Generator, Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problem
from utils.io_utils import compute_adjusted_dt


class BaseRK2:
    """Burgers FEM solver : M·U_t + A(U)·U + ν·K₀·U + C_fs(U) = f.

    Explicit time marching (RK2) is used with mass-lumping."""

    # TODO: add valid sgs models modes.
    _VALID_SIMULATION_MODES: frozenset[str] = frozenset({"dns", "no_model", "shakib"})
    _VALID_BC_TYPES: frozenset[str] = frozenset({"dirichlet", "fixed", "periodic"})

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretizationConfig,
        simulation_mode: str,
        master_path: Path,
        snapshot_factor: int = 1,
        t_start: float = 0.0,
    ) -> None:

        if simulation_mode not in self._VALID_SIMULATION_MODES:
            raise ValueError(
                f"Unknown simulation_mode {simulation_mode!r}. "
                f"Expected one of {self._VALID_SIMULATION_MODES}."
            )

        if problem.boundary_condition_type not in self._VALID_BC_TYPES:
            raise ValueError(
                f"Unknown boundary_condition_type {problem.boundary_condition_type!r}. "
                f"Expected one of {self._VALID_BC_TYPES}."
            )

        self.problem_name = problem.name

        # Simulation settings
        self.simulation_mode: str = simulation_mode
        self.domain_timespan: float = problem.domain_timespan
        self.simulation_time_elapsed: float = t_start
        self.domain_length: float = problem.domain_length
        self._dt: float = (
            disc_cfg.dt_dns if simulation_mode == "dns" else disc_cfg.dt_les
        )
        self.dt, self._n_time_steps = compute_adjusted_dt(
            self._dt, self.domain_timespan
        )
        self.time_steps: NDArray = np.linspace(
            t_start, t_start + self.domain_timespan, self._n_time_steps + 1
        )
        self.viscosity: float = float(problem.viscosity)

        # Mesh
        self.n_nodes: int = (
            disc_cfg.n_nodes_dns if simulation_mode == "dns" else disc_cfg.n_nodes_les
        )
        self.n_elements: int = self.n_nodes - 1
        self.nodes: NDArray = np.arange(0, self.n_nodes)
        self.boundary_nodes: set[int] = {int(self.nodes[0]), int(self.nodes[-1])}
        self.mesh: NDArray = np.linspace(0, self.domain_length, self.n_nodes)
        self.elements: NDArray = self.initialize_elements(self.nodes)
        self.element_size: float = self.domain_length / (self.n_nodes - 1)

        # Lumped Mass Matrix
        self.element_mass_matrix = self.element_size / 6 * np.array([[2, 1], [1, 2]])
        self.lumped_element_mass_matrix = self.element_mass_matrix.sum(axis=1)
        self.mass_lumped: NDArray = self.calculate_lumped_mass()
        assert np.isclose(self.mass_lumped.sum(), self.domain_length), (
            f"Lumped mass sum {self.mass_lumped.sum()} != domain_length {self.domain_length}"
        )
        self.inverted_mass: NDArray = 1.0 / self.mass_lumped

        # Boundary conditions
        self.boundary_condition_type: str = problem.boundary_condition_type
        self.boundary_condition_value: float | tuple[float, float] | None = (
            problem.boundary_condition_value
        )

        # Initial condition and solution
        self.initial_condition: NDArray = self.set_initial_condition(
            problem.initial_condition
        )
        self.solution: NDArray = self.initial_condition.copy()

        # Forcing
        self.forcing: NDArray | Callable | None = problem.forcing
        self.forcing_is_steady: bool = problem.forcing_is_steady
        self.forcing_current: NDArray | None = None

        # Output
        self.snapshot_factor = snapshot_factor
        self._snapshot_step_indices: frozenset[int] = frozenset(
            range(0, self._n_time_steps + 1, snapshot_factor)
        )
        self.requested_snapshots: NDArray = self.time_steps[
            sorted(self._snapshot_step_indices)
        ]
        self.is_written_at_times: list[bool] | None = (
            [False] * len(self.requested_snapshots)
            if self.requested_snapshots is not None
            else None
        )
        self.master_path: Path = master_path
        self.master_path.mkdir(parents=True, exist_ok=True)

        # Memory
        self.snapshots_solution: list[NDArray] = []
        self.snapshots_forcing: list[NDArray] = []

        # Benchmarking
        self.run_id: str = datetime.now().strftime("%m%d_%H%M%S")
        self.timings_performance: dict = {}
        self.residual_history: list = []
        self.update_history: list = []
        self.energy_history: list = []
        self.dissipation_history: list = []
        self.logger = self._setup_logger(
            suppress_file_logging=disc_cfg.suppress_file_logging
        )

    # ------------------------------------------------------------------ #
    #  Core Number Crunching
    # ------------------------------------------------------------------ #

    def run_simulation(self) -> None:
        """Run the full time-marching simulation and write output."""

        # add IC to snapshots
        self.resolve_current_forcing()
        self._extract_snapshot()

        with self.timer("total_simulation"):
            with tqdm(
                total=self._n_time_steps,
                desc=f"Eating Burgers | {self.throbber(0)}",
                file=sys.stdout,
            ) as pbar:
                for time_step in range(self._n_time_steps):
                    step_start = perf_counter()

                    self.advance_time_step()

                    if (time_step + 1) in self._snapshot_step_indices:
                        self._extract_snapshot()

                    pbar.set_description(f"Eating Burgers | {self.throbber(time_step)}")
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "t": f"{self.simulation_time_elapsed:.3f}",
                            "dt": f"{self.dt:.3f}",
                            "step_time": f"{perf_counter() - step_start:.3f}s",
                        }
                    )

            # end of simulation output
            self.write_config_to_json()
            self.write_solution_to_csv()

    # TODO: All outputting needs to occur at the initial condition and post-step for the complete picture

    def advance_time_step(self) -> None:
        """Advance the solution by one time step: U^{n+1} ← U^n."""
        self.resolve_current_forcing()
        self.solution = self.time_march_rk2(self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt

    def time_march_rk2(self, solution_current: NDArray) -> NDArray:
        """RK2 time marching."""
        k1 = self.inverted_mass * self.calculate_residual(
            solution_current, t=self.simulation_time_elapsed
        )
        predicted_coefficients = solution_current + self.dt * k1

        k2 = self.inverted_mass * self.calculate_residual(
            predicted_coefficients, t=self.simulation_time_elapsed + self.dt
        )

        return solution_current + self.dt / 2 * (k1 + k2)

    def calculate_residual(self, nodal_coefficients, t) -> NDArray:
        """Calculate the RHS residual."""
        

    def calculate_lumped_mass(self) -> NDArray:
        """Create lumped mass matrix."""
        n_nodes = self.n_nodes
        mass_lumped = np.zeros(n_nodes)

        for element in self.elements:
            mass_lumped[element] += self.lumped_element_mass_matrix

        return mass_lumped

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def initialize_elements(nodes) -> NDArray:
        """Build element connectivity array [[0,1], [1,2], …]."""
        return np.column_stack((nodes[:-1], nodes[1:]))

    def set_initial_condition(self, initial_condition: NDArray | Callable) -> NDArray:
        """Evaluate or copy the initial condition onto the mesh."""
        if callable(initial_condition):
            return initial_condition(self.mesh)
        return initial_condition.copy()

    @staticmethod
    def throbber(time_step: int, _every: int = 2) -> str:
        """Cycle through eating-animation states based on time step."""
        states = ["nom..        ", "nom nom..    ", "nom nom nom.."]
        return states[(time_step // _every) % len(states)]

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def resolve_current_forcing(self) -> None:
        if callable(self.forcing):
            self.forcing_current = (
                self.forcing(self.mesh, self.simulation_time_elapsed)
                if not self.forcing_is_steady
                else self.forcing(self.mesh)
            )
        elif self.forcing is None:
            self.forcing_current = np.zeros_like(self.solution)
        else:
            self.forcing_current = self.forcing

    def _extract_snapshot(self) -> None:
        """Store current solution and forcing as a snapshot."""
        self.snapshots_solution.append(self.solution.copy())
        self.snapshots_forcing.append(
            self.forcing_current.copy()
            if self.forcing_current is not None
            else np.zeros_like(self.solution)
        )

    # ------------------------------------------------------------------ #
    #  Timing
    # ------------------------------------------------------------------ #

    @contextmanager
    def timer(self, name: str) -> Generator[None, Any, None]:
        """Accumulate wall-clock time for a named phase."""
        start = perf_counter()
        yield
        self.timings_performance[name] = (
            self.timings_performance.get(name, 0.0) + perf_counter() - start
        )

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #

    def write_solution_to_csv(self, save_path: Path | None = None) -> None:
        """Write extracted solution snapshots to CSV files."""
        solutions = self.snapshots_solution
        forcings = self.snapshots_forcing
        if self.requested_snapshots is None:
            return

        times = self.requested_snapshots[: len(solutions)]

        for solution, time_value, forcing in zip(solutions, times, forcings):
            master_path = save_path if save_path is not None else self.master_path
            filepath = master_path / f"sol_t{time_value:.6f}.csv"
            with open(filepath, mode="w", newline="") as file_handle:
                writer = csv.writer(file_handle)
                writer.writerow(["node_index", "x_coordinate", "velocity", "forcing"])
                for i in range(len(self.nodes)):
                    writer.writerow(
                        [
                            self.nodes[i],
                            self.mesh[i],
                            solution[i],
                            forcing[i],
                        ]
                    )

        print(f"wrote {len(solutions)} snapshots at {self.master_path}")

    def write_config_to_json(self) -> None:
        """Serialize run configuration to config.json in the run directory."""
        raw_config: dict = {
            "run_id": self.run_id,
            "simulation_mode": self.simulation_mode,
            "domain_timespan": self.domain_timespan,
            "domain_length": self.domain_length,
            "dt": self.dt,
            "n_time_steps": self._n_time_steps,
            "viscosity": self.viscosity,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "element_size": self.element_size,
            "boundary_condition_type": self.boundary_condition_type,
            "boundary_condition_value": self.boundary_condition_value,
            "snapshot_factor": self.snapshot_factor,
            "forcing": self.forcing,
            "forcing_is_steady": self.forcing_is_steady,
            "problem_name": self.problem_name,
        }

        config_serializable = {
            k: (
                v.tolist()
                if isinstance(v, np.ndarray)
                else f"<callable: {getattr(v, '__name__', repr(v))}>"
                if callable(v)
                else v
            )
            for k, v in raw_config.items()
        }

        with open(self.master_path / "config.json", "w") as file_handle:
            json.dump(config_serializable, file_handle, indent=2)

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #

    def _setup_logger(self, suppress_file_logging: bool = False) -> logging.Logger:
        """Initialize a file logger for this run."""
        logger_ = logging.getLogger(str(self.run_id))
        logger_.setLevel(logging.INFO)
        if logger_.handlers or suppress_file_logging:
            return logger_

        formatter = logging.Formatter("[%(levelname)s] - %(message)s")
        fh = logging.FileHandler(
            self.master_path / f"{self.run_id}.log", encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger_.addHandler(fh)
        logger_.propagate = False
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        return logger_
