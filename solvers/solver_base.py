import csv
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Any, Sequence, Generator

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from setup.config_discretization import DiscretizationConfig
from utils.io_utils import compute_adjusted_dt
from setup.problems import Problem


TOLERANCE_RESIDUAL: float = 1e-6
TOLERANCE_UPDATE: float = 1e-6
MAXIMUM_ITERATIONS_DNS: int = 20
MAXIMUM_ITERATIONS_LES: int = 5


class SolverBase:
    """Burgers FEM solver: M·U_t + A(U)·U + ν·K₀·U + C_fs(U) = f."""

    _VALID_SIMULATION_MODES: frozenset[str] = frozenset(
        {"dns", "no_model", "tau_model"}
    )

    _VALID_TAU_MODES: frozenset[str] = frozenset({"2", "3", "3_dt_augmented"})

    _VALID_BC_TYPES: frozenset[str] = frozenset({"dirichlet", "fixed"})

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        simulation_mode: str,
        master_path: Path,
        tau_model: str | None = None,
        snapshot_factor: int = 1,
        t_start: float = 0.0,
    ) -> None:

        if simulation_mode not in self._VALID_SIMULATION_MODES:
            raise ValueError(
                f"Unknown simulation_mode {simulation_mode!r}. "
                f"Expected one of {self._VALID_SIMULATION_MODES}."
            )

        if simulation_mode == "tau_model" and tau_model is None:
            raise ValueError(
                f'Choose tau model if simulation mode is "tau_model". '
                f"Expected one of {self._VALID_TAU_MODES}"
            )

        if problem.boundary_condition_type not in self._VALID_BC_TYPES:
            raise ValueError(
                f"Unknown boundary_condition_type {problem.boundary_condition_type!r}. "
                f"Expected one of {self._VALID_BC_TYPES}."
            )

        self.problem_name = problem.name

        # simulation settings
        self.simulation_mode: str = simulation_mode
        self.tau_model: str | None = tau_model
        self.domain_timespan: float = problem.domain_timespan
        self.simulation_time_elapsed: float = t_start
        self.domain_length: float = problem.domain_length
        self._dt: float = (
            disc_config.dt_dns if simulation_mode == "dns" else disc_config.dt_les
        )
        self.dt, self._n_time_steps = compute_adjusted_dt(
            self._dt, self.domain_timespan
        )
        self.time_steps: NDArray = np.linspace(
            t_start, t_start + self.domain_timespan, self._n_time_steps + 1
        )
        self.viscosity: float = problem.viscosity
        self.max_iterations: int = (
            MAXIMUM_ITERATIONS_DNS
            if simulation_mode == "dns"
            else MAXIMUM_ITERATIONS_LES
        )
        self.convergence_tol_update = TOLERANCE_RESIDUAL
        self.convergence_tol_residual = TOLERANCE_UPDATE

        # mesh
        self.n_nodes: int = (
            disc_config.n_nodes_dns
            if simulation_mode == "dns"
            else disc_config.n_nodes_les
        )
        self.n_elements: int = self.n_nodes - 1
        self.nodes: NDArray = np.arange(0, self.n_nodes)
        self.boundary_nodes: set[int] = {int(self.nodes[0]), int(self.nodes[-1])}
        self.mesh: NDArray = np.linspace(0, self.domain_length, self.n_nodes)
        self.elements: NDArray = self.initialize_elements(self.nodes)
        self.element_size: float = self.domain_length / (self.n_nodes - 1)

        # boundary conditions
        self.boundary_condition_type: str = problem.boundary_condition_type
        self.boundary_condition_value: float | tuple[float, float] | None = (
            problem.boundary_condition_value
        )

        # initial condition
        self.initial_condition: NDArray = self.set_initial_condition(
            problem.initial_condition
        )
        self.solution: NDArray = self.initial_condition.copy()
        self.solution_previous: NDArray = self.initial_condition.copy()

        # forcing
        self.forcing: NDArray | Callable | None = problem.forcing
        self.forcing_is_steady: bool = problem.forcing_is_steady
        self.forcing_current: NDArray | None = None

        # output
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

        # snapshots
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
            suppress_file_logging=disc_config.suppress_file_logging
        )

    # ------------------------------------------------------------------ #
    #  Core
    # ------------------------------------------------------------------ #

    def run_simulation(self) -> None:
        """Run the full simulation and writes output."""
        self.resolve_current_forcing()  # add IC to snapshots
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

            self.write_config_to_json()
            self.write_solution_to_csv()

    def advance_time_step(self) -> None:
        """Advance the solution by one time step: U^{n+1} ← U^n.

        Previous solutions are stored for BDF2 time-marching."""
        self.resolve_current_forcing()

        new_solution = self.nr_iteration(self.solution, self.solution_previous)
        self.solution_previous = self.solution
        self.solution = new_solution

        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt

    def nr_iteration(self, solution: NDArray, solution_prev: NDArray) -> NDArray:
        """Newton–Raphson iteration; returns U^{n+1}."""
        solution_nm1 = solution_prev.copy()
        solution_n = solution.copy()
        solution_k = solution.copy()
        residual_history_loop: list = []
        update_history_loop: list = []

        for _ in range(self.max_iterations):
            elemental_residuals, elemental_jacobians = zip(
                *(
                    self.calculate_elemental_residual_jacobian(
                        element=element,
                        u_k=solution_k[element],
                        u_n=solution_n[element],
                        u_nm1=solution_nm1[element],
                        f_e=(
                            self.forcing_current[element]
                            if self.forcing_current is not None
                            else None
                        ),
                    )
                    for element in self.elements
                )
            )

            global_residual, global_jacobian = self.global_assembly(
                elemental_residuals, elemental_jacobians
            )

            global_residual, global_jacobian = self._apply_boundary_conditions(
                global_residual, global_jacobian, solution_k
            )
            residual_history_loop.append(np.linalg.norm(global_residual))

            delta_u = np.linalg.solve(global_jacobian, -global_residual)
            update_history_loop.append(np.linalg.norm(delta_u))

            solution_k += delta_u
            if self.is_update_converged(delta_u) or self.is_residual_converged(
                global_residual
            ):
                break

        self.residual_history.append(residual_history_loop)
        self.update_history.append(update_history_loop)
        return solution_k

    def calculate_elemental_residual_jacobian(
        self,
        element: tuple[int, int],
        u_k: NDArray,
        u_n: NDArray,
        u_nm1: NDArray,
        f_e: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Compute R_e ∈ ℝ² and J_e ∈ ℝ²ˣ² for one element."""
        residual_element = np.zeros(2)
        jacobian_element = np.zeros((2, 2))

        weight = abs(self.element_size / 2)
        points, weights = self.gauss_legendre(3)
        gradient_basis = self.basis_functions_gradient()

        tau_e = (
            self.compute_tau(u_k, self.basis_functions_gradient() @ u_k)
            if self._use_vms
            else None
        )

        for gauss_point, gauss_weight in zip(points, weights):
            basis = self.basis_functions(gauss_point)
            interpolated_fields = self._interpolate_fields(
                basis, gradient_basis, u_k, u_n, u_nm1
            )
            scale = gauss_weight * weight
            f_interp = float(basis @ f_e) if f_e is not None else 0.0
            strong_res = self._strong_residual(interpolated_fields)

            for e_i in range(len(element)):
                residual_element[e_i] += scale * self.residual_integrand(
                    e_i, basis, gradient_basis, interpolated_fields, f_interp
                )
                if tau_e is not None:
                    residual_element[e_i] += scale * self._vms_residual_integrand(
                        e_i,
                        gradient_basis,
                        interpolated_fields,
                        strong_res,
                        tau_e,
                    )
                for e_j in range(len(element)):
                    jacobian_element[e_i, e_j] += scale * self._jacobian_integrand(
                        e_i, e_j, basis, gradient_basis, interpolated_fields
                    )
                    if tau_e is not None:
                        jacobian_element[e_i, e_j] += (
                            scale
                            * self._vms_jacobian_integrand(
                                e_i,
                                e_j,
                                basis,
                                gradient_basis,
                                interpolated_fields,
                                strong_res,
                                tau_e,
                            )
                        )
        return residual_element, jacobian_element

    def global_assembly(
        self,
        elemental_residuals: Sequence[NDArray],
        elemental_jacobians: Sequence[NDArray],
    ) -> tuple[NDArray, NDArray]:
        """Assemble element contributions into global R and J."""
        global_residual = np.zeros(self.n_nodes)
        global_jacobian = np.zeros((self.n_nodes, self.n_nodes))

        for e, element in enumerate(self.elements):
            i, j = element
            global_residual[i] += elemental_residuals[e][0]
            global_residual[j] += elemental_residuals[e][1]
            global_jacobian[i, i] += elemental_jacobians[e][0, 0]
            global_jacobian[i, j] += elemental_jacobians[e][0, 1]
            global_jacobian[j, i] += elemental_jacobians[e][1, 0]
            global_jacobian[j, j] += elemental_jacobians[e][1, 1]

        return global_residual, global_jacobian

    # ------------------------------------------------------------------ #
    #  Integrand helpers
    # ------------------------------------------------------------------ #

    def _strong_residual(self, f: dict) -> float:
        """Calculates the strong residual using BDF2 for the time term."""
        return (1.5 * f["u_k"] - 2 * f["u_n"] + 0.5 * f["u_nm1"]) / self.dt + f[
            "u_k"
        ] * f["du_k"]

    def residual_integrand(
        self,
        e_i: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        forcing_interp: float = 0.0,
    ) -> float:
        """BDF2 implicit-Euler based residual integrand."""
        time_derivative = (
            basis[e_i] * (3.0 * f["u_k"] - 4 * f["u_n"] + f["u_nm1"]) / (2 * self.dt)
        )
        diffusion = self.viscosity * f["du_k"] * gradient_basis[e_i]
        advection = basis[e_i] * f["u_k"] * f["du_k"]
        forcing = basis[e_i] * forcing_interp
        return time_derivative + diffusion + advection - forcing

    def _jacobian_integrand(
        self,
        e_i: int,
        e_j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        """Endpoint Galerkin Jacobian integrand for nodes i, j."""
        mass = 1.5 * basis[e_i] * basis[e_j] / self.dt
        stiffness = self.viscosity * gradient_basis[e_i] * gradient_basis[e_j]
        advection = basis[e_i] * (
            basis[e_j] * f["du_k"] + f["u_k"] * gradient_basis[e_j]
        )
        return mass + stiffness + advection

    @staticmethod
    def _vms_residual_integrand(
        i, gradient_basis: NDArray, f: dict[str, float], strong_res, tau_e: float
    ) -> float:
        """Computes VMS part of the residual."""
        u_k = f["u_k"]
        return (u_k * gradient_basis[i]) * tau_e * strong_res

    def _vms_jacobian_integrand(
        self,
        i,
        j,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        strong_res,
        tau_e: float,
    ) -> float:
        u_k, du_k = f["u_k"], f["du_k"]
        d_a_duj = basis[j]  # d(u_k)/du_j
        d_strong_duj = 1.5 * basis[j] / self.dt + (
            basis[j] * du_k + u_k * gradient_basis[j]
        )
        return gradient_basis[i] * tau_e * (d_a_duj * strong_res + u_k * d_strong_duj)

    @staticmethod
    def _interpolate_fields(
        basis: NDArray,
        gradient_basis: NDArray,
        u_k: NDArray,
        u_n: NDArray,
        u_nm1: NDArray,
    ) -> dict:
        """Interpolate solution and gradient at a quadrature point."""
        return {
            "u_k": basis @ u_k,
            "u_n": basis @ u_n,
            "u_nm1": basis @ u_nm1,
            "du_k": gradient_basis @ u_k,
            "du_n": gradient_basis @ u_n,
        }

    # ------------------------------------------------------------------ #
    #  Tau models
    # ------------------------------------------------------------------ #

    def compute_tau(self, u_e: NDArray, u_x_e: NDArray | None = None) -> float:
        """Dispatch to the configured tau model. u_e: nodal values for the element (length 2)."""
        if self.tau_model == "2":
            return self.tau_model_two_params(u_e)
        elif self.tau_model == "3" and u_x_e is not None:
            return self.tau_model_three_params(u_e, u_x_e)
        elif self.tau_model == "3_dt_augmented" and u_x_e is not None:
            return self.tau_model_three_dt_aug(u_e, u_x_e)
        raise ValueError(f"Unknown tau_model {self.tau_model!r}")

    def tau_model_two_params(self, u_e: NDArray) -> float:
        """τ = [ (2⟨ū⟩_e/h)² + (4ν/h²)² ]^(-1/2), ⟨ū⟩_e = element-averaged u."""
        u_bar_e = 0.5 * (u_e[0] + u_e[1])
        term_adv = (2.0 * u_bar_e / self.element_size) ** 2
        term_diff = (4.0 * self.viscosity / self.element_size**2) ** 2
        return (term_adv + term_diff) ** -0.5

    def tau_model_three_params(
        self,
        u_e: NDArray,
        u_x_e,
        alpha: float = 0.099,
        beta: float = 9.39,
        gamma: float = 2.16,
    ) -> float:
        u_bar_e = 0.5 * (u_e[0] + u_e[1])
        u_x_bar_e = u_x_e  # gradient of u is constant for linear elements (right?)
        part_a = (alpha * u_bar_e / self.element_size) ** 2
        part_b = (beta * self.viscosity / self.element_size**2) ** 2
        part_c = (gamma * u_x_bar_e) ** 2
        return (part_a + part_b + part_c) ** -0.5

    def tau_model_three_dt_aug(
        self,
        u_e: NDArray,
        u_x_e,
        alpha: float = 0.099,
        beta: float = 9.39,
        gamma: float = 2.16,
        delta: float = 1.0,
    ) -> float:
        u_bar_e = 0.5 * (u_e[0] + u_e[1])
        u_x_bar_e = u_x_e  # gradient of u is constant for linear elements (right?)
        part_a = (alpha * u_bar_e / self.element_size) ** 2
        part_b = (beta * self.viscosity / self.element_size**2) ** 2
        part_c = (gamma * u_x_bar_e) ** 2
        part_dt = (delta * 2 / self.dt) ** 2
        return (part_a + part_b + part_c + part_dt) ** -0.5

    # ------------------------------------------------------------------ #
    #  FEM primitives
    # ------------------------------------------------------------------ #

    @staticmethod
    def gauss_legendre(number_of_points: int) -> tuple[Any, Any]:
        """Return Gauss–Legendre quadrature points and weights."""
        return np.polynomial.legendre.leggauss(deg=number_of_points)

    @staticmethod
    def basis_functions(ksi: float) -> NDArray:
        """Linear basis functions on the reference element [-1, 1]."""
        return np.array([0.5 * (1 - ksi), 0.5 * (1 + ksi)])

    def basis_functions_gradient(self) -> NDArray:
        """Gradient of linear basis functions mapped to physical space."""
        return np.array([-0.5, 0.5]) * (2 / self.element_size)

    @property
    def _use_vms(self) -> bool:
        """True only for analytic VMS mode (les)."""
        return self.simulation_mode == "tau_model"

    # ------------------------------------------------------------------ #
    #  Boundary conditions
    # ------------------------------------------------------------------ #

    def _apply_boundary_conditions(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Dispatch to Dirichlet or periodic BC enforcement."""
        if self.boundary_condition_type in ("dirichlet", "fixed"):
            return self._apply_dirichlet_bcs(
                global_residual, global_jacobian, solution_k
            )
        else:
            raise ValueError("only fixed bcs!")

    def _apply_dirichlet_bcs(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce Dirichlet BCs via row-replacement."""
        if self.boundary_condition_value is None:
            raise ValueError(
                "Applying fixed BCS requires a value of either float or tuple[float, float]!"
            )

        bc_values = (
            self.boundary_condition_value
            if isinstance(self.boundary_condition_value, tuple)
            else (self.boundary_condition_value, self.boundary_condition_value)
        )
        for node, bc_value in zip(sorted(self.boundary_nodes), bc_values):
            global_residual[node] = solution_k[node] - bc_value
            global_jacobian[node, :] = 0
            global_jacobian[node, node] = 1
        return global_residual, global_jacobian

    # ------------------------------------------------------------------  #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_residual_converged(
        residual: float | NDArray, tolerance: float = 1e-6
    ) -> bool:
        """True when the relative residual norm is below tolerance."""
        norm = np.linalg.norm(residual)
        return norm < tolerance * (1 + norm)

    @staticmethod
    def is_update_converged(correction: NDArray, tolerance: float = 1e-6) -> bool:
        """True when the update norm is below tolerance."""
        return np.linalg.norm(correction) < tolerance

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

    @contextmanager
    def timer(self, name: str) -> Generator[None, Any, None]:
        """Accumulate wall-clock time for a named phase."""
        start = perf_counter()
        yield
        self.timings_performance[name] = (
            self.timings_performance.get(name, 0.0) + perf_counter() - start
        )

    # ------------------------------------------------------------------  #
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

    def post_logging(self) -> None:
        """Write a structured run summary to the log file."""
        self.logger.info("=" * 60)
        self.logger.info("RUN COMPLETE — id: %s", self.run_id)
        self.logger.info("Time Integration: Second Order Implicit Euler")
        self.logger.info("Simulation mode: %s", self.simulation_mode)
        self.logger.info("-" * 40)

        for key, value in self._format_config_for_display().items():
            self.logger.info("  %-30s %s", key, value)

        self.logger.info("-" * 40)

        if self.timings_performance:
            total = self.timings_performance.get("total_simulation") or sum(
                v
                for k, v in self.timings_performance.items()
                if k != "total_simulation"
            )
            for phase, time_elapsed in sorted(self.timings_performance.items()):
                if phase != "total_simulation":
                    pct = (100 * time_elapsed / total) if total > 0 else float("nan")
                    self.logger.info(
                        "  %-25s %.4fs (%5.1f%%)", phase, time_elapsed, pct
                    )
            self.logger.info("  %-25s %.4fs", "TOTAL", total)
        self.logger.info("=" * 60)

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
            "max_iterations": self.max_iterations,
            "convergence_tol_residual": self.convergence_tol_residual,
            "convergence_tol_update": self.convergence_tol_update,
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

    def post_processing(self) -> None:
        """Run post-plotting and post-logging."""
        self.post_plotting()
        self.post_logging()

    def _format_config_for_display(self) -> dict[str, str]:
        """Format solver state for display in logs and console output."""
        if callable(self.forcing):
            forcing_str = f"{self.forcing.__name__} (from {getattr(self.forcing, '__module__', '')})"
        elif self.forcing is None:
            forcing_str = "None"
        else:
            forcing_str = f"array, shape {np.array(self.forcing).shape}"

        snapshots_str = (
            f"{len(self.requested_snapshots)} snapshots, first 5: "
            f"{[round(float(t), 4) for t in self.requested_snapshots[:5]]}"
            if self.requested_snapshots is not None
            else "None"
        )

        return {
            "domain_timespan": f"{self.domain_timespan:.6g}",
            "domain_length": f"{self.domain_length:.6g}",
            "dt": f"{self.dt:.4e}",
            "n_time_steps": str(self._n_time_steps),
            "viscosity": f"{self.viscosity:.4e}",
            "n_nodes": str(self.n_nodes),
            "n_elements": str(self.n_elements),
            "element_size": f"{self.element_size:.4e}",
            "boundary_condition_type": self.boundary_condition_type,
            "boundary_condition_value": str(self.boundary_condition_value),
            "max_iterations": str(self.max_iterations),
            "snapshot_factor": str(self.snapshot_factor),
            "requested_snapshots": snapshots_str,
            "forcing": forcing_str,
            "forcing_is_steady": str(self.forcing_is_steady),
            "problem_name": self.problem_name,
        }

    def print_configuration(self) -> None:
        """Print run configuration in a clean tabular format."""
        W = 72
        COL = 30

        def _row(label: str, value: str) -> None:
            print(f"  {label:<{COL}} {value}")

        def _sep(char: str = "─") -> None:
            print(char * W)

        def _section(title: str) -> None:
            print()
            print(f"  {title}")
            _sep()

        _sep("═")
        print(
            f"  Solver Configuration  ·  mode: {self.simulation_mode}  ·  integration: 2nd-order implicit Euler"
        )
        _sep("═")

        # --- Mesh ---
        _section("Mesh")
        _row("nodes", str(self.n_nodes))
        _row("elements", str(self.n_elements))
        _row("domain length", f"{self.domain_length:.4g}")
        _row("element size h", f"{self.element_size:.4e}")

        # --- Time ---
        _section("Time")
        _row("timespan", f"{self.domain_timespan:.4g}")
        _row("dt", f"{self.dt:.4e}")
        _row("total steps", str(self._n_time_steps))
        if self.requested_snapshots is not None:
            n_ext = len(self.requested_snapshots)
            first_five = "  ".join(f"{t:.4f}" for t in self.requested_snapshots[:5])
            _row("snapshots", str(n_ext))
            _row("  first 5 times", first_five)

        # --- Physics ---
        _section("Physics")
        _row("viscosity ν", f"{self.viscosity:.4e}")
        if callable(self.forcing):
            forcing_str = f"{self.forcing.__name__} (from {getattr(self.forcing, '__module__', '')})"
        elif self.forcing is None:
            forcing_str = "None"
        else:
            forcing_str = f"array, shape {np.array(self.forcing).shape}"
        _row("forcing", forcing_str)
        _row("forcing steady", str(self.forcing_is_steady))

        # --- Boundary conditions ---
        _section("Boundary conditions")
        _row("type", self.boundary_condition_type)
        _row("value", str(self.boundary_condition_value))

        # --- Solver ---
        _section("Solver")
        _row("max iterations", str(self.max_iterations))
        _row("tol residual", f"{self.convergence_tol_residual:.2e}")
        _row("tol update", f"{self.convergence_tol_update:.2e}")

        # --- Paths ---
        _section("Paths")
        _row("output", str(self.master_path))

        _sep("═")

    # ------------------------------------------------------------------ #
    #  Energy and spectral analysis
    # ------------------------------------------------------------------ #

    def compute_energy(self, solution: NDArray) -> float:
        """Integrate ½u² over the domain via Gaussian quadrature."""
        energy = 0.0
        jacobian = self.element_size / 2
        points, weights = self.gauss_legendre(2)
        for element in self.elements:
            u_e = solution[element]
            for g_p, g_w in zip(points, weights):
                energy += (
                    0.5 * g_w * abs(jacobian) * (self.basis_functions(g_p) @ u_e) ** 2
                )
        return energy

    def compute_dissipation(self, solution: NDArray) -> float:
        """Integrate ν(∂u/∂x)² over the domain."""
        dissipation = 0.0
        dn_dx = self.basis_functions_gradient()
        for element in self.elements:
            u_e = solution[element]
            dissipation += self.viscosity * self.element_size * (dn_dx @ u_e) ** 2
        return dissipation

    def compute_energy_spectrum(self, solution: NDArray) -> tuple[NDArray, NDArray]:
        """Return wavenumbers and spectral energy of the solution."""
        u_hat = np.fft.fft(solution)
        wavenumbers = (
            np.fft.fftfreq(len(solution), d=self.domain_length / len(solution))
            * 2
            * np.pi
        )
        spectrum = 0.5 * np.abs(u_hat) ** 2 / len(solution)
        return wavenumbers, spectrum

        # ------------------------------------------------------------------ #
        #  Post-plotting
        # ------------------------------------------------------------------ #

    @staticmethod
    def get_positive_spectrum(
        wavenumbers: NDArray, spectrum: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Filter to non-negative wavenumbers."""
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

    def post_plotting(self, show_plot: bool = False) -> None:
        """Plot solution and convergence diagnostics; save to disk."""
        sgs_label = {
            "dns": "DNS",
            "les": "LES-VMS",
            "sgsp": "LES-SGSP",
            "avc": "LES-AVC",
        }.get(self.simulation_mode, self.simulation_mode)

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            self.mesh,
            self.solution,
            color="royalblue",
            linestyle="-",
            linewidth=2.0 if self.simulation_mode == "dns" else 1.0,
            marker="none" if self.simulation_mode == "dns" else ".",
            label="Resolved solution",
        )
        ax0.plot(
            self.mesh,
            self.initial_condition,
            color="grey",
            linestyle="--",
            label="Initial solution",
        )
        ax0.set_xlabel(r"$x$")
        ax0.set_ylabel("Velocity")
        ax0.grid(True)
        ax0.legend()
        ax0.set_title(f"Solution  [Mode: {sgs_label}]")

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(
            self.time_steps[: len(self.energy_history)],
            self.energy_history,
            color="red",
            label="Total energy",
        )
        ax1.plot(
            self.time_steps[: len(self.dissipation_history)],
            self.dissipation_history,
            color="purple",
            label="Dissipation",
        )
        ax1.set_xlabel("Time step")
        ax1.set_title("Energy and dissipation evolution")
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(gs[1, 1])
        wn, sp = self.get_positive_spectrum(
            *self.compute_energy_spectrum(self.solution)
        )
        ax2.loglog(wn[1:], sp[1:], marker=".", color="orangered")
        ax2.set_xlabel("Wavenumber k")
        ax2.set_ylabel("E(k)")
        ax2.set_title("Spectral analysis (final state)")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.master_path / f"post_plotting_{self.simulation_mode}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Post-simulation plot saved to: {self.master_path / f'post_plotting_{self.simulation_mode}.png'}"
        )
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    # Verify solver using MMS, u(x,t) = sin(x) cos(t)
    CURRENT_DIR = Path(__file__).parent.parent.parent.resolve()
    path = CURRENT_DIR / "test_suite" / "manufactured_test"
    nu = 2 * np.pi / 100

    def manufactured_solution(x, t):
        return np.sin(x) * np.cos(t)

    def manufactured_forcing(x, t):
        a = -np.sin(x) * np.sin(t)
        b = np.sin(x) * np.cos(x) * (np.cos(t)) ** 2
        c = nu * np.sin(x) * np.cos(t)
        return a + b + c

    mms_problem = Problem(
        name="manufactured_problem",
        domain_length=2 * np.pi,
        domain_timespan=2 * np.pi,
        reynolds=100,  # ← was reynolds=100; solver needs .viscosity directly
        initial_condition=np.sin,
        forcing=manufactured_forcing,
        forcing_is_steady=False,
        boundary_condition_type="fixed",
        boundary_condition_value=0,
    )

    disc_cfg = DiscretizationConfig(
        n_nodes_les=16,
        temporal_refinement=1,
        courant_les=0.5,
        domain_length=2 * np.pi,
    )

    solver = SolverBase(
        problem=mms_problem,
        disc_config=disc_cfg,
        simulation_mode="no_model",
        tau_model="2",
        master_path=path,
    )

    solver.run_simulation()

    simulated_solution = solver.solution
    exact_solution = manufactured_solution(
        x=solver.mesh, t=solver.simulation_time_elapsed
    )

    plt.plot(solver.mesh, exact_solution, label="exact")
    plt.plot(solver.mesh, simulated_solution, label="simulated")
    plt.legend()
    plt.show()
