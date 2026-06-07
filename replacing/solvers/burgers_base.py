"""1D FEM Galerkin solver for the viscous Burgers equation.

Simulation modes
----------------
dns      - Pure Galerkin, no SGS model.
no_model - Same as dns; use when solver settings are coarse (LES-like).
les      - Galerkin + analytic VMS/SGS stabilisation (τ-based).
sgsp      - Galerkin + ANN-predicted SGS corrections; analytic VMS disabled.
"""

import csv
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Generator, Iterable

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from replacing.constants import (
    MAXIMUM_ITERATIONS_DNS,
    MAXIMUM_ITERATIONS_LES,
)
from replacing.input_settings.disc_config import DiscretisationConfig
from replacing.input_settings.problems import Problem

logger = logging.getLogger(__name__)


def compute_adjusted_dt(
    dt_nominal: float, time_end: float | int, time_start: float = 0.0
) -> tuple[float, int]:
    """Adjust dt so that time_end is hit exactly.

    Rounds to the nearest integer step count and recomputes dt.
    The relative change in dt is O(1/n_steps), negligible in practice.
    """
    time_span = time_end - time_start
    n_steps = max(1, round(time_span / dt_nominal))
    dt_adjusted = time_span / n_steps
    return dt_adjusted, n_steps


class BurgersBase:
    """Burgers FEM solver: M·U_t + A(U)·U + ν·K₀·U + C_fs(U) = f."""

    _VALID_SIMULATION_MODES: frozenset[str] = frozenset(
        {"dns", "no_model", "les", "sgsp", "avc"}
    )
    _VALID_BC_TYPES: frozenset[str] = frozenset({"dirichlet", "fixed", "periodic"})

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretisationConfig,
        simulation_mode: str,
        master_path: Path,
        snapshot_factor: int = 1,
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

        # Simulation settings
        self.simulation_mode: str = simulation_mode
        self.domain_timespan: float = problem.domain_timespan
        self.simulation_time_elapsed: float = 0.0
        self.domain_length: float = problem.domain_length
        self._dt: float = (
            disc_cfg.dt_dns if simulation_mode == "dns" else disc_cfg.dt_dns
        )
        self.dt, self._n_time_steps = compute_adjusted_dt(
            self._dt, self.domain_timespan
        )
        self.time_steps: NDArray = np.linspace(0, self.domain_timespan, self._n_time_steps + 1)
        self.viscosity: float = problem.viscosity

        self.max_iterations: int = (
            MAXIMUM_ITERATIONS_DNS
            if simulation_mode == "dns"
            else MAXIMUM_ITERATIONS_LES
        )

        # Mesh
        self.n_nodes: int = (
            disc_cfg.n_nodes_dns if simulation_mode == "dns" else disc_cfg.n_nodes_les
        )
        self.n_elements: int = self.n_nodes - 1
        self.nodes: NDArray = np.arange(0, self.n_nodes)
        self.boundary_nodes: set[int] = {int(self.nodes[0]), int(self.nodes[-1])}
        self.mesh: NDArray = np.linspace(0, self.domain_length, self.n_nodes)
        self.elements: NDArray = self.initialize_elements()
        self.element_size: float = self.domain_length / (self.n_nodes - 1)

        # Boundary conditions
        self.boundary_condition_type: str = problem.boundary_condition_type
        self.boundary_condition_value = problem.boundary_condition_value

        # Initial condition and solution
        self.initial_condition: NDArray = self.set_initial_condition(
            problem.initial_condition
        )
        self.solution: NDArray = self.initial_condition.copy()

        # Forcing
        self.forcing: NDArray | Callable | None = problem.forcing
        self.forcing_is_steady: bool = problem.forcing_is_steady
        self.forcing_current: NDArray | None = None

        # Benchmarking
        self.run_id: str = datetime.now().strftime("%m%d_%H%M%S")
        self.timings_performance: dict = {}
        self.residual_history: list = []
        self.update_history: list = []
        self.energy_history: list = []
        self.dissipation_history: list = []

        # Output
        self.snapshot_factor: int = snapshot_factor
        self.requested_snapshots: NDArray = self.time_steps[::snapshot_factor]
        self.is_written_at_times: list[bool] = [False] * len(self.requested_snapshots)

        self.snapshots: list[NDArray] = []
        self.snapshots_solution: list[NDArray] = []
        self.snapshots_forcing: list[NDArray] = []

        self.master_path: Path = master_path
        self.master_path.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------ #
    #  Main solver loop
    # ------------------------------------------------------------------ #

    def run_simulation(self) -> None:
        """Run the full time-marching simulation and write output."""
        total_steps = int(self.domain_timespan / self.dt)
        self.snapshots_solution = []
        self.snapshots_forcing = []
        idx_extract = 0

        with self.timer("total_simulation"):
            with tqdm(
                total=total_steps,
                desc=f"Eating Burgers | {self.throbber(0)}",
                file=sys.stdout,
            ) as pbar:
                for time_step in range(total_steps):
                    step_start = perf_counter()
                    self.time_steps.append(time_step)
                    self.advance_time_step()
                    idx_extract = self._maybe_extract_solution(idx_extract)
                    pbar.set_description(f"Eating Burgers | {self.throbber(time_step)}")
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "t": f"{self.simulation_time_elapsed:.3f}",
                            "dt": f"{self.dt:.3f}",
                            "step_time": f"{perf_counter() - step_start:.3f}s",
                        }
                    )

            if self.requested_snapshots is not None:
                while idx_extract < len(self.requested_snapshots):
                    self.snapshots_solution.append(self.solution.copy())
                    self.snapshots_forcing.append(
                        self.forcing_current.copy()
                        if self.forcing_current is not None
                        else np.zeros_like(self.solution)
                    )
                    logger.info(
                        "Extracted solution at t=%.4f (end-of-simulation flush)",
                        self.requested_snapshots[idx_extract],
                    )
                    idx_extract += 1

        if self.write_solutions:
            self.write_config_to_json()
            self.write_solution_to_csv()

    def advance_time_step(self) -> None:
        """Advance the solution by one time step: U^{n+1} ← U^n."""
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

        self.solution = self.nr_iteration(self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """Newton–Raphson iteration; returns U^{n+1}."""
        solution_n = solution.copy()
        solution_k = solution.copy()
        residual_history_loop: list = []
        update_history_loop: list = []

        for _ in range(self.max_iterations):
            with self.timer("elemental_iterations"):
                elemental_residuals, elemental_jacobians = zip(
                    *(
                        self.calculate_elemental_residual_jacobian(
                            element=element,
                            u_k=solution_k[element],
                            u_n=solution_n[element],
                            f_e=(
                                self.forcing_current[element]
                                if self.forcing_current is not None
                                else None
                            ),
                        )
                        for element in self.elements
                    )
                )

            with self.timer("global_assembly"):
                global_residual, global_jacobian = self.global_assembly(
                    elemental_residuals, elemental_jacobians
                )

            global_residual, global_jacobian = self._apply_boundary_conditions(
                global_residual, global_jacobian, solution_k
            )
            residual_history_loop.append(np.linalg.norm(global_residual))

            with self.timer("linear_solve"):
                delta_u = np.linalg.solve(global_jacobian, -global_residual)
                if self.boundary_condition_type == "periodic":
                    delta_u_full = np.zeros_like(solution_k)
                    delta_u_full[:-1] = delta_u
                    delta_u_full[-1] = delta_u[0]
                    delta_u = delta_u_full

            update_history_loop.append(np.linalg.norm(delta_u))

            with self.timer("solution_update"):
                solution_k += (
                    delta_u * (1 - self.relaxation_factor)
                    if self.relaxation_factor is not None
                    else delta_u
                )

            with self.timer("convergence_checking"):
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
        f_e: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Compute R_e ∈ ℝ² and J_e ∈ ℝ²ˣ² for one element."""
        residual_element = np.zeros(2)
        jacobian_element = np.zeros((2, 2))

        weight = abs(self.element_size / 2)
        points, weights = self.gauss_legendre(3)
        gradient_basis = self.reference_gradient_basis_functions()

        for g_p, g_w in zip(points, weights):
            basis = self.reference_basis_functions(g_p)
            interp_fields = self._interpolate_fields(basis, gradient_basis, u_k, u_n)
            midpoint_fields = self._midpoint_fields(interp_fields)
            strong_res = self._strong_residual(interp_fields)
            scale = g_w * weight
            f_interp = float(basis @ f_e) if f_e is not None else 0.0

            for i in range(len(element)):
                residual_element[i] += scale * self._residual_integrand(
                    i, basis, gradient_basis, interp_fields, midpoint_fields, f_interp
                )
                if self._use_vms:
                    residual_element[i] += scale * self._vms_residual_integrand(
                        i, gradient_basis, interp_fields
                    )
                for j in range(len(element)):
                    jacobian_element[i, j] += scale * self._jacobian_integrand(
                        i, j, basis, gradient_basis, interp_fields
                    )
                    if self._use_vms:
                        jacobian_element[i, j] += scale * self._vms_jacobian_integrand(
                            i, j, basis, gradient_basis, interp_fields, strong_res
                        )

        return residual_element, jacobian_element

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
        return self._apply_periodic_bcs(global_residual, global_jacobian)

    def _apply_dirichlet_bcs(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce Dirichlet BCs via row-replacement."""
        for node in self.boundary_nodes:
            global_residual[node] = solution_k[node] - self.boundary_condition_value
            global_jacobian[node, :] = 0
            global_jacobian[node, node] = 1
        return global_residual, global_jacobian

    @staticmethod
    def _apply_periodic_bcs(
        global_residual: NDArray,
        global_jacobian: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce periodic BCs by folding the last DOF into the first."""
        global_residual[0] += global_residual[-1]
        global_jacobian[0, :] += global_jacobian[-1, :]
        global_jacobian[:, 0] += global_jacobian[:, -1]
        return global_residual[:-1], global_jacobian[:-1, :-1]

    # ------------------------------------------------------------------ #
    #  Global assembly
    # ------------------------------------------------------------------ #

    def global_assembly(
        self,
        elemental_residuals: Iterable[NDArray],
        elemental_jacobians: list[NDArray],
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
    #  Analytical SGS model (VMS)
    # ------------------------------------------------------------------ #

    def compute_tau(self, variable_u: float | NDArray) -> float:
        """VMS stabilisation parameter τ."""
        term_time = (2 / self.dt) ** 2
        term_adv = (2 * abs(variable_u) / self.element_size) ** 2
        term_diff = (4 * self.viscosity / self.element_size**2) ** 2
        return 0.5 / np.sqrt(term_time + term_adv + term_diff)

    # ------------------------------------------------------------------ #
    #  Newton–Raphson convergence
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

    # ------------------------------------------------------------------ #
    #  Integrand helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _interpolate_fields(
        basis: NDArray,
        gradient_basis: NDArray,
        u_k: NDArray,
        u_n: NDArray,
    ) -> dict:
        """Interpolate solution and gradient at a quadrature point."""
        return {
            "u_k": basis @ u_k,
            "u_n": basis @ u_n,
            "du_k": gradient_basis @ u_k,
            "du_n": gradient_basis @ u_n,
        }

    @staticmethod
    def _midpoint_fields(f: dict[str, float]) -> dict[str, float]:
        """Crank–Nicolson midpoint averages."""
        return {
            "u_mid": 0.5 * (f["u_k"] + f["u_n"]),
            "du_mid": 0.5 * (f["du_k"] + f["du_n"]),
        }

    def _strong_residual(self, f: dict[str, float]) -> float:
        """Strong-form residual at the midpoint for VMS stabilisation."""
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])
        return (f["u_k"] - f["u_n"]) / self.dt + u_mid * du_mid

    def _residual_integrand(
        self,
        i: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        mid: dict[str, float],
        f_interp: float = 0.0,
    ) -> float:
        """Galerkin weak-form residual integrand for node i."""
        time_derivative = basis[i] * (f["u_k"] - f["u_n"]) / self.dt
        diffusion = self.viscosity * mid["du_mid"] * gradient_basis[i]
        advection = basis[i] * mid["u_mid"] * mid["du_mid"]
        forcing = basis[i] * f_interp
        return time_derivative + diffusion + advection - forcing

    def _vms_residual_integrand(
        self,
        i: int,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        """VMS fine-scale residual contribution for node i."""
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])
        return (u_mid * gradient_basis[i]) * self.compute_tau(u_mid) * u_mid * du_mid

    def _jacobian_integrand(
        self,
        i: int,
        j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        """Galerkin Jacobian integrand for nodes i, j."""
        mass = basis[i] * basis[j] / self.dt
        stiffness = self.viscosity * gradient_basis[i] * gradient_basis[j]
        advection = basis[i] * (basis[j] * f["du_k"] + f["u_k"] * gradient_basis[j])
        return mass + 0.5 * (stiffness + advection)

    def _vms_jacobian_integrand(
        self,
        i: int,
        j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        strong_res: float,
    ) -> float:
        """VMS Jacobian contribution for nodes i, j."""
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])
        tau_mid = self.compute_tau(u_mid)
        spatial_part = basis[j] * du_mid + u_mid * gradient_basis[j]
        tau_part = basis[j] * gradient_basis[i] * tau_mid * strong_res
        return spatial_part + tau_part

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def initialize_elements(self) -> NDArray:
        """Build element connectivity array [[0,1], [1,2], …]."""
        return np.column_stack((self.nodes[:-1], self.nodes[1:]))

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
    #  FEM primitives
    # ------------------------------------------------------------------ #

    @staticmethod
    def gauss_legendre(number_of_points: int) -> tuple[Any, Any]:
        """Return Gauss–Legendre quadrature points and weights."""
        return np.polynomial.legendre.leggauss(deg=number_of_points)

    @staticmethod
    def reference_basis_functions(ksi: float) -> NDArray:
        """Linear basis functions on the reference element [-1, 1]."""
        return np.array([0.5 * (1 - ksi), 0.5 * (1 + ksi)])

    def reference_gradient_basis_functions(self) -> NDArray:
        """Gradient of linear basis functions mapped to physical space."""
        return np.array([-0.5, 0.5]) * (2 / self.element_size)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def _use_vms(self) -> bool:
        """True only for analytic VMS mode (les)."""
        return self.simulation_mode == "les"

    @property
    def _use_sgsp(self) -> bool:
        """True only for ANN-coupled mode (sgsp)."""
        return self.simulation_mode == "sgsp"

    @property
    def total_convergence_history(self) -> tuple[NDArray, NDArray]:
        """Flattened residual and update norm history across all time steps."""
        return (
            np.array(list(chain.from_iterable(self.residual_history))),
            np.array(list(chain.from_iterable(self.update_history))),
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
    #  Logging and output
    # ------------------------------------------------------------------ #

    def _format_config_for_display(self) -> dict[str, str]:
        """Shared config formatter used by both print and log output."""
        formatted: dict[str, str] = {}
        skip_keys = {
            "simulation_mode",
            "solution_initial",
            "master_path",
            "initial_condition",
        }
        for key, value in self.configuration.items():
            if key in skip_keys:
                continue
            if key == "extract_at_times":
                formatted[key] = (
                    f"{len(value)} extractions, first 5: {[float(t) for t in value[:5]]}"
                    if value is not None
                    else "None"
                )
            elif key in ("convergence_tol_residual", "convergence_tol_update"):
                formatted[key] = f"{value:.2e}"
            elif key == "external_forcing":
                if callable(value):
                    formatted[key] = (
                        f"{value.__name__} (from {getattr(value, '__module__', '')})"
                    )
                elif value is None:
                    formatted[key] = "None"
                else:
                    formatted[key] = f"array, shape {np.array(value).shape}"
            elif isinstance(value, float):
                formatted[key] = f"{value:.6g}"
            else:
                formatted[key] = str(value)
        return formatted

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
        _row("total steps", str(int(self.domain_timespan / self.dt)))
        if self.requested_snapshots is not None:
            n_ext = len(self.requested_snapshots)
            first_five = "  ".join(f"{t:.4f}" for t in self.requested_snapshots[:5])
            _row("extractions", str(n_ext))
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
        _row("tol residual", f"{self.configuration['convergence_tol_residual']:.2e}")
        _row("tol update", f"{self.configuration['convergence_tol_update']:.2e}")
        _row("relaxation", str(self.relaxation_factor))
        _row("objective", str(self.configuration.get("run_objective", "N/A")))

        # --- Paths ---
        _section("Paths")
        _row("output", str(self.master_path))

        _sep("═")

    def _setup_logger(self) -> logging.Logger:
        """Initialise a file logger for this run; skipped when suppress_file_logging is set."""
        logger_ = logging.getLogger(str(self.run_id))
        logger_.setLevel(logging.INFO)
        if logger_.handlers:
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

        if self.residual_history:
            res, upd = self.total_convergence_history
            self.logger.info(
                "Residual — initial: %.4e  final: %.4e  max: %.4e",
                res[0],
                res[-1],
                np.max(res),
            )
            self.logger.info(
                "Update   — initial: %.4e  final: %.4e  max: %.4e",
                upd[0],
                upd[-1],
                np.max(upd),
            )
            tol = self.configuration.get("convergence_tol_residual", 1e-6)
            status = (
                "CONVERGED"
                if res[-1] < tol
                else f"NOT CONVERGED (final {res[-1]:.4e} > tol {tol:.4e})"
            )
            self.logger.info("Status: %s", status)

        self.logger.info("=" * 60)

    def write_config_to_json(self) -> None:
        """Serialise run configuration to config.json in the run directory."""
        config_serializable = {
            k: (
                v.tolist()
                if isinstance(v, np.ndarray)
                else f"<callable: {getattr(v, '__name__', repr(v))}>"
                if callable(v)
                else v
            )
            for k, v in self.configuration.items()
            if k not in ("solution_initial", "master_path")
        }
        config_serializable["run_id"] = self.run_id
        with open(self.master_path / "config.json", "w") as file_handle:
            json.dump(config_serializable, file_handle, indent=2)

    def write_solution_to_csv(self) -> None:
        """Write extracted solution snapshots to CSV files."""
        if self.requested_snapshots is None:
            solutions = [self.solution]
            forcings = [self.forcing_current]
            times = [self.simulation_time_elapsed]
        else:
            solutions = self.snapshots_solution
            forcings = self.snapshots_forcing
            times = self.requested_snapshots[: len(solutions)]

        for solution, time_value, forcing in zip(solutions, times, forcings):
            filepath = self.master_path / f"sol_t{time_value:.6f}.csv"
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

    # ------------------------------------------------------------------ #
    #  Post-plotting
    # ------------------------------------------------------------------ #

    @staticmethod
    def moving_stats(arr: list, window: int = 5) -> tuple[NDArray, NDArray]:
        """Rolling mean and std for convergence plots."""
        arr = np.array(arr)
        if arr.size == 0:
            return np.array([]), np.array([])
        w = min(window, len(arr))
        means = np.convolve(arr, np.ones(w) / w, mode="same")
        stds = np.array(
            [
                np.std(arr[max(0, i - w // 2) : min(len(arr), i + w // 2 + 1)])
                for i in range(len(arr))
            ]
        )
        return means, stds

    def post_plotting(self, show_plot: bool = False) -> None:
        """Plot solution and convergence diagnostics; save to disk."""
        first_res = [r[0] for r in self.residual_history if r]
        last_res = [r[-1] for r in self.residual_history if r]
        first_upd = [u[0] for u in self.update_history if u]
        last_upd = [u[-1] for u in self.update_history if u]

        fr_mean, fr_std = self.moving_stats(first_res)
        lr_mean, lr_std = self.moving_stats(last_res)
        fu_mean, fu_std = self.moving_stats(first_upd)
        lu_mean, lu_std = self.moving_stats(last_upd)

        sgs_label = {
            "dns": "DNS",
            "les": "LES-VMS",
            "sgsp": "LES-SGSP",
            "avc": "LES-AVC",
        }.get(self.simulation_mode, self.simulation_mode)

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 2)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            self.mesh,
            self.solution,
            color="royalblue",
            linestyle="-",
            marker="o",
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
        ax0.set_title(f"Solution  [SGS: {sgs_label}]")

        ax1 = fig.add_subplot(gs[1, 0])
        t_axis = np.arange(len(fr_mean))
        for mean, std, color, style, label in [
            (fr_mean, fr_std, "royalblue", "-", "Residual (first)"),
            (lr_mean, lr_std, "navy", "--", "Residual (last)"),
            (fu_mean, fu_std, "tab:orange", "-", "Update (first)"),
            (lu_mean, lu_std, "darkorange", "--", "Update (last)"),
        ]:
            ax1.plot(t_axis, mean, color=color, linestyle=style, label=label)
            ax1.fill_between(t_axis, mean - std, mean + std, color=color, alpha=0.15)
        tol_r = self.configuration["convergence_tol_residual"]
        tol_u = self.configuration["convergence_tol_update"]
        ax1.axhline(
            y=tol_r,
            color="lightskyblue" if tol_r != tol_u else "lightgray",
            linestyle="--",
        )
        if tol_r != tol_u:
            ax1.axhline(y=tol_u, color="lightsalmon", linestyle="--")
        ax1.set_yscale("log")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Norm")
        ax1.set_title("Global convergence (smoothed)")
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(gs[1, 1])
        if self.residual_history:
            ax2.plot(
                self.residual_history[-1], "o-", label="Residual", color="royalblue"
            )
            if self.update_history[-1]:
                ax2.plot(
                    self.update_history[-1], "x--", label="Update", color="tab:orange"
                )
        ax2.set_yscale("log")
        ax2.set_xlabel("Newton iteration")
        ax2.set_ylabel("Norm")
        ax2.set_title("Last Newton iteration convergence")
        ax2.grid(True)
        ax2.legend()

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(
            self.time_steps, self.energy_history, color="red", label="Total energy"
        )
        ax3.plot(
            self.time_steps,
            self.dissipation_history,
            color="purple",
            label="Dissipation",
        )
        ax3.set_xlabel("Time step")
        ax3.set_title("Energy and dissipation evolution")
        ax3.grid(True)
        ax3.legend()

        ax4 = fig.add_subplot(gs[2, 1])
        wn, sp = self.get_positive_spectrum(
            *self.compute_energy_spectrum(self.solution)
        )
        ax4.loglog(wn[1:], sp[1:], marker="o")
        ax4.set_xlabel("Wavenumber k")
        ax4.set_ylabel("E(k)")
        ax4.set_title("Spectral analysis (final state)")
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.master_path / f"post_plotting_{self.simulation_mode}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Post-simulation plot saved to: {self.master_path / f'post_plotting_{self.simulation_mode}.png'}"
        )
        plt.show() if show_plot else plt.close(fig)

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
                    0.5
                    * g_w
                    * abs(jacobian)
                    * (self.reference_basis_functions(g_p) @ u_e) ** 2
                )
        return energy

    def compute_dissipation(self, solution: NDArray) -> float:
        """Integrate ν(∂u/∂x)² over the domain via Gaussian quadrature."""
        dissipation = 0.0
        jacobian = self.element_size / 2
        dn_dx = self.reference_gradient_basis_functions()
        points, weights = self.gauss_legendre(2)
        for element in self.elements:
            u_e = solution[element]
            for g_p, g_w in zip(points, weights):
                dissipation += self.viscosity * g_w * abs(jacobian) * (dn_dx @ u_e) ** 2
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

    @staticmethod
    def get_positive_spectrum(
        wavenumbers: NDArray, spectrum: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Filter to non-negative wavenumbers."""
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _maybe_extract_solution(self, idx_extract: int) -> int:
        """Extract and store the current solution if the next checkpoint time is reached."""
        if self.requested_snapshots is None:
            return idx_extract
        if (
            idx_extract < len(self.requested_snapshots)
            and self.simulation_time_elapsed >= self.requested_snapshots[idx_extract]
        ):
            self.snapshots_solution.append(self.solution.copy())
            self.snapshots_forcing.append(
                self.forcing_current.copy()
                if self.forcing_current is not None
                else np.zeros_like(self.solution)
            )
            idx_extract += 1
        return idx_extract
