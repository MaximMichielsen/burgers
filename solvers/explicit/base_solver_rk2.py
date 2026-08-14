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
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problem
from utils.io_utils import compute_adjusted_dt


class BaseRK2:
    """Burgers FEM solver : M·U_t + A(U)·U + ν·K₀·U + C_fs(U) = f.

    Explicit time marching (RK2) is used with mass-lumping."""

    # TODO: add valid sgs models modes.
    _VALID_SIMULATION_MODES: frozenset[str] = frozenset(
        {"dns", "no_model", "tau_model"}
    )
    _VALID_TAU_MODELS: frozenset[str] = frozenset(
        {"shakib_one", "shakib_two", "shakib_three"}
    )
    _VALID_BC_TYPES: frozenset[str] = frozenset({"dirichlet", "fixed", "periodic"})

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretizationConfig,
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
                f"Expected one of {self._VALID_TAU_MODELS}"
            )

        if problem.boundary_condition_type not in self._VALID_BC_TYPES:
            raise ValueError(
                f"Unknown boundary_condition_type {problem.boundary_condition_type!r}. "
                f"Expected one of {self._VALID_BC_TYPES}."
            )

        self.problem_name = problem.name

        # Simulation settings
        self.simulation_mode: str = simulation_mode
        self.tau_model: str | None = tau_model
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
        """
        Assemble R(d) = F - C(d)d - K d - S(d) (VMS term only in 'shakib' mode).
        Does NOT apply M^-1 -- that's done by the caller (time_march_rk2).
        """
        residual = np.zeros_like(nodal_coefficients)

        gauss_points, gauss_weights = self.gauss_legendre(3)
        if not self.forcing_is_steady:  # wrapper to handle k_2 predicted time-step
            forcing = (
                self.forcing(self.mesh, t)
                if self.forcing_current is not None
                else np.zeros(self.n_nodes)
            )
        else:
            forcing = (
                self.forcing_current
                if self.forcing_current is not None
                else np.zeros(self.n_nodes)
            )
        h = self.element_size
        nu = self.viscosity
        dN_dx = self.reference_gradient_basis_functions()

        for element in self.elements:
            u_e = nodal_coefficients[element]
            f_e = forcing[element]

            local_residual = np.zeros(2)

            for gauss_point, gauss_weight in zip(gauss_points, gauss_weights):
                N = self.reference_basis_functions(gauss_point)
                u_interp = N @ u_e
                du_dx_interp = dN_dx @ u_e
                f_interp = N @ f_e
                jacobian = h / 2

                # Advection
                local_residual -= (
                    N * (u_interp * du_dx_interp) * gauss_weight * jacobian
                )

                # Diffusion
                local_residual -= dN_dx * (nu * du_dx_interp) * gauss_weight * jacobian

                # Forcing
                local_residual += N * f_interp * gauss_weight * jacobian

                if self.simulation_mode == "tau_model":
                    if self.tau_model == "shakib_one":
                        tau = self.tau_shakib_one(u_e)
                    elif self.tau_model == "shakib_two":
                        tau = self.tau_shakib_two(u_e)
                    elif self.tau_model == "shakib_three":
                        tau = self.tau_shakib_three(u_e)
                    else:
                        raise ValueError("qmdlkfjqdmlkfjqdl")
                    # strong-form residual, quasi-static closure:
                    # ∂²u/∂x² = 0 exactly for linear elements; ∂u/∂t dropped
                    # (algebraic sub-scale approximation)
                    strong_residual = u_interp * du_dx_interp - f_interp

                    # SUPG weighting: u_h ∂N_a/∂x
                    local_residual -= (
                        (u_interp * dN_dx)
                        * tau
                        * strong_residual
                        * gauss_weight
                        * jacobian
                    )

            residual[element] += local_residual

        if self.boundary_condition_type in ("dirichlet", "fixed"):
            for node in self.boundary_nodes:
                residual[node] = 0.0

        # TODO: add periodic handling

        return residual

    def tau_shakib_one(self, u_e) -> float:
        """Compute tau based on the Shakib model, taken from Wouter Edeling eq. 6.8"""
        elemental_average = (u_e[0] + u_e[1]) / 2
        h = self.element_size
        a = 2 * elemental_average / h
        b = 4 * self.viscosity / h**2
        return (a**2 + b**2) ** (-0.5)

    # TODO: what is the a in eq 6.9? i assumed this is the same term as the eq 6.8
    def tau_shakib_two(self, u_e) -> float:
        """Compute tau based on the Shakib model, taken from Wouter Edeling eq. 6.9"""
        elemental_average = (u_e[0] + u_e[1]) / 2
        h = self.element_size
        a = 2 * elemental_average / h
        b = 4 * self.viscosity / h**2
        return (a**2 + 9 * b**2) ** (-0.5)

    def tau_shakib_three(
        self, u_e, alpha: float = 0.099, beta: float = 9.39, gamma: float = 2.16
    ) -> float:
        """Compute tau based on the Shakib model, taken from Wouter Edeling eq. 6.10"""
        h = self.element_size
        elemental_average = (u_e[0] + u_e[1]) / 2
        elemental_gradient = (u_e[1] - u_e[0]) / h

        a = alpha * elemental_average / h
        b = beta * self.viscosity / h**2
        c = gamma * elemental_gradient

        return (a**2 + b**2 + c**2) ** (-0.5)

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
        """Integrate ν(∂u/∂x)² over the domain."""
        dissipation = 0.0
        dn_dx = self.reference_gradient_basis_functions()
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

    @staticmethod
    def get_positive_spectrum(
        wavenumbers: NDArray, spectrum: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Filter to non-negative wavenumbers."""
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

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

    def post_processing(self) -> None:
        """Run post-plotting and post-logging."""
        self.post_plotting()
        self.post_logging()

    # ------------------------------------------------------------------ #
    #  Post-plotting
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #

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
            f"  Solver Configuration  ·  mode: {self.simulation_mode}  ·  Time-Marching: RK2"
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

        # --- Paths ---
        _section("Paths")
        _row("output", str(self.master_path))

        _sep("═")

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
        self.logger.info("Time Integration: RK2")
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


if __name__ == "__main__":
    # Verify solver using MMS, u(x,t) = sin(x) cos(t)
    CURRENT_DIR = Path(__file__).parent.parent.parent.resolve()
    path = CURRENT_DIR / "test_suite" / "manufactured_test"
    reynolds = 100
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
        reynolds=100,
        initial_condition=np.sin,
        forcing=manufactured_forcing,
        forcing_is_steady=False,
        boundary_condition_type="fixed",
        boundary_condition_value=0,
    )

    disc_cfg = DiscretizationConfig(
        n_nodes_les=19,
        temporal_refinement=1,
        courant_les=0.1,
        domain_length=2 * np.pi,
    )

    solver = BaseRK2(
        problem=mms_problem,
        disc_cfg=disc_cfg,
        simulation_mode="shakib_two",
        master_path=path,
    )

    solver.run_simulation()

    simulated_solution = solver.solution
    exact_solution = manufactured_solution(x=disc_cfg.mesh_les, t=2 * np.pi)

    plt.plot(disc_cfg.mesh_les, exact_solution)
    plt.plot(disc_cfg.mesh_les, simulated_solution)
    plt.show()
