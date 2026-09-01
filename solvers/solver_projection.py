import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase


@dataclass
class ElementSGSTerms:
    """SGS closure terms for one element, in two forms.

    scatter: shape (2, 5) — per local node (left=0, right=1), columns are
        (cross, reynolds, temporal_l, temporal_r, viscous). Used by the solver.
    label: shape (5,) — element-level training target using right-node gradient
        weight for gradient terms. Used as SGSP predictor output.
    """

    scatter: NDArray  # (2, 5)
    label: NDArray  # (5,)


def nodal_project(
    solution_dns: NDArray,
    mesh_dns: NDArray,
    mesh_les: NDArray,
) -> NDArray:
    """Nodal projection of a DNS snapshot onto the LES mesh."""
    return np.interp(mesh_les, mesh_dns, solution_dns)


class BurgersDataGenerator(SolverBase):
    """DNS runner that simultaneously assembles SGSP training data.

    Generates DNS snapshots and computes per-element closure terms stored as
    ElementSGSTerms, which carries both the solver scatter form (2, 5) and the
    SGSP training label (5,). Replaces the old projection + training_data_assembly pipeline.
    """

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretizationConfig,
        simulation_mode: str,
        master_path: Path,
        dns_save_path: Path | None = None,
        projection_save_path: Path | None = None,
        snapshot_factor: int = 1,
        projection_mode: str = "nodal",
        t_start: float = 0.0,
        append_mode: bool = False,
    ) -> None:
        super().__init__(
            problem=problem,
            disc_config=disc_cfg,
            simulation_mode=simulation_mode,
            master_path=master_path,
            snapshot_factor=snapshot_factor,
            t_start=t_start,
        )

        self.dns_save_path = (
            dns_save_path
            if dns_save_path is not None
            else master_path / "solver_data" / "DNS"
        )

        self.projection_save_path = (
            projection_save_path
            if projection_save_path is not None
            else master_path / "solver_data" / "projection"
        )

        self._projection_mode = projection_mode

        self._disc_cfg = disc_cfg
        self._n_nodes_les = disc_cfg.n_nodes_les
        self._mesh_les = disc_cfg.mesh_les
        self._n_nodes_dns = disc_cfg.n_nodes_dns
        self._mesh_dns = disc_cfg.mesh_dns
        self.nodes_les: NDArray = np.arange(0, self._n_nodes_les)

        self.u_bar_now: NDArray = np.zeros(self._n_nodes_les)
        self.du_bar_dt_now: NDArray = np.zeros_like(self.u_bar_now)
        self.u_prime_now: NDArray = np.zeros(self._n_nodes_dns)
        self.interp_les_to_dns_u: NDArray = np.zeros_like(self.u_prime_now)
        self.projected_forcing: NDArray = np.zeros_like(self.u_bar_now)

        self.solution_history: list[NDArray] = []
        self.u_bar_history: list[NDArray] = []
        self.du_bar_dt_history: list[NDArray] = []
        self.u_prime_history: list[NDArray] = []
        self.forcing_history: list[NDArray] = []

        self.append_mode = append_mode

    # ------------------------------------------------------------------
    # Time-stepping
    # ------------------------------------------------------------------

    def advance_time_step(self) -> None:
        """Advance the simulation by one time step."""
        self.resolve_current_forcing()

        new_solution = self.nr_iteration(self.solution, self.solution_previous)
        self.solution_previous = self.solution
        self.solution = new_solution

        self.simulation_time_elapsed += self.dt
        self.u_bar_now, self.interp_les_to_dns_u, self.projected_forcing = (
            self.project_u_to_les()
        )
        self.u_prime_now = self.compute_u_prime(
            interpolated_les_solution=self.interp_les_to_dns_u
        )

        self.u_bar_history.append(self.u_bar_now)
        self.du_bar_dt_now = self.compute_du_bar_dt(
            u_bar_now=self.u_bar_history[-1],
            u_bar_prev=self.u_bar_history[-2],
        )

        self.solution_history.append(self.solution)
        self.du_bar_dt_history.append(self.du_bar_dt_now)
        self.u_prime_history.append(self.u_prime_now)
        self.forcing_history.append(self.projected_forcing)

    def run_simulation(self) -> None:
        """Run the full time-marching simulation and write output."""
        # IC: init histories and save snapshot, but skip training data (stencil incomplete)
        self.resolve_current_forcing()
        self._extract_snapshot()
        self.u_bar_now, self.interp_les_to_dns_u, self.projected_forcing = (
            self.project_u_to_les()
        )
        self.u_prime_now = self.compute_u_prime(
            interpolated_les_solution=self.interp_les_to_dns_u
        )
        self.u_prime_history.append(self.u_prime_now)
        self.solution_history.append(self.solution)
        self.u_bar_history.append(self.u_bar_now)
        self.forcing_history.append(self.projected_forcing)

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
            self.save_all_data()

    def run_projection_only(self) -> None:
        """Recompute projection and training data from existing DNS snapshots on disk.

        Used after DNS extension to regenerate training data over the full timespan
        without re-running the solver.
        """
        csv_files = sorted(self.dns_save_path.glob("sol_t*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No DNS snapshots found in {self.dns_save_path}")

        # reset histories
        self.u_bar_history.clear()
        self.du_bar_dt_history.clear()
        self.u_prime_history.clear()
        self.forcing_history.clear()

        for csv_path in csv_files:
            velocity_values = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=2)
            self.solution = velocity_values

            self.resolve_current_forcing()
            self.u_bar_now, self.interp_les_to_dns_u, self.projected_forcing = (
                self.project_u_to_les()
            )
            self.u_prime_now = self.compute_u_prime(
                interpolated_les_solution=self.interp_les_to_dns_u
            )

            self.u_bar_history.append(self.u_bar_now)
            self.u_prime_history.append(self.u_prime_now)
            self.forcing_history.append(self.projected_forcing)

            if len(self.u_bar_history) >= 2:
                self.du_bar_dt_now = self.compute_du_bar_dt(
                    u_bar_now=self.u_bar_history[-1],
                    u_bar_prev=self.u_bar_history[-2],
                )
                self.du_bar_dt_history.append(self.du_bar_dt_now)

            self.simulation_time_elapsed += self.dt

        self.projection_save_path.mkdir(parents=True, exist_ok=True)
        self.write_projected_solution_to_csv(save_path=self.projection_save_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compute_du_bar_dt(self, u_bar_now: NDArray, u_bar_prev: NDArray) -> NDArray:
        """First-order backward difference du_bar/dt."""
        return (u_bar_now - u_bar_prev) / self._disc_cfg.dt_les

    def project_u_to_les(self) -> tuple[NDArray, NDArray, NDArray]:
        """Project DNS solution to LES grid and interpolate back to DNS grid."""
        u_bar_now = nodal_project(
            self.solution, mesh_les=self._mesh_les, mesh_dns=self._mesh_dns
        )
        u_les_to_dns = np.array(np.interp(self._mesh_dns, self._mesh_les, u_bar_now))
        projected_forcing = (
            nodal_project(
                self.forcing_current,
                mesh_les=self._mesh_les,
                mesh_dns=self._mesh_dns,
            )
            if self.forcing_current is not None
            else np.zeros_like(self.u_bar_now)
        )
        return u_bar_now, u_les_to_dns, projected_forcing

    def compute_u_prime(self, interpolated_les_solution: NDArray) -> NDArray:
        """Compute fine-scale component u' = u_DNS - u_bar_interpolated."""
        return self.solution - interpolated_les_solution

    # ------------------------------------------------------------------
    # Data saving
    # ------------------------------------------------------------------

    def save_all_data(self) -> None:
        """Save DNS solutions and optionally SGSP training data to disk."""
        self.dns_save_path.mkdir(parents=True, exist_ok=True)
        self.write_solution_to_csv(save_path=self.dns_save_path)
        if self.projection_save_path is not None:
            self.projection_save_path.mkdir(parents=True, exist_ok=True)
            self.write_projected_solution_to_csv(save_path=self.projection_save_path)

    def write_projected_solution_to_csv(self, save_path: Path | None = None) -> None:
        """Write extracted solution snapshots to CSV files."""
        solutions = self.u_bar_history
        if self.requested_snapshots is None:
            return

        times = self.requested_snapshots[: len(solutions)]

        for solution, time_value in zip(solutions, times):
            master_path = save_path if save_path is not None else self.master_path
            filepath = master_path / f"sol_t{time_value:.6f}.csv"
            with open(filepath, mode="w", newline="") as file_handle:
                writer = csv.writer(file_handle)
                writer.writerow(["node_index", "x_coordinate", "velocity"])
                for i in range(len(solution)):
                    writer.writerow(
                        [
                            self.nodes_les[i],
                            self._mesh_les[i],
                            solution[i],
                        ]
                    )

        print(f"wrote {len(solutions)} snapshots at {self.master_path}")

    # ------------------------------------------------------------------
    # Debug / diagnostics
    # ------------------------------------------------------------------

    def plotting_interpolation_and_projection(self) -> None:
        """Plot DNS, u_bar, and u' for the current state."""
        plt.plot(self._mesh_dns, self.solution, label="dns", color="gray", alpha=0.8)
        plt.plot(
            self._mesh_les,
            self.u_bar_now,
            label="u_bar",
            color="royalblue",
            marker="x",
            linestyle="--",
        )
        plt.plot(
            self._mesh_dns,
            self.u_prime_now,
            label="u_prime",
            color="tab:orange",
        )
        plt.grid(True)
        plt.legend()
        plt.show()
