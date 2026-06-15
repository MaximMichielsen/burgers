"""Current implementation:
u, u_bar and u_prime calculations
stencil input creation

Still to do:
sgs terms computation and assembly
holistic overview and congruency
wall handling
post-simulation validation: recreate dns solution from u_bar and sgs_terms"""

import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.initial_conditions import (
    sin_initial_condition,
)
from problems_and_configurations.problems import Problem, Problems
from solvers.burgers_base import BurgersBase

_VALID_PROJECTION_MODES: frozenset[str] = frozenset({"l2", "L2", "nodal"})


def nodal_project(
    solution_dns: NDArray,
    mesh_dns: NDArray,
    mesh_les: NDArray,
) -> NDArray:
    """Nodal projection of a DNS snapshot onto the LES mesh."""
    u_bar = np.interp(mesh_les, mesh_dns, solution_dns)
    return u_bar


def l2_project(
    solution_dns: NDArray, mesh_dns: NDArray, mesh_les: NDArray
) -> tuple[NDArray, NDArray]:
    """L2-project a DNS snapshot onto the LES P1 mesh.

    Solves M_LES @ u_bar = rhs, where rhs_i = ∫ u_DNS(x) φ_i(x) dx,
    assembled by Gauss quadrature over each LES element.
    Returns (u_bar, uu_bar) to preserve the same interface as box_filter.
    """
    n_les = len(mesh_les)
    n_les_elements = n_les - 1

    mass_matrix = np.zeros((n_les, n_les))
    rhs_u = np.zeros(n_les)
    rhs_uu = np.zeros(n_les)

    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(deg=2)

    for elem_idx in range(n_les_elements):
        x_left = mesh_les[elem_idx]
        x_right = mesh_les[elem_idx + 1]
        h_elem = x_right - x_left
        jacobian = h_elem / 2.0

        for gauss_point, gauss_weight in zip(gauss_points, gauss_weights):
            x_phys = 0.5 * (x_left + x_right) + 0.5 * h_elem * gauss_point
            phi = np.array([0.5 * (1.0 - gauss_point), 0.5 * (1.0 + gauss_point)])
            u_dns_val = float(np.interp(x_phys, mesh_dns, solution_dns))
            scale = gauss_weight * jacobian

            mass_matrix[elem_idx, elem_idx] += scale * phi[0] * phi[0]
            mass_matrix[elem_idx, elem_idx + 1] += scale * phi[0] * phi[1]
            mass_matrix[elem_idx + 1, elem_idx] += scale * phi[1] * phi[0]
            mass_matrix[elem_idx + 1, elem_idx + 1] += scale * phi[1] * phi[1]

            rhs_u[elem_idx] += scale * phi[0] * u_dns_val
            rhs_u[elem_idx + 1] += scale * phi[1] * u_dns_val
            rhs_uu[elem_idx] += scale * phi[0] * u_dns_val**2
            rhs_uu[elem_idx + 1] += scale * phi[1] * u_dns_val**2

    u_bar = np.linalg.solve(mass_matrix, rhs_u)
    uu_bar = np.linalg.solve(mass_matrix, rhs_uu)
    return u_bar, uu_bar


def gradient_basis_functions(element_size: float) -> NDArray:
    """Constant gradient of linear basis on physical element: dN/dx = [-1, 1] / h."""
    return np.array([-1.0, 1.0]) / element_size


class BurgersDataGenerator(BurgersBase):
    """SGSP data generator.

    Use to create DNS data where each requested snapshot also creates LES-projected solution and SGSP training data.
    """

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretisationConfig,
        simulation_mode: str,
        master_path: Path,
        snapshot_factor: int | None = 1,
        projection_mode: str = "nodal",
        warmup_steps: int = 2,
    ) -> None:
        super().__init__(
            problem, disc_cfg, simulation_mode, master_path, snapshot_factor
        )

        if projection_mode not in _VALID_PROJECTION_MODES:
            raise ValueError(
                f"Invalid projection mode. Choose mode for projection, options: {_VALID_PROJECTION_MODES}"
            )
        self._projection_mode = projection_mode

        self.warmup_steps = warmup_steps

        self._disc_cfg = disc_cfg
        self._n_nodes_les = disc_cfg.n_nodes_les
        self._mesh_les = disc_cfg.mesh_les
        self._n_nodes_dns = disc_cfg.n_nodes_dns
        self._mesh_dns = disc_cfg.mesh_dns

        self.nodes_les: NDArray = np.arange(0, self._n_nodes_les)
        self.nodes_dns: NDArray = np.arange(0, self._n_nodes_dns)

        self.elements_les = self.initialize_elements(nodes=self.nodes_les)
        self.elements_dns = self.initialize_elements(nodes=self.nodes_dns)

        self.u_bar_now: NDArray = np.zeros(self._n_nodes_les)
        self.du_bar_dt_now: NDArray = np.zeros_like(self.u_bar_now)
        self.u_prime_now: NDArray = np.zeros(self._n_nodes_dns)
        self.interp_les_to_dns_u: NDArray = np.zeros_like(self.u_prime_now)

        self.projected_forcing = np.zeros_like(self.u_bar_now)

        self.u_bar_history: list[NDArray] = []
        self.du_bar_dt_history: list[NDArray] = []
        self.forcing_history: list[NDArray] = []

        self.assembled_input_stencils: list[NDArray] = []
        self.assembled_sgs_terms: list[NDArray] = []

    def advance_time_step(self) -> None:
        self.resolve_current_forcing()
        self.solution = self.nr_iteration(self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt

        self.u_bar_now, self.interp_les_to_dns_u, self.projected_forcing = (
            self.project_u_to_les()
        )
        self.u_prime_now = self.compute_u_prime(
            interpolated_les_solution=self.interp_les_to_dns_u
        )

        self.u_bar_history.append(self.u_bar_now)
        self.du_bar_dt_now = self.compute_du_bar_dt(
            u_bar_now=self.u_bar_history[-1], u_bar_prev=self.u_bar_history[-2]
        )
        self.du_bar_dt_history.append(self.du_bar_dt_now)

        self.forcing_history.append(self.projected_forcing)

    def run_simulation(self) -> None:
        """Run the full time-marching simulation and write output."""

        # add IC and projections to snapshots
        self.resolve_current_forcing()
        self._extract_snapshot()
        self.u_bar_now, self.interp_les_to_dns_u, self.projected_forcing = (
            self.project_u_to_les()
        )
        self.u_prime_now = self.compute_u_prime(
            interpolated_les_solution=self.interp_les_to_dns_u
        )

        self.u_bar_history.append(self.u_bar_now)

        with self.timer("total_simulation"):
            with tqdm(
                total=self._n_time_steps,
                desc=f"Eating Burgers | {self.throbber(0)}",
                file=sys.stdout,
            ) as pbar:
                for time_step in range(self._n_time_steps):
                    step_start = perf_counter()

                    self.advance_time_step()

                    if (
                        time_step + 1
                    ) in self._snapshot_step_indices and time_step > self.warmup_steps:
                        self._extract_snapshot()
                        input_stencils, sgs_terms = self.create_training_data_at_t()
                        self.assembled_input_stencils.append(input_stencils)
                        self.assembled_sgs_terms.append(sgs_terms)

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

    def create_training_data_at_t(self) -> tuple[list[NDArray], list[NDArray]]:
        input_stencils = []
        sgs_terms = []
        for node in self.nodes_les:
            input_stencil = self.create_input_stencil(node)
            node_sgs_terms = self.compute_sgs_terms(node)

            input_stencils.append(input_stencil)
            sgs_terms.append(node_sgs_terms)

        return input_stencils, sgs_terms

    def create_input_stencil(self, node_idx: int):
        # if i = 0 or [-1] set wall 'nodes' to 0.
        """Build the 20-feature FS2 input vector for element element_idx at time level n.

        Stencil: [ū^{n,n-1,n-2}_{i-2:i+1}, (∂ū/∂t)^n_{i-2:i+1}, f^n_{i-2:i+1}].
        Returns None if the stencil falls outside the domain.
        """
        if len(self.u_bar_history) < self.warmup_steps:
            return None

        stencil_nodes = np.array([node_idx - 2, node_idx - 1, node_idx, node_idx + 1])

        if stencil_nodes[0] < 0 or stencil_nodes[-1] >= self._n_nodes_les:
            # TODO add wall handling
            return None

        return np.concatenate(
            [
                self.u_bar_history[-1][stencil_nodes],
                self.u_bar_history[-2][stencil_nodes],
                self.u_bar_history[-3][stencil_nodes],
                self.du_bar_dt_history[-1][stencil_nodes],
                self.forcing_history[-1][stencil_nodes],
            ]
        )

    def compute_sgs_terms(self, element: int):
        pass

    def compute_du_bar_dt(self, u_bar_now, u_bar_prev) -> NDArray:
        return (u_bar_now - u_bar_prev) / self._disc_cfg.dt_les

    def project_u_to_les(self) -> tuple[NDArray, NDArray, NDArray]:
        """Project DNS solution to LES grid and interpolate it back to the DNS grid."""
        u_bar_now = nodal_project(
            self.solution, mesh_les=self._mesh_les, mesh_dns=self._mesh_dns
        )
        u_les_to_dns = np.array(np.interp(self._mesh_dns, self._mesh_les, u_bar_now))
        projected_forcing = (
            nodal_project(
                self.forcing_current, mesh_les=self._mesh_les, mesh_dns=self._mesh_dns
            )
            if self.forcing_current is not None
            else np.zeros_like(self.u_bar_now)
        )
        return u_bar_now, u_les_to_dns, projected_forcing

    def compute_u_prime(self, interpolated_les_solution: NDArray) -> NDArray:
        """Subtracts interpolated to-LES projected solution from the DNS solution."""
        u_prime = self.solution - interpolated_les_solution
        return u_prime

    def plotting_interpolation_and_projection(self):

        plt.plot(self._mesh_dns, self.solution, label="dns", color="gray", alpha=0.8)
        plt.plot(
            self._mesh_les,
            self.u_bar_now,
            label="u_bar",
            color="royalblue",
            marker="x",
            linestyle="--",
        )
        plt.plot(self._mesh_dns, self.u_prime_now, label="u_prime", color="tab:orange")
        plt.grid(True)

        plt.legend()
        plt.show()


if __name__ == "__main__":
    disc_cfg = DiscretisationConfig(
        n_nodes_les=9,
        temporal_refinement=1,
        courant_les=0.1,
        domain_length=2 * np.pi,
    )

    path = Path(r"C:\Users\poopy\PycharmProjects\burgers\test_suite")
    problem = Problems.pipeline_test
    problem = replace(
        problem, initial_condition=sin_initial_condition, domain_length=2 * np.pi
    )
    solver = BurgersDataGenerator(
        problem,
        disc_cfg=disc_cfg,
        simulation_mode="dns",
        master_path=path,
    )

    solver.run_simulation()

    solver.plotting_interpolation_and_projection()
