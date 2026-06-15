from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problem
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
        projection_mode: str = "nodal"
    ) -> None:
        super().__init__(
            problem, disc_cfg, simulation_mode, master_path, snapshot_factor
        )

        if projection_mode not in _VALID_PROJECTION_MODES:
            raise ValueError(
                f"Invalid projection mode. Choose mode for projection, options: {_VALID_PROJECTION_MODES}"
            )
        self._projection_mode = projection_mode

        self.n_nodes_dns = disc_cfg.n_nodes_dns
        self.n_nodes_les = disc_cfg.n_nodes_les

        self.projected_snapshots: list[NDArray] = []
        self.input_stencils: list[NDArray] = []
        self.closure_terms: list[NDArray] = []

        self.u_bar_history: list = []
        self.du_bar_dt_history: list = []
        self.forcing_history: list = []

    def generate_data(self):
        """Generate DNS, projected-to-LES DNS data, and SGSP training data."""
        self._extract_snapshot()
        self.projected_snapshots.append(nodal_project(self.solution))
        input_stencil = self.create_sgsp_input_stencil(self.solution)
        self.input_stencils.append(input_stencil) if input_stencil is not None else None
        self.closure_terms.append(self.compute_sgsp_closure_term(self.solution))
        self.forcing_history.append(self.forcing_current)

        super().run_simulation()

    def advance_time_step(self) -> None:
        """Advances the solution by one timestep.

        Per snapshot: Projects the DNS solution to the LES grid and appends to self.projected_snapshots
                      Creates SGSP input stencils and appends to self.input_stencils
                      Computes LES closure terms and appends to self.closure_terms
        """
        self.resolve_current_forcing()
        self.solution = self.nr_iteration(self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt

        self.projected_snapshots.append(nodal_project(self.solution))
        input_stencil = self.create_sgsp_input_stencil(self.solution)
        self.input_stencils.append(input_stencil) if input_stencil is not None else None
        self.closure_terms.append(self.compute_sgsp_closure_term(self.solution))

        #TODO: append u_bar and du_dt_bar in create_sgsp_input_stencil!

    def create_sgsp_input_stencil(self, element_idx: int) -> NDArray | None:
        """Build the 20-feature FS2 input vector for element element_idx at time level n.

        Stencil: [ū^{n,n-1,n-2}_{i-2:i+1}, (∂ū/∂t)^n_{i-2:i+1}, f^n_{i-2:i+1}].
        Returns None if the stencil falls outside the domain.
        """
        if len(self.u_bar_history) < 3:
            return None

        stencil_nodes = np.array(
            [element_idx - 2, element_idx - 1, element_idx, element_idx + 1]
        )
        #TODO: check how normally wall handling is implemented!
        if stencil_nodes[0] < 0 or stencil_nodes[-1] >= self.n_nodes_les:
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

    def compute_sgsp_closure_term(self) -> NDArray:
        pass
