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
from problems_and_configurations.problems import Problem, Problems
from solvers.burgers_base import BurgersBase


WARMUP_STEPS: int = 3
PROJECTION_GAUSS_POINTS: int = 6


def nodal_project(
    solution_dns: NDArray,
    mesh_dns: NDArray,
    mesh_les: NDArray,
) -> NDArray:
    """Nodal projection of a DNS snapshot onto the LES mesh."""
    u_bar = np.interp(mesh_les, mesh_dns, solution_dns)
    return u_bar


def gradient_basis_functions(element_size: float) -> NDArray:
    """Constant gradient of linear basis on physical element: dN/dx = [-1, 1] / h."""
    return np.array([-1.0, 1.0]) / element_size


class BurgersDataGenerator(BurgersBase):
    """SGSP data generator.

    Use to create DNS data where each requested snapshot also creates LES-projected solution and SGSP training data.
    Overrides BurgersBase run_simulation and advance_time_step.
    Creates SGSP input stencils and closure terms.
    """

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretisationConfig,
        simulation_mode: str,
        master_path: Path,
        snapshot_factor: int | None = 1,
        projection_mode: str = "nodal",
        warmup_steps: int = WARMUP_STEPS,
    ) -> None:
        super().__init__(
            problem, disc_cfg, simulation_mode, master_path, snapshot_factor
        )

        self._projection_mode = projection_mode
        self.warmup_steps = warmup_steps

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
        self.projected_forcing = np.zeros_like(self.u_bar_now)

        self.solution_history: list[NDArray] = []
        self.u_bar_history: list[NDArray] = []
        self.du_bar_dt_history: list[NDArray] = []
        self.u_prime_history: list[NDArray] = []
        self.forcing_history: list[NDArray] = []
        self.assembled_input_stencils: list[list[NDArray | None]] = []
        self.assembled_sgs_terms: list[list[NDArray]] = []

    def advance_time_step(self) -> None:
        """Advances the simulation by one time-step."""
        self.resolve_current_forcing()
        self.solution = self.nr_iteration(self.solution)
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

        self.solution_history.append(self.solution)
        self.du_bar_dt_history.append(self.du_bar_dt_now)
        self.u_prime_history.append(self.u_prime_now)
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
        self.u_prime_history.append(self.u_prime_now)
        input_stencils, sgs_terms = self.create_snapshot_training_data()

        self.solution_history.append(self.solution)
        self.u_bar_history.append(self.u_bar_now)
        self.assembled_input_stencils.append(input_stencils)
        self.assembled_sgs_terms.append(sgs_terms)

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
                        if time_step >= self.warmup_steps:
                            self._extract_snapshot()
                            input_stencils, sgs_terms = (
                                self.create_snapshot_training_data()
                            )
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

    def create_snapshot_training_data(
        self,
    ) -> tuple[list[NDArray | None], list[NDArray]]:
        """Compute SGS term contributions per element, returning (left, right) node pairs."""
        input_stencils: list[NDArray | None] = []
        sgs_terms: list[NDArray] = []

        for element_left_node in self.nodes_les[:-1]:
            input_stencils.append(self.create_input_stencil(node_idx=element_left_node))
            sgs_terms.append(self.compute_element_sgs_terms(element_left_node))

        return input_stencils, sgs_terms

    def create_input_stencil(self, node_idx: int) -> NDArray | None:
        """Build the 20-feature FS2 input vector for node node_idx at time level n.

        Stencil: [ū^{n,n-1,n-2}_{i-2:i+1}, (∂ū/∂t)^n_{i-2:i+1}, f^n_{i-2:i+1}].
        Out-of-domain nodes are zero-padded (wall condition).
        """
        if len(self.u_bar_history) < self.warmup_steps:
            return None

        stencil_nodes = np.array([node_idx - 2, node_idx - 1, node_idx, node_idx + 1])

        def _gather_with_wall_pad(field: NDArray) -> NDArray:
            """Extract stencil values, padding out-of-bounds indices with zero."""
            values = np.zeros(4)
            for local_idx, global_idx in enumerate(stencil_nodes):
                if 0 <= global_idx < self._n_nodes_les:
                    values[local_idx] = field[global_idx]
            return values

        return np.concatenate(
            [
                _gather_with_wall_pad(self.u_bar_history[-1]),
                _gather_with_wall_pad(self.u_bar_history[-2]),
                _gather_with_wall_pad(self.u_bar_history[-3]),
                _gather_with_wall_pad(self.du_bar_dt_history[-1]),
                _gather_with_wall_pad(self.forcing_history[-1]),
            ]
        )

    # TODO: Implement temporal closure term

    def compute_element_sgs_terms(self, element_left_node: int) -> NDArray:
        """Integrate SGS terms over element [i, i+1]; returns shape (2, 5) — (left_node, right_node) x (cross, reynolds, temp_l, temp_r, viscous)."""
        u_bar_now = self.interp_les_to_dns_u
        u_prime_now = self.u_prime_history[-1].copy()
        u_prime_now[0] = 0.0
        u_prime_now[-1] = 0.0

        mesh_dns = self._mesh_dns
        du_prime_dx_dns = np.gradient(u_prime_now, self._disc_cfg.h_dns)

        x_left = float(self._mesh_les[element_left_node])
        x_right = float(self._mesh_les[element_left_node + 1])

        gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(
            deg=PROJECTION_GAUSS_POINTS
        )
        grad_basis = gradient_basis_functions(self._disc_cfg.h_les)  # [-1/h, +1/h]
        jacobian = self._disc_cfg.h_les / 2.0

        # accumulate for left (index 0) and right (index 1) nodes
        cross, reynolds, viscous = np.zeros(2), np.zeros(2), np.zeros(2)

        for gauss_pt, gauss_wt in zip(gauss_pts, gauss_wts):
            x_phys = 0.5 * (x_left + x_right) + 0.5 * self._disc_cfg.h_les * gauss_pt
            scale = gauss_wt * jacobian

            u_bar_gp = float(np.interp(x_phys, mesh_dns, u_bar_now))
            u_prime_gp = float(np.interp(x_phys, mesh_dns, u_prime_now))
            du_prime_dx_gp = float(np.interp(x_phys, mesh_dns, du_prime_dx_dns))

            for node_local in range(2):  # 0=left, 1=right
                w_x = grad_basis[node_local]
                cross[node_local] += scale * w_x * u_bar_gp * u_prime_gp
                reynolds[node_local] += scale * w_x * 0.5 * u_prime_gp**2
                viscous[node_local] += scale * w_x * du_prime_dx_gp

        # shape (2, 5): [cross, reynolds, temp_left, temp_right, viscous] per node
        return np.stack(
            [
                cross,
                reynolds,
                np.zeros(2),
                np.zeros(2),
                viscous,
            ],
            axis=1,
        )

    def compute_du_bar_dt(self, u_bar_now, u_bar_prev) -> NDArray:
        """First order discretization of du_bar / dt."""
        return (u_bar_now - u_bar_prev) / self._disc_cfg.dt_les

    def project_u_to_les(self) -> tuple[NDArray, NDArray, NDArray]:
        """Project (nodal/H1_0) DNS solution to LES grid and interpolate it back to the DNS grid."""
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
        return self.solution - interpolated_les_solution

    def plotting_interpolation_and_projection(self) -> None:
        """Plot DNS solution along with u_bar and u_prime."""
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


class ProjDNSReconstructor(BurgersBase):
    def __init__(
        self,
        dns_solutions: list[NDArray],
        u_bar_solutions: list[NDArray],
        closure_terms: list[list[NDArray]],
        problem: Problem,
        disc_cfg: DiscretisationConfig,
        master_path: Path,
        simulation_mode: str = "no_model",
        snapshot_factor: int | None = 1,
        use_closure_terms: bool = True,
    ) -> None:
        super().__init__(
            problem, disc_cfg, simulation_mode, master_path, snapshot_factor
        )

        self.disc_cfg = disc_cfg
        self.nodes_les = np.arange(0, disc_cfg.n_nodes_les)

        self.dns_solutions = dns_solutions
        self.u_bar_solutions = u_bar_solutions
        self.closure_terms = closure_terms
        self.use_closure_terms = use_closure_terms

        self.time_steps_stepped: int = 0

    def recreate_solution(self):
        """Run the full time-marching simulation and write output."""

        # add IC to snapshots
        self.resolve_current_forcing()
        self._extract_snapshot()

        for time_step in range(self._n_time_steps):
            self.advance_time_step()
            if (time_step + 1) in self._snapshot_step_indices:
                self._extract_snapshot()

        self.write_config_to_json()
        self.write_solution_to_csv()

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """Newton–Raphson iteration; returns U^{n+1}."""
        solution_n = solution.copy()
        solution_k = solution.copy()
        residual_history_loop: list = []
        update_history_loop: list = []

        self.max_iterations = 1

        for _ in range(self.max_iterations):
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

            global_residual, global_jacobian = self.global_assembly(
                elemental_residuals, elemental_jacobians
            )
            if self.use_closure_terms:
                global_residual = self.add_closure_terms_to_residual(
                    global_residual, self.time_steps_stepped
                )

            global_residual, global_jacobian = self._apply_boundary_conditions(
                global_residual, global_jacobian, solution_k
            )
            residual_history_loop.append(np.linalg.norm(global_residual))

            delta_u = np.linalg.solve(global_jacobian, -global_residual)
            if self.boundary_condition_type == "periodic":
                delta_u_full = np.zeros_like(solution_k)
                delta_u_full[:-1] = delta_u
                delta_u_full[-1] = delta_u[0]
                delta_u = delta_u_full

            update_history_loop.append(np.linalg.norm(delta_u))

            solution_k += delta_u

            if self.is_update_converged(delta_u) or self.is_residual_converged(
                global_residual
            ):
                break

        self.residual_history.append(residual_history_loop)
        self.update_history.append(update_history_loop)
        return solution_k

    def advance_time_step(self) -> None:
        """Advance the solution by one time step: U^{n+1} ← U^n."""
        self.resolve_current_forcing()
        self.solution = self.nr_iteration(self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt
        self.time_steps_stepped += 1

    def add_closure_terms_to_residual(
        self, residual: NDArray, time_step: int
    ) -> NDArray:
        """Scatter element SGS contributions to both left and right nodes."""
        snapshot_idx: int = min(time_step, len(self.closure_terms) - 1)
        for element_idx, element_left_node in enumerate(self.nodes_les[:-1]):
            element_terms: NDArray = self.closure_terms[snapshot_idx][element_idx]
            for local_node, global_node in enumerate(
                [element_left_node, element_left_node + 1]
            ):
                if global_node in (0, self.nodes_les[-1]):
                    continue
                cross_term: float = element_terms[local_node, 0]
                reynolds_term: float = element_terms[local_node, 1]
                viscous_term: float = element_terms[local_node, -1]
                correction: float = (
                    cross_term + reynolds_term - self.viscosity * viscous_term
                )
                residual[global_node] -= correction
        return residual

    def plot_solution_comparison(
        self,
        snapshot_idx: int,
        dns_solutions: list[NDArray],
        u_bar_solutions: list[NDArray],
        reconstructed_no_model: NDArray | None = None,
    ) -> None:
        """Compare DNS, LES projection, reconstructed (with/without closure) at a snapshot."""
        dns_sol: NDArray = dns_solutions[snapshot_idx]
        u_bar_sol: NDArray = u_bar_solutions[snapshot_idx]
        reconstructed_sol: NDArray = self.snapshots[snapshot_idx][0]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.disc_cfg.mesh_dns, dns_sol, label="DNS", color="gray", alpha=0.8)
        ax.plot(
            self.disc_cfg.mesh_les,
            u_bar_sol,
            label="ū (projection)",
            color="royalblue",
            marker="x",
            linestyle="--",
        )
        ax.plot(
            self.disc_cfg.mesh_les,
            reconstructed_sol,
            label="reconstructed + closure",
            color="tab:orange",
            marker="o",
            linestyle="--",
        )
        if reconstructed_no_model is not None:
            ax.plot(
                self.disc_cfg.mesh_les,
                reconstructed_no_model[snapshot_idx][0],
                label="reconstructed (no closure)",
                color="tab:green",
                marker="s",
                linestyle=":",
            )
        ax.set_title(f"Snapshot {snapshot_idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    _disc_cfg = DiscretisationConfig(
        n_nodes_les=9,
        temporal_refinement=1,
        courant_les=0.1,
        domain_length=1,
    )
    _path = Path(r"C:\Users\poopy\PycharmProjects\burgers\test_suite")
    _problem = replace(Problems.raj_one, domain_timespan=3.0)

    _solver = BurgersDataGenerator(
        _problem,
        disc_cfg=_disc_cfg,
        simulation_mode="dns",
        master_path=_path,
        warmup_steps=0,
    )
    _solver.run_simulation()

    _recreator_no_model = ProjDNSReconstructor(
        dns_solutions=_solver.solution_history,
        u_bar_solutions=_solver.u_bar_history,
        closure_terms=_solver.assembled_sgs_terms,
        problem=_problem,
        disc_cfg=_disc_cfg,
        simulation_mode="les",
        master_path=_path,
        use_closure_terms=False,
    )
    _recreator_no_model.recreate_solution()

    _recreator = ProjDNSReconstructor(
        dns_solutions=_solver.solution_history,
        u_bar_solutions=_solver.u_bar_history,
        closure_terms=_solver.assembled_sgs_terms,
        problem=_problem,
        disc_cfg=_disc_cfg,
        simulation_mode="no_model",
        master_path=_path,
        use_closure_terms=True,
    )
    _recreator.recreate_solution()
    _recreator.plot_solution_comparison(
        snapshot_idx=len(_solver.solution_history) - 10,
        dns_solutions=_solver.solution_history,
        u_bar_solutions=_solver.u_bar_history,
        reconstructed_no_model=_recreator_no_model.snapshots,
    )
