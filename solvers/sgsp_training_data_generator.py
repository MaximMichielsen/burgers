"""Generate DNS and SGSP data.

BurgersDataGenerator assembles, normalizes, and saves (X, y) CSV data directly during the DNS run.

Output stencil follows Rajampeta (2022) / Research Proposal formulation.
Per element e, the 5 interaction terms are:

    [0] (w_x, u_bar*u')_e     cross term
    [1] (w_x, u'^2/2)_e       Reynolds term
    [2] (w_l, u'_t)_e         temporal, left weight
    [3] (w_r, u'_t)_e         temporal, right weight
    [4] (w_x, u'_x)_e         viscous SGS term

Input stencil (Rajampeta FS2):
    [u_bar^{n,n-1,n-2}_{i-2:i+1}, (du_bar/dt)^n_{i-2:i+1}, f^n_{i-2:i+1}] -> 20 features

References: Rajampeta (2022) Sec. 4.3 / Table 4.4, Research Proposal Sec. 2.3.1,
            Robijns (2019) Sec. 3.2.1.
"""

import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problem, Problems
from solvers.burgers_base import BurgersBase


WARMUP_STEPS: int = 3
PROJECTION_GAUSS_POINTS: int = 6

# CSV column headers
_INPUT_COLS: list[str] = [f"x_{feat_idx:02d}" for feat_idx in range(20)]
_OUTPUT_COLS: list[str] = ["cross", "reynolds", "temporal_l", "temporal_r", "viscous"]
_TRAINING_CSV_HEADER: list[str] = _INPUT_COLS + _OUTPUT_COLS


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


def _basis_functions(ksi: float) -> NDArray:
    """Linear basis functions on the reference element [-1, 1]."""
    return np.array([0.5 * (1.0 - ksi), 0.5 * (1.0 + ksi)])


def gradient_basis_functions(element_size: float) -> NDArray:
    """Constant gradient of linear basis on physical element: dN/dx = [-1, 1] / h."""
    return np.array([-1.0, 1.0]) / element_size


def build_input_stencil_wall_padded(
    u_bar_history: list[NDArray],
    du_bar_dt_history: list[NDArray],
    forcing_history: list[NDArray],
    node_idx: int,
    n_nodes: int,
) -> NDArray | None:
    """Build the 20-feature FS2 input vector for element left-node node_idx at time n.

    Stencil: [u_bar^{n,n-1,n-2}_{i-2:i+1}, (du_bar/dt)^n_{i-2:i+1}, f^n_{i-2:i+1}].
    Out-of-domain nodes are zero-padded (wall BC). Returns None if fewer than 3
    time levels are available. Used by both BurgersDataGenerator and BurgersSGSP
    to ensure training and inference use identical stencil construction.
    """
    if len(u_bar_history) < 3 or len(du_bar_dt_history) < 1:
        return None

    stencil_nodes = np.array([node_idx - 2, node_idx - 1, node_idx, node_idx + 1])

    def _gather(field: NDArray) -> NDArray:
        values = np.zeros(4)
        for local_idx, global_idx in enumerate(stencil_nodes):
            if 0 <= global_idx < n_nodes:
                values[local_idx] = field[global_idx]
        return values

    return np.concatenate(
        [
            _gather(u_bar_history[-1]),
            _gather(u_bar_history[-2]),
            _gather(u_bar_history[-3]),
            _gather(du_bar_dt_history[-1]),
            _gather(forcing_history[-1]),
        ]
    )


class BurgersDataGenerator(BurgersBase):
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
        sgsp_training_data_path: Path | None = None,
        projection_save_path: Path | None = None,
        snapshot_factor: int = 1,
        projection_mode: str = "nodal",
        warmup_steps: int = WARMUP_STEPS,
        t_start: float = 0.0,
        append_mode: bool = False,
    ) -> None:
        super().__init__(
            problem, disc_cfg, simulation_mode, master_path, snapshot_factor, t_start
        )

        self.dns_save_path = (
            dns_save_path
            if dns_save_path is not None
            else master_path / "solver_data" / "DNS"
        )
        self.sgs_save_path = (
            sgsp_training_data_path
            if sgsp_training_data_path is not None
            else master_path / "training_data" / "sgsp"
        )
        self.projection_save_path = (
            projection_save_path
            if projection_save_path is not None
            else master_path / "solver_data" / "projection"
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
        self.projected_forcing: NDArray = np.zeros_like(self.u_bar_now)

        self.solution_history: list[NDArray] = []
        self.u_bar_history: list[NDArray] = []
        self.du_bar_dt_history: list[NDArray] = []
        self.u_prime_history: list[NDArray] = []
        self.forcing_history: list[NDArray] = []

        # per-snapshot list of per-element ElementSGSTerms
        self.assembled_sgs_terms: list[list[ElementSGSTerms]] = []
        # SGSP input stencils: list[list[NDArray | None]] where inner NDArray is (20,)
        self.assembled_input_stencils: list[list[NDArray | None]] = []

        self.append_mode = append_mode

    # ------------------------------------------------------------------
    # Time-stepping
    # ------------------------------------------------------------------

    def advance_time_step(self) -> None:
        """Advance the simulation by one time step."""
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

                        if time_step >= self.warmup_steps:
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
        self.assembled_input_stencils.clear()
        self.assembled_sgs_terms.clear()

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

            input_stencils, sgs_terms = self.create_snapshot_training_data()
            self.assembled_input_stencils.append(input_stencils)
            self.assembled_sgs_terms.append(sgs_terms)

            self.simulation_time_elapsed += self.dt

        self.sgs_save_path.mkdir(parents=True, exist_ok=True)
        self.projection_save_path.mkdir(parents=True, exist_ok=True)
        self.write_projected_solution_to_csv(save_path=self.projection_save_path)
        self.save_sgsp_training_csv(append_mode=False)  # full recompute, no append

    # ------------------------------------------------------------------
    # Snapshot data assembly
    # ------------------------------------------------------------------

    def create_snapshot_training_data(
        self,
    ) -> tuple[list[NDArray | None], list[ElementSGSTerms]]:
        """Compute input stencils and closure terms for all elements at the current snapshot."""
        input_stencils: list[NDArray | None] = []
        sgs_terms: list[ElementSGSTerms] = []

        for element_left_node in self.nodes_les[:-1]:
            input_stencils.append(self.create_input_stencil(node_idx=element_left_node))
            sgs_terms.append(self.compute_element_closure_terms(element_left_node))

        return input_stencils, sgs_terms

    def create_input_stencil(self, node_idx: int) -> NDArray | None:
        """Build the 20-feature FS2 input vector for element left-node node_idx."""
        return build_input_stencil_wall_padded(
            u_bar_history=self.u_bar_history,
            du_bar_dt_history=self.du_bar_dt_history,
            forcing_history=self.forcing_history,
            node_idx=node_idx,
            n_nodes=self._n_nodes_les,
        )

    def compute_element_closure_terms(self, element_left_node: int) -> ElementSGSTerms:
        """Integrate SGS terms over element [i, i+1].

        Returns an ElementSGSTerms with:
            scatter: (2, 5) per-node contributions for solver residual scatter.
            label: (5,) element-level target for SGSP predictor training,
                using right-node gradient weight (w_x = +1/h) for gradient terms,
                consistent with Rajampeta (2022) Table 4.4.
        """
        u_bar_interp = self.interp_les_to_dns_u
        u_prime_now = self.u_prime_history[-1].copy()
        u_prime_now[0] = 0.0
        u_prime_now[-1] = 0.0

        mesh_dns = self._mesh_dns
        du_prime_dx_dns = np.gradient(u_prime_now, self._disc_cfg.h_dns)

        # temporal derivative of u' (first-order backward, zero at IC)
        if len(self.u_prime_history) >= 2:
            u_prime_prev = self.u_prime_history[-2].copy()
            u_prime_prev[0] = 0.0
            u_prime_prev[-1] = 0.0
            du_prime_dt_dns = (u_prime_now - u_prime_prev) / self._disc_cfg.dt_les
        else:
            du_prime_dt_dns = np.zeros_like(u_prime_now)

        x_left = float(self._mesh_les[element_left_node])
        x_right = float(self._mesh_les[element_left_node + 1])

        gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(
            deg=PROJECTION_GAUSS_POINTS
        )
        grad_basis = gradient_basis_functions(self._disc_cfg.h_les)
        jacobian = self._disc_cfg.h_les / 2.0

        # per-node accumulators for solver scatter: shape (2,) each
        cross_scatter = np.zeros(2)
        reynolds_scatter = np.zeros(2)
        temporal_l_scatter = np.zeros(2)
        temporal_r_scatter = np.zeros(2)
        viscous_scatter = np.zeros(2)

        for gauss_pt, gauss_wt in zip(gauss_pts, gauss_wts):
            x_phys = 0.5 * (x_left + x_right) + 0.5 * self._disc_cfg.h_les * gauss_pt
            basis_vals = _basis_functions(gauss_pt)
            scale = gauss_wt * jacobian

            u_bar_gp = float(np.interp(x_phys, mesh_dns, u_bar_interp))
            u_prime_gp = float(np.interp(x_phys, mesh_dns, u_prime_now))
            du_prime_dx_gp = float(np.interp(x_phys, mesh_dns, du_prime_dx_dns))
            du_prime_dt_gp = float(np.interp(x_phys, mesh_dns, du_prime_dt_dns))

            # scatter terms (per-node)
            for node_local in range(2):
                w_x = grad_basis[node_local]
                cross_scatter[node_local] += scale * w_x * u_bar_gp * u_prime_gp
                reynolds_scatter[node_local] += scale * w_x * 0.5 * u_prime_gp**2
                viscous_scatter[node_local] += scale * w_x * du_prime_dx_gp
            # temporal terms go to their respective node only
            temporal_l_scatter[0] += scale * basis_vals[0] * du_prime_dt_gp
            temporal_r_scatter[1] += scale * basis_vals[1] * du_prime_dt_gp

        scatter_array = np.stack(
            [
                cross_scatter,
                reynolds_scatter,
                temporal_l_scatter,
                temporal_r_scatter,
                viscous_scatter,
            ],
            axis=1,
        )

        label_array = np.array(
            [
                cross_scatter[1],
                reynolds_scatter[1],
                temporal_l_scatter[0],
                temporal_r_scatter[1],
                viscous_scatter[1],
            ]
        )

        return ElementSGSTerms(scatter=scatter_array, label=label_array)

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
        if self.sgs_save_path is not None:
            self.sgs_save_path.mkdir(parents=True, exist_ok=True)
            self.save_sgsp_training_csv(append_mode=self.append_mode)

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

    def save_sgsp_training_csv(
        self,
        train_fraction: float = 0.8,
        random_seed: int = 42,
        append_mode: bool = False,
    ) -> None:
        """Flatten assembled stencils/labels, normalize, split, and save to CSV.

        In append_mode, loads existing X_raw.csv/y_raw.csv and prepends to new
        rows before normalizing over the full combined dataset.

        Saves:
            X_raw.csv, y_raw.csv          (un-normalized, full dataset)
            X_train.csv, y_train.csv
            X_val.csv,   y_val.csv
            normalisation_stats.csv
        """
        x_rows: list[NDArray] = []
        y_rows: list[NDArray] = []

        n_snapshots = len(self.assembled_input_stencils)
        times_sliced = self.requested_snapshots[:n_snapshots]

        # LFS: features at level n predict closure at level n+1 (Rajampeta Sec. 5.3).
        # Requires consecutive snapshots — one list entry == one dt.
        assert self.snapshot_factor == 1, (
            "LFS label shift assumes snapshot_factor == 1; "
            f"got {self.snapshot_factor}. Shift would span multiple dt."
        )

        for stencil_list, sgs_term_list in zip(
            self.assembled_input_stencils[:-1],
            self.assembled_sgs_terms[1:],
        ):
            for stencil_vec, element_terms in zip(stencil_list, sgs_term_list):
                if stencil_vec is None:
                    continue
                x_rows.append(stencil_vec)
                y_rows.append(element_terms.label)

        if not x_rows:
            print("No valid training pairs found; nothing saved.")
            return

        x_new = np.array(x_rows, dtype=np.float64)
        y_new = np.array(y_rows, dtype=np.float64)

        if append_mode:
            x_raw_path = self.sgs_save_path / "X_raw.csv"
            y_raw_path = self.sgs_save_path / "y_raw.csv"
            if x_raw_path.exists() and y_raw_path.exists():
                x_existing = np.loadtxt(x_raw_path, delimiter=",", skiprows=1)
                y_existing = np.loadtxt(y_raw_path, delimiter=",", skiprows=1)
                x_matrix = np.vstack([x_existing, x_new])
                y_matrix = np.vstack([y_existing, y_new])
            else:
                x_matrix = x_new
                y_matrix = y_new
        else:
            x_matrix = x_new
            y_matrix = y_new

        self._write_csv(
            save_path=self.sgs_save_path / "X_raw.csv",
            data=x_matrix,
            header=_INPUT_COLS,
        )
        self._write_csv(
            save_path=self.sgs_save_path / "y_raw.csv",
            data=y_matrix,
            header=_OUTPUT_COLS,
        )

        x_mean, x_std, y_mean, y_std = self._compute_normalisation_stats(
            x_matrix, y_matrix
        )
        x_normalised = (x_matrix - x_mean) / x_std
        y_normalised = (y_matrix - y_mean) / y_std

        rng = np.random.default_rng(random_seed)
        shuffled_indices = rng.permutation(x_normalised.shape[0])
        split_index = int(train_fraction * len(shuffled_indices))
        train_indices = shuffled_indices[:split_index]
        val_indices = shuffled_indices[split_index:]

        self._write_csv(
            save_path=self.sgs_save_path / "X_train.csv",
            data=x_normalised[train_indices],
            header=_INPUT_COLS,
        )
        self._write_csv(
            save_path=self.sgs_save_path / "y_train.csv",
            data=y_normalised[train_indices],
            header=_OUTPUT_COLS,
        )
        self._write_csv(
            save_path=self.sgs_save_path / "X_val.csv",
            data=x_normalised[val_indices],
            header=_INPUT_COLS,
        )
        self._write_csv(
            save_path=self.sgs_save_path / "y_val.csv",
            data=y_normalised[val_indices],
            header=_OUTPUT_COLS,
        )
        self._save_normalisation_stats_csv(
            save_path=self.sgs_save_path / "normalisation_stats.csv",
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )

        print(
            f"SGSP training data saved to '{self.sgs_save_path}':\n"
            f"  train: {len(train_indices)} samples | val: {len(val_indices)} samples\n"
            f"  snapshots used: {n_snapshots} | times: {times_sliced[:3]}..."
        )

    @staticmethod
    def _compute_normalisation_stats(
        x_matrix: NDArray, y_matrix: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Compute zero-mean unit-variance stats; clip std to avoid division by zero."""
        x_mean = x_matrix.mean(axis=0)
        x_std = x_matrix.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        y_mean = y_matrix.mean(axis=0)
        y_std = y_matrix.std(axis=0)
        y_std[y_std < 1e-12] = 1.0
        return x_mean, x_std, y_mean, y_std

    @staticmethod
    def _write_csv(save_path: Path, data: NDArray, header: list[str]) -> None:
        """Write a 2-D array to CSV with a header row."""
        with open(save_path, mode="w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(header)
            writer.writerows(data.tolist())

    @staticmethod
    def _save_normalisation_stats_csv(
        save_path: Path,
        x_mean: NDArray,
        x_std: NDArray,
        y_mean: NDArray,
        y_std: NDArray,
    ) -> None:
        """Save normalization stats to CSV; rows = [stat_name, *values]."""
        with open(save_path, mode="w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["stat"] + _INPUT_COLS + _OUTPUT_COLS)
            writer.writerow(["x_mean"] + x_mean.tolist() + [""] * len(_OUTPUT_COLS))
            writer.writerow(["x_std"] + x_std.tolist() + [""] * len(_OUTPUT_COLS))
            writer.writerow(["y_mean"] + [""] * len(_INPUT_COLS) + y_mean.tolist())
            writer.writerow(["y_std"] + [""] * len(_INPUT_COLS) + y_std.tolist())

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


# ---------------------------------------------------------------------------
# CSV loading utilities (for use by SGSPredictor / coupled solver)
# ---------------------------------------------------------------------------


def load_sgsp_training_csv(
    data_path: Path,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load normalized (X_train, y_train, X_val, y_val) from CSV files.

    Returns float64 arrays; callers should convert to float32 for PyTorch.
    """
    data_path = Path(data_path)

    def _read(filename: str) -> NDArray:
        return np.loadtxt(data_path / filename, delimiter=",", skiprows=1)

    return (
        _read("X_train.csv"),
        _read("y_train.csv"),
        _read("X_val.csv"),
        _read("y_val.csv"),
    )


def load_normalisation_stats_csv(
    data_path: Path,
) -> dict[str, NDArray]:
    """Load normalization stats saved by BurgersDataGenerator.

    Returns dict with keys: x_mean, x_std (shape 20), y_mean, y_std (shape 5).
    Raises ValueError if shapes are inconsistent with expected stencil dimensions.
    """
    data_path = Path(data_path)
    stats: dict[str, NDArray] = {}
    with open(data_path / "normalisation_stats.csv", newline="") as file_handle:
        reader = csv.reader(file_handle)
        next(reader)  # skip header row
        for row in reader:
            stat_name: str = row[0]
            values: list[float] = [float(v_str) for v_str in row[1:] if v_str != ""]
            stats[stat_name] = np.array(values, dtype=np.float64)

    expected_shapes = {"x_mean": 20, "x_std": 20, "y_mean": 5, "y_std": 5}
    for key, expected_len in expected_shapes.items():
        if key not in stats:
            raise ValueError(f"Missing key '{key}' in normalisation_stats.csv")
        if len(stats[key]) != expected_len:
            raise ValueError(
                f"Expected '{key}' to have length {expected_len}, got {len(stats[key])}"
            )

    return stats


# ---------------------------------------------------------------------------
# ProjDNSReconstructor
# ---------------------------------------------------------------------------


class ProjDNSReconstructor(BurgersBase):
    """Reconstruct DNS on the LES grid using exact closure terms from BurgersDataGenerator.

    Used to validate that computed closure terms perfectly reproduce the DNS projection.
    """

    def __init__(
        self,
        dns_solutions: list[NDArray],
        u_bar_solutions: list[NDArray],
        closure_terms: list[list[ElementSGSTerms]],
        problem: Problem,
        disc_cfg: DiscretizationConfig,
        master_path: Path,
        simulation_mode: str = "no_model",
        snapshot_factor: int = 1,
        use_closure_terms: bool = True,
        use_temporal_terms: bool = True,
        warmup_offset: int = WARMUP_STEPS,
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
        self.use_temporal_terms = use_temporal_terms
        self.warmup_offset = warmup_offset

        self.time_steps_stepped: int = 0

    def recreate_solution(self) -> None:
        """March from the first closured level using exact closure terms.

        Seeds from the projected DNS at level ``warmup_offset`` (the last level
        with no closure term) so ``closure_terms[0]`` (IT at warmup_offset+1)
        lands on the first solve.
        """
        self.solution = self.u_bar_solutions[self.warmup_offset].copy()
        self.initial_condition = self.solution.copy()
        self.simulation_time_elapsed = self.warmup_offset * self.dt

        self.resolve_current_forcing()
        self._extract_snapshot()

        for _ in range(len(self.closure_terms)):
            self.advance_time_step()
            self._extract_snapshot()

        self.write_config_to_json()
        self.write_solution_to_csv()

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """Newton-Raphson iteration; returns U^{n+1}."""
        solution_n = solution.copy()
        solution_k = solution.copy()
        residual_history_loop: list = []
        update_history_loop: list = []

        self.max_iterations = 50

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
        """Advance the solution by one time step: U^{n+1} <- U^n."""
        self.resolve_current_forcing()
        self.solution = self.nr_iteration(self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt
        self.time_steps_stepped += 1

    def add_closure_terms_to_residual(
        self, residual: NDArray, time_step: int
    ) -> NDArray:
        """Scatter all five SGS contributions from each element to both nodes."""
        snapshot_idx = min(time_step, len(self.closure_terms) - 1)
        for element_idx, element_left_node in enumerate(self.nodes_les[:-1]):
            element_terms: ElementSGSTerms = self.closure_terms[snapshot_idx][
                element_idx
            ]
            for local_node, global_node in enumerate(
                [element_left_node, element_left_node + 1]
            ):
                if global_node in (0, self.nodes_les[-1]):
                    continue
                cross_term: float = element_terms.scatter[local_node, 0]
                reynolds_term: float = element_terms.scatter[local_node, 1]
                if self.use_temporal_terms:
                    temporal_l_term: float = element_terms.scatter[local_node, 2]
                    temporal_r_term: float = element_terms.scatter[local_node, 3]
                else:
                    temporal_l_term = 0
                    temporal_r_term = 0
                viscous_term: float = element_terms.scatter[local_node, 4]

                correction: float = (
                    cross_term
                    + reynolds_term
                    + temporal_l_term
                    + temporal_r_term
                    - self.viscosity * viscous_term
                )

                residual[global_node] -= correction
        return residual

    def plot_solution_comparison(
        self,
        reconstructed_idx: int,
        dns_solutions: list[NDArray],
        u_bar_solutions: list[NDArray],
        reconstructed_no_model: NDArray | None = None,
    ) -> None:
        """Compare DNS, LES projection, and reconstruction at one reconstructed step."""
        truth_idx: int = self.warmup_offset + reconstructed_idx
        dns_sol: NDArray = dns_solutions[truth_idx]
        u_bar_sol: NDArray = u_bar_solutions[truth_idx]
        reconstructed_sol: NDArray = self.snapshots[reconstructed_idx][0]


        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.disc_cfg.mesh_dns, dns_sol, label="DNS", color="gray", alpha=0.8)
        ax.plot(
            self.disc_cfg.mesh_les,
            u_bar_sol,
            label="DNS (projection)",
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
                reconstructed_no_model[reconstructed_idx][0],
                label="reconstructed (no closure)",
                color="tab:green",
                marker="s",
                linestyle=":",
            )
        ax.set_title(f"Snapshot {reconstructed_idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _disc_cfg = DiscretizationConfig(
        n_nodes_les=17,
        temporal_refinement=1,
        courant_les=0.1,
        domain_length=1,
    )
    _path = Path(__file__).parent.parent / "test_suite"
    _problem = replace(Problems.raj_two, domain_timespan=1.0)

    _solver = BurgersDataGenerator(
        _problem,
        disc_cfg=_disc_cfg,
        simulation_mode="dns",
        master_path=_path,
        warmup_steps=3,
    )
    _solver.run_simulation()
    _solver.plotting_interpolation_and_projection()

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
        use_temporal_terms=True,
    )
    _recreator.recreate_solution()
    _recreator.plot_solution_comparison(
        reconstructed_idx=len(_recreator.snapshots) - 1,
        dns_solutions=_solver.solution_history,
        u_bar_solutions=_solver.u_bar_history,
        reconstructed_no_model=_recreator_no_model.snapshots,
    )
