"""Assemble input/output training data for the VMS-ANN predictor model.

Output stencil follows Rajampeta (2022) / Research Proposal formulation.
Per element e, the 5 interaction terms are:

    [0] (w_x, ū·u')_e        cross term
    [1] (w_x, u'²/2)_e       Reynolds term
    [2] (w_l, u'_t)_e        temporal, left weight
    [3] (w_r, u'_t)_e        temporal, right weight
    [4] (w_x, u'_x)_e        viscous SGS term

Input stencil (Rajampeta FS2):
    [ū^{n,n-1,n-2}_{i-2:i+1}, (∂ū/∂t)^n_{i-2:i+1}, f^n_{i-2:i+1}] → 20 features

References: Rajampeta (2022) Sec. 4.3 / Table 4.4, Research Proposal Sec. 2.3.1,
            Robijns (2019) Sec. 3.2.1.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------


def _basis_functions(ksi: float) -> NDArray:
    """Linear basis functions on the reference element [-1, 1]."""
    return np.array([0.5 * (1.0 - ksi), 0.5 * (1.0 + ksi)])


def _gradient_basis_functions(element_size: float) -> NDArray:
    """Constant gradient of linear basis on physical element: dN/dx = [-1, 1] / h."""
    return np.array([-1.0, 1.0]) / element_size


# ---------------------------------------------------------------------------
# Per-element output term computation
# ---------------------------------------------------------------------------


def compute_element_output_terms(
    u_bar_left: float,
    u_bar_right: float,
    u_prime_left: float,
    u_prime_right: float,
    du_bar_dt_left: float,
    du_bar_dt_right: float,
    du_prime_dt_left: float,
    du_prime_dt_right: float,
    du_prime_dx_left: float,
    du_prime_dx_right: float,
    element_size: float,
) -> NDArray:
    """Compute the 5-component output vector for one element via 2-point Gauss quadrature.

    Returns NDArray of shape (5,):
        [cross, Reynolds, temporal_L, temporal_R, viscous].
    """
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(deg=2)
    grad_basis = _gradient_basis_functions(element_size)
    jacobian = element_size / 2.0

    u_bar_nodes = np.array([u_bar_left, u_bar_right])
    u_prime_nodes = np.array([u_prime_left, u_prime_right])
    du_prime_dt_nodes = np.array([du_prime_dt_left, du_prime_dt_right])
    du_prime_dx_nodes = np.array([du_prime_dx_left, du_prime_dx_right])

    cross_term = reynolds_term = temporal_left = temporal_right = viscous_term = 0.0
    w_x_val = grad_basis[1]  # = 1/h, constant over element

    for gauss_pt, gauss_wt in zip(gauss_pts, gauss_wts):
        basis_vals = _basis_functions(gauss_pt)
        scale = gauss_wt * jacobian

        u_bar_gp = float(basis_vals @ u_bar_nodes)
        u_prime_gp = float(basis_vals @ u_prime_nodes)
        du_prime_dt_gp = float(basis_vals @ du_prime_dt_nodes)
        du_prime_dx_gp = float(basis_vals @ du_prime_dx_nodes)

        cross_term += scale * w_x_val * u_bar_gp * u_prime_gp
        reynolds_term += scale * w_x_val * 0.5 * u_prime_gp**2
        temporal_left += scale * basis_vals[0] * du_prime_dt_gp
        temporal_right += scale * basis_vals[1] * du_prime_dt_gp
        viscous_term += scale * w_x_val * du_prime_dx_gp

    return np.array(
        [cross_term, reynolds_term, temporal_left, temporal_right, viscous_term]
    )


# ---------------------------------------------------------------------------
# u' and its derivatives
# ---------------------------------------------------------------------------


def compute_u_prime_field(u_dns_on_les: NDArray, u_bar: NDArray) -> NDArray:
    """Unresolved field: u' = u_dns_projected - ū at every LES node."""
    return u_dns_on_les - u_bar


def compute_du_prime_dx(u_prime: NDArray, element_size: float) -> NDArray:
    """Approximate ∂u'/∂x via central differences (forward/backward at boundaries)."""
    du_dx = np.empty_like(u_prime)
    du_dx[1:-1] = (u_prime[2:] - u_prime[:-2]) / (2.0 * element_size)
    du_dx[0] = (u_prime[1] - u_prime[0]) / element_size
    du_dx[-1] = (u_prime[-1] - u_prime[-2]) / element_size
    return du_dx


def compute_du_prime_dt(
    u_prime_now: NDArray,
    u_prime_prev: NDArray | None,
    dt: float,
) -> NDArray:
    """Backward-Euler time derivative of u'; zero at first snapshot."""
    if u_prime_prev is None:
        return np.zeros_like(u_prime_now)
    return (u_prime_now - u_prime_prev) / dt


# ---------------------------------------------------------------------------
# Input stencil assembly
# ---------------------------------------------------------------------------


def build_input_stencil(
    u_bar_history: list[NDArray],
    du_bar_dt_history: list[NDArray],
    forcing_history: list[NDArray],
    element_idx: int,
    n_les_nodes: int,
) -> NDArray | None:
    """Build the 20-feature FS2 input vector for element element_idx at time level n.

    Stencil: [ū^{n,n-1,n-2}_{i-2:i+1}, (∂ū/∂t)^n_{i-2:i+1}, f^n_{i-2:i+1}].
    Returns None if the stencil falls outside the domain.
    """
    if len(u_bar_history) < 3:
        return None

    stencil_nodes = np.array(
        [element_idx - 2, element_idx - 1, element_idx, element_idx + 1]
    )
    if stencil_nodes[0] < 0 or stencil_nodes[-1] >= n_les_nodes:
        return None

    return np.concatenate(
        [
            u_bar_history[-1][stencil_nodes],
            u_bar_history[-2][stencil_nodes],
            u_bar_history[-3][stencil_nodes],
            du_bar_dt_history[-1][stencil_nodes],
            forcing_history[-1][stencil_nodes],
        ]
    )


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------


def assemble_training_data(
    solutions_les: NDArray,
    solutions_dns_projected: NDArray,
    forcings_les: NDArray,
    dt: float,
    element_size: float,
) -> tuple[NDArray, NDArray, dict]:
    """Assemble and normalise (X, y) training pairs from projected LES/DNS data.

    Skips snapshots 0–1 (need 3 time levels) and boundary elements (stencil
    falls outside domain). Returns (X_normalised, y_normalised, norm_stats).
    """
    n_timesteps, n_les_nodes = solutions_les.shape
    n_elements = n_les_nodes - 1

    u_prime_all, du_prime_dt_all, du_prime_dx_all, du_bar_dt_all = [], [], [], []

    for time_idx in range(n_timesteps):
        u_bar = solutions_les[time_idx]
        u_prime = compute_u_prime_field(solutions_dns_projected[time_idx], u_bar)
        u_prime_all.append(u_prime)
        du_prime_dx_all.append(compute_du_prime_dx(u_prime, element_size))
        du_prime_dt_all.append(
            compute_du_prime_dt(
                u_prime, u_prime_all[time_idx - 1] if time_idx > 0 else None, dt
            )
        )
        u_bar_prev = solutions_les[time_idx - 1] if time_idx > 0 else None
        du_bar_dt_all.append(
            (u_bar - u_bar_prev) / dt
            if u_bar_prev is not None
            else np.zeros_like(u_bar)
        )

    input_rows: list[NDArray] = []
    output_rows: list[NDArray] = []

    for time_idx in range(2, n_timesteps):
        u_bar_history = [
            solutions_les[time_idx - 2],
            solutions_les[time_idx - 1],
            solutions_les[time_idx],
        ]
        du_bar_dt_history = [
            du_bar_dt_all[time_idx - 2],
            du_bar_dt_all[time_idx - 1],
            du_bar_dt_all[time_idx],
        ]
        forcing_history = [
            forcings_les[time_idx - 2],
            forcings_les[time_idx - 1],
            forcings_les[time_idx],
        ]

        u_bar_cur = solutions_les[time_idx]
        u_prime_cur = u_prime_all[time_idx]
        du_prime_dt_cur = du_prime_dt_all[time_idx]
        du_prime_dx_cur = du_prime_dx_all[time_idx]
        du_bar_dt_cur = du_bar_dt_all[time_idx]

        for elem_idx in range(n_elements):
            input_vec = build_input_stencil(
                u_bar_history=u_bar_history,
                du_bar_dt_history=du_bar_dt_history,
                forcing_history=forcing_history,
                element_idx=elem_idx,
                n_les_nodes=n_les_nodes,
            )
            if input_vec is None:
                continue

            node_left, node_right = elem_idx, elem_idx + 1
            output_vec = compute_element_output_terms(
                u_bar_left=float(u_bar_cur[node_left]),
                u_bar_right=float(u_bar_cur[node_right]),
                u_prime_left=float(u_prime_cur[node_left]),
                u_prime_right=float(u_prime_cur[node_right]),
                du_bar_dt_left=float(du_bar_dt_cur[node_left]),
                du_bar_dt_right=float(du_bar_dt_cur[node_right]),
                du_prime_dt_left=float(du_prime_dt_cur[node_left]),
                du_prime_dt_right=float(du_prime_dt_cur[node_right]),
                du_prime_dx_left=float(du_prime_dx_cur[node_left]),
                du_prime_dx_right=float(du_prime_dx_cur[node_right]),
                element_size=element_size,
            )
            input_rows.append(input_vec)
            output_rows.append(output_vec)

    x_matrix = np.array(input_rows, dtype=np.float64)
    y_matrix = np.array(output_rows, dtype=np.float64)

    x_std = x_matrix.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_std = y_matrix.std(axis=0)
    y_std[y_std < 1e-12] = 1.0

    x_mean = x_matrix.mean(axis=0)
    y_mean = y_matrix.mean(axis=0)

    norm_stats = {"X_mean": x_mean, "X_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return (x_matrix - x_mean) / x_std, (y_matrix - y_mean) / y_std, norm_stats


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------


def split_and_save(
    x_data: NDArray,
    y_data: NDArray,
    normalisation_stats: dict,
    output_dir: Path,
    train_fraction: float = 0.8,
    random_seed: int = 42,
) -> None:
    """Shuffle, split into train/val, and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    shuffled_indices = rng.permutation(x_data.shape[0])
    split_index = int(train_fraction * len(shuffled_indices))

    train_indices = shuffled_indices[:split_index]
    val_indices = shuffled_indices[split_index:]
    val_indices_sorted = np.sort(val_indices)

    np.save(output_dir / "X_train.npy", x_data[train_indices])
    np.save(output_dir / "y_train.npy", y_data[train_indices])
    np.save(output_dir / "X_val.npy", x_data[val_indices])
    np.save(output_dir / "y_val.npy", y_data[val_indices])
    np.save(output_dir / "X_val_sequential.npy", x_data[val_indices_sorted])
    np.save(output_dir / "y_val_sequential.npy", y_data[val_indices_sorted])
    np.savez(output_dir / "normalisation_stats.npz", **normalisation_stats)

    print(
        f"Saved {len(train_indices)} training and {len(val_indices)} validation samples to '{output_dir}'.\n"
        f"  X_train shape: {x_data[train_indices].shape} | y_train shape: {y_data[train_indices].shape}"
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_training_data_assembly(
    projection_path: Path,
    output_dir: Path,
    dt: float,
    element_size: float,
    train_fraction: float = 0.8,
    random_seed: int = 42,
) -> tuple[NDArray, NDArray, dict]:
    """Load projected data, assemble (X, y), split, and save. Returns (X, y, norm_stats)."""
    projection_path = Path(projection_path)

    solutions_les = np.load(projection_path / "solutions_projection.npy")

    dns_on_les_path = projection_path / "dns_on_les.npy"
    if not dns_on_les_path.exists():
        raise FileNotFoundError(
            f"DNS-on-LES snapshot file not found at '{dns_on_les_path}'. "
            "Run the projection step first."
        )
    solutions_dns_projected = np.load(dns_on_les_path)

    forcings_path = projection_path / "forcings_projection.npy"
    if forcings_path.exists():
        forcings_les = np.load(forcings_path)
    else:
        forcings_les = np.zeros_like(solutions_les)
        print("Warning: no forcing file found; using zeros.")

    x_data, y_data, norm_stats = assemble_training_data(
        solutions_les=solutions_les,
        solutions_dns_projected=solutions_dns_projected,
        forcings_les=forcings_les,
        dt=dt,
        element_size=element_size,
    )
    split_and_save(
        x_data=x_data,
        y_data=y_data,
        normalisation_stats=norm_stats,
        output_dir=output_dir,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    return x_data, y_data, norm_stats
