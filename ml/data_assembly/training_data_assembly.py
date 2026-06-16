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


from problems_and_configurations.disc_config import DiscretisationConfig

PROJECTION_GAUSS_POINTS = 6


def _basis_functions(ksi: float) -> NDArray:
    """Linear basis functions on the reference element [-1, 1]."""
    return np.array([0.5 * (1.0 - ksi), 0.5 * (1.0 + ksi)])


def gradient_basis_functions(element_size: float) -> NDArray:
    """Constant gradient of linear basis on physical element: dN/dx = [-1, 1] / h."""
    return np.array([-1.0, 1.0]) / element_size


def compute_element_output_terms(
    u_dns_now: NDArray,
    u_dns_prev: NDArray | None,
    u_bar_now: NDArray,
    u_bar_prev: NDArray | None,
    mesh_dns: NDArray,
    x_left: float,
    x_right: float,
    element_size_les: float,
    element_size_dns: float,
    dt: float,
) -> NDArray:
    """Compute the 5-component output vector for one LES element.

    Integrates interaction terms at full DNS resolution via Gauss quadrature,
    interpolating DNS fields to each quadrature point. Returns shape (5,):
        [cross, Reynolds, temporal_L, temporal_R, viscous].
    """
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(deg=PROJECTION_GAUSS_POINTS)
    grad_basis = gradient_basis_functions(element_size_les)
    jacobian = element_size_les / 2.0
    w_x_val = grad_basis[1]

    u_prime_now_dns = u_dns_now - u_bar_now
    du_prime_dx_dns = np.gradient(u_prime_now_dns, element_size_dns)

    if u_dns_prev is not None and u_bar_prev is not None:
        u_prime_prev_dns = u_dns_prev - u_bar_prev
        du_prime_dt_dns = (u_prime_now_dns - u_prime_prev_dns) / dt
    else:
        du_prime_dt_dns = np.zeros_like(u_prime_now_dns)

    cross_term = reynolds_term = temporal_left = temporal_right = viscous_term = 0.0

    for gauss_pt, gauss_wt in zip(gauss_pts, gauss_wts):
        x_phys = 0.5 * (x_left + x_right) + 0.5 * element_size_les * gauss_pt
        basis_vals = _basis_functions(gauss_pt)
        scale = gauss_wt * jacobian

        u_bar_gp = float(np.interp(x_phys, mesh_dns, u_bar_now))
        u_prime_gp = float(np.interp(x_phys, mesh_dns, u_prime_now_dns))
        du_prime_dt_gp = float(np.interp(x_phys, mesh_dns, du_prime_dt_dns))
        du_prime_dx_gp = float(np.interp(x_phys, mesh_dns, du_prime_dx_dns))

        cross_term += scale * w_x_val * u_bar_gp * u_prime_gp
        reynolds_term += scale * w_x_val * 0.5 * u_prime_gp**2
        temporal_left += scale * basis_vals[0] * du_prime_dt_gp
        temporal_right += scale * basis_vals[1] * du_prime_dt_gp
        viscous_term += scale * w_x_val * du_prime_dx_gp

    return np.array(
        [cross_term, reynolds_term, temporal_left, temporal_right, viscous_term]
    )


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


def assemble_training_data(
    solutions_les: NDArray,
    solutions_dns_raw: NDArray,
    forcings_les: NDArray,
    mesh_dns: NDArray,
    mesh_les: NDArray,
    dt: float,
    element_size_les: float,
    element_size_dns: float,
) -> tuple[NDArray, NDArray, dict]:
    """Assemble and normalize (X, y) training pairs from projected LES/DNS data.

    Computes interaction terms at full DNS resolution by integrating u' = u_DNS - u_bar
    against LES basis functions via Gauss quadrature. Skips snapshots 0-1 (need 3 time
    levels) and boundary elements (stencil falls outside domain).
    Returns (X_normalized, y_normalized, norm_stats).
    """
    n_timesteps, n_les_nodes = solutions_les.shape
    n_elements = n_les_nodes - 1

    # Interpolate L2-projected LES solutions onto DNS mesh for u' computation
    u_bar_on_dns_all = [
        np.interp(mesh_dns, mesh_les, solutions_les[t]) for t in range(n_timesteps)
    ]

    # Precompute du_bar_dt on LES mesh for input stencil construction
    du_bar_dt_all: list[NDArray] = []
    for time_idx in range(n_timesteps):
        u_bar_prev = solutions_les[time_idx - 1] if time_idx > 0 else None
        du_bar_dt_all.append(
            (solutions_les[time_idx] - u_bar_prev) / dt
            if u_bar_prev is not None
            else np.zeros(n_les_nodes)
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

            x_left = float(mesh_les[elem_idx])
            x_right = float(mesh_les[elem_idx + 1])

            output_vec = compute_element_output_terms(
                u_dns_now=solutions_dns_raw[time_idx],
                u_dns_prev=solutions_dns_raw[time_idx - 1],
                u_bar_now=np.asarray(u_bar_on_dns_all[time_idx]),
                u_bar_prev=np.asarray(u_bar_on_dns_all[time_idx - 1]),
                mesh_dns=mesh_dns,
                x_left=x_left,
                x_right=x_right,
                element_size_les=element_size_les,
                element_size_dns=element_size_dns,
                dt=dt,
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


def run_training_data_assembly(
    projection_path: Path,
    output_dir: Path,
    disc_cfg: DiscretisationConfig,
    train_fraction: float = 0.8,
    random_seed: int = 42,
) -> tuple[NDArray, NDArray, dict]:
    """Load projected data, assemble (X, y), split, and save. Returns (X, y, norm_stats)."""
    projection_path = Path(projection_path)
    solutions_proj = np.load(projection_path / "solutions_projection.npy")
    dns_raw_path = projection_path / "solutions_dns_raw.npy"
    if not dns_raw_path.exists():
        raise FileNotFoundError(
            f"DNS raw snapshot file not found at '{dns_raw_path}'. "
            "Run the projection step first."
        )
    solutions_dns_raw = np.load(dns_raw_path)
    mesh_dns = np.load(projection_path / "mesh_dns.npy")
    forcings_path = projection_path / "forcings_projection.npy"
    forcings_les = (
        np.load(forcings_path)
        if forcings_path.exists()
        else np.zeros_like(solutions_proj)
    )

    x_data, y_data, norm_stats = assemble_training_data(
        solutions_les=solutions_proj,
        solutions_dns_raw=solutions_dns_raw,
        forcings_les=forcings_les,
        mesh_dns=mesh_dns,
        mesh_les=disc_cfg.mesh_les,
        dt=disc_cfg.dt_les,
        element_size_les=disc_cfg.h_les,
        element_size_dns=disc_cfg.h_dns,
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
