"""Assemble input/output training data for the VMS-ANN predictor model.

Output stencil follows Rajampeta (2022) / Research Proposal formulation.
The unresolved-scale interaction terms (from the two-scale VMS Burgers eq.) are:

    I = (w, u'_t) - (w_x, ū·u') - (w_x, u'²/2) + ν(w_x, u'_x)

Per element e with left/right shape functions w_l, w_r (constant gradient w_x):

    output = [
        (w_x,  ū·u')_e          # cross term (sign-flipped in residual)
        (w_x,  u'²/2)_e         # Reynolds term (sign-flipped in residual)
        (w_l,  u'_t)_e          # u't-term, left weight
        (w_r,  u'_t)_e          # u't-term, right weight
        (w_x,  u'_x)_e          # u'x-term (viscous, sign-flipped in residual)
    ]  → 5 scalars per element

This separates the cross and Reynolds contributions that Robijns (2019) merged
into a single ⟨w_x, ū² + ½u'u'⟩ term. The ū² part is fully resolved and handled
by the Galerkin operator; including it in the target conflates known and unknown
quantities, which is the source of the comment you received.

Input stencil (Rajampeta FS2 / Research Proposal):
    Lagged extended stencil — uses time levels n, n-1, n-2 to avoid CPI.
    Spatial span: nodes i-2 … i+1 (element boundaries + one outer neighbour
    on each side).

    input = [ū^{n,n-1,n-2}_{i-2:i+1},  (∂ū/∂t)^n_{i-2:i+1},  f^n_{i-2:i+1}]
          = 4*3 + 4 + 4 = 20 features per element

References
----------
Rajampeta (2022), Section 4.3 / Table 4.4 (FS2).
Research Proposal, Section 2.3.1.
Robijns (2019), Section 3.2.1.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Quadrature helpers (Gauss–Legendre on reference element [-1, 1])
# ---------------------------------------------------------------------------


def _gauss_legendre_2pt() -> tuple[NDArray, NDArray]:
    """Return 2-point Gauss–Legendre nodes and weights on [-1, 1]."""
    points, weights = np.polynomial.legendre.leggauss(deg=2)
    return points, weights


def _basis_functions(ksi: float) -> NDArray:
    """Linear basis functions on the reference element."""
    return np.array([0.5 * (1.0 - ksi), 0.5 * (1.0 + ksi)])


def _gradient_basis_functions(element_size: float) -> NDArray:
    """Constant gradient of linear basis on physical element.

    For a uniform mesh: dN/dx = [-1, 1] * (1/h).
    """
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
    """Compute the 5-component output vector for one element via Gaussian quadrature.

    All arguments are nodal values at the left (i) and right (i+1) boundaries.

    The five outputs correspond to the interaction terms integrated against the
    weighting functions of the element (Rajampeta 2022, eq. 4.9):

        [0] (w_x, ū·u')_e        cross term
        [1] (w_x, u'²/2)_e       Reynolds term
        [2] (w_l, u'_t)_e        temporal, left weight
        [3] (w_r, u'_t)_e        temporal, right weight
        [4] (w_x, u'_x)_e        viscous sub-grid term

    Returns
    -------
    output_terms : NDArray of shape (5,)
    """
    gauss_pts, gauss_wts = _gauss_legendre_2pt()
    grad_basis = _gradient_basis_functions(element_size)  # shape (2,), constant
    jacobian = element_size / 2.0

    u_bar_nodes = np.array([u_bar_left, u_bar_right])
    u_prime_nodes = np.array([u_prime_left, u_prime_right])
    du_prime_dt_nodes = np.array([du_prime_dt_left, du_prime_dt_right])
    du_prime_dx_nodes = np.array([du_prime_dx_left, du_prime_dx_right])

    # Accumulated integrals
    cross_term = 0.0
    reynolds_term = 0.0
    temporal_left = 0.0
    temporal_right = 0.0
    viscous_term = 0.0

    for gauss_pt, gauss_wt in zip(gauss_pts, gauss_wts):
        basis_vals = _basis_functions(gauss_pt)
        scale = gauss_wt * jacobian

        u_bar_gp = float(basis_vals @ u_bar_nodes)
        u_prime_gp = float(basis_vals @ u_prime_nodes)
        du_prime_dt_gp = float(basis_vals @ du_prime_dt_nodes)
        du_prime_dx_gp = float(basis_vals @ du_prime_dx_nodes)

        # w_x is constant over the element (linear elements), use grad_basis[1]
        # (same magnitude as grad_basis[0] but opposite sign; the sign is
        #  absorbed into the residual assembly, here we store the raw integral)
        w_x_val = grad_basis[1]  # = 1/h

        cross_term += scale * w_x_val * u_bar_gp * u_prime_gp
        reynolds_term += scale * w_x_val * 0.5 * u_prime_gp**2
        temporal_left += scale * basis_vals[0] * du_prime_dt_gp
        temporal_right += scale * basis_vals[1] * du_prime_dt_gp
        viscous_term += scale * w_x_val * du_prime_dx_gp

    return np.array(
        [cross_term, reynolds_term, temporal_left, temporal_right, viscous_term]
    )


# ---------------------------------------------------------------------------
# u' and its derivatives from projected LES / DNS data
# ---------------------------------------------------------------------------


def compute_u_prime_field(
    u_dns_on_les: NDArray,
    u_bar: NDArray,
) -> NDArray:
    """Unresolved field: u' = u_dns_projected - ū at every LES node."""
    return u_dns_on_les - u_bar


def compute_du_prime_dx(
    u_prime: NDArray,
    element_size: float,
) -> NDArray:
    """Approximate ∂u'/∂x at LES nodes via central differences (forward/backward at BCs)."""
    du_prime_dx_vals = np.empty_like(u_prime)
    du_prime_dx_vals[1:-1] = (u_prime[2:] - u_prime[:-2]) / (2.0 * element_size)
    du_prime_dx_vals[0] = (u_prime[1] - u_prime[0]) / element_size
    du_prime_dx_vals[-1] = (u_prime[-1] - u_prime[-2]) / element_size
    return du_prime_dx_vals


def compute_du_prime_dt(
    u_prime_now: NDArray,
    u_prime_prev: NDArray | None,
    dt: float,
) -> NDArray:
    """Backward-Euler time derivative of u'.  Zero at first snapshot."""
    if u_prime_prev is None:
        return np.zeros_like(u_prime_now)
    return (u_prime_now - u_prime_prev) / dt


# ---------------------------------------------------------------------------
# Input stencil assembly (Rajampeta FS2 / lagged extended stencil)
# ---------------------------------------------------------------------------


def build_input_stencil(
    u_bar_history: list[NDArray],
    du_bar_dt_history: list[NDArray],
    forcing_history: list[NDArray],
    element_idx: int,
    n_les_nodes: int,
) -> NDArray | None:
    """Build the 20-feature input vector for element *element_idx* at time level n.

    Uses the lagged extended stencil (FS2):
        [ū^{n,n-1,n-2}_{i-2:i+1},  (∂ū/∂t)^n_{i-2:i+1},  f^n_{i-2:i+1}]

    The stencil spans nodes i-2, i-1, i, i+1 (4 nodes = element + 1 outer
    neighbour per side).  Re and h are held constant so excluded per RP.

    Parameters
    ----------
    u_bar_history:
        List of LES snapshots ordered oldest … newest; must have ≥ 3 entries.
    du_bar_dt_history:
        Same ordering for ∂ū/∂t arrays.
    forcing_history:
        Same ordering for forcing arrays.
    element_idx:
        Zero-based index of the *element* (not node); nodes are i = element_idx
        and i+1 = element_idx + 1.
    n_les_nodes:
        Total number of LES nodes (N_les).

    Returns
    -------
    input_vector : NDArray of shape (20,) or None if boundary padding required
        but element is at domain boundary.
    """
    if len(u_bar_history) < 3:  # need n, n-1, n-2
        return None

    u_bar_n = u_bar_history[-1]
    u_bar_nm1 = u_bar_history[-2]
    u_bar_nm2 = u_bar_history[-3]
    du_bar_dt_n = du_bar_dt_history[-1]
    forcing_n = forcing_history[-1]

    # Left boundary of element: node i = element_idx
    # Right boundary: node i+1 = element_idx + 1
    # Stencil nodes: i-2, i-1, i, i+1  (4 nodes)
    node_left = element_idx
    stencil_nodes = [node_left - 2, node_left - 1, node_left, node_left + 1]

    # Boundary check — skip elements whose stencil falls outside domain
    if stencil_nodes[0] < 0 or stencil_nodes[-1] >= n_les_nodes:
        return None

    stencil_indices = np.array(stencil_nodes)

    u_bar_n_stencil = u_bar_n[stencil_indices]  # (4,)
    u_bar_nm1_stencil = u_bar_nm1[stencil_indices]  # (4,)
    u_bar_nm2_stencil = u_bar_nm2[stencil_indices]  # (4,)
    du_bar_dt_stencil = du_bar_dt_n[stencil_indices]  # (4,)
    forcing_stencil = forcing_n[stencil_indices]  # (4,)

    input_vector = np.concatenate(
        [
            u_bar_n_stencil,  # 4
            u_bar_nm1_stencil,  # 4
            u_bar_nm2_stencil,  # 4
            du_bar_dt_stencil,  # 4
            forcing_stencil,  # 4
        ]
    )  # total: 20

    return input_vector


# ---------------------------------------------------------------------------
# Main training data assembly routine
# ---------------------------------------------------------------------------


def assemble_training_data(
    solutions_les: NDArray,
    solutions_dns_projected: NDArray,
    forcings_les: NDArray,
    dt: float,
    element_size: float,
) -> tuple[NDArray, NDArray, dict]:
    """Assemble and normalise (X, y) training pairs from projected LES / DNS data.

    Parameters
    ----------
    solutions_les:
        Filtered LES snapshots, shape (T, N_les).  These are ū at each time step.
    solutions_dns_projected:
        DNS snapshots projected onto the LES mesh, shape (T, N_les).
        Used to recover u' = u_dns_projected - ū.
    forcings_les:
        Box-filtered forcing on LES mesh, shape (T, N_les).
    dt:
        LES time step.
    element_size:
        Uniform element size h_les = L / (N_les - 1).

    Returns
    -------
    X_normalised : NDArray, shape (n_samples, 20)
        Normalised input feature matrix.
    y_normalised : NDArray, shape (n_samples, 5)
        Normalised output target matrix.
    normalisation_stats : dict
        Keys: X_mean, X_std, y_mean, y_std — needed at inference time.

    Notes
    -----
    Snapshots 0, 1 are skipped (need 3 time levels for lagged stencil).
    Boundary elements are skipped (stencil i-2…i+1 falls outside domain).
    """
    n_timesteps, n_les_nodes = solutions_les.shape
    n_elements = n_les_nodes - 1

    # Pre-compute u' and its derivatives at every time step
    u_prime_all: list[NDArray] = []
    du_prime_dt_all: list[NDArray] = []
    du_prime_dx_all: list[NDArray] = []
    du_bar_dt_all: list[NDArray] = []

    for time_idx in range(n_timesteps):
        u_bar_snapshot = solutions_les[time_idx]
        u_dns_proj_snapshot = solutions_dns_projected[time_idx]

        u_prime_snapshot = compute_u_prime_field(u_dns_proj_snapshot, u_bar_snapshot)
        u_prime_all.append(u_prime_snapshot)

        du_prime_dx_snapshot = compute_du_prime_dx(u_prime_snapshot, element_size)
        du_prime_dx_all.append(du_prime_dx_snapshot)

        u_prime_prev = u_prime_all[time_idx - 1] if time_idx > 0 else None
        du_prime_dt_snapshot = compute_du_prime_dt(u_prime_snapshot, u_prime_prev, dt)
        du_prime_dt_all.append(du_prime_dt_snapshot)

        u_bar_prev = solutions_les[time_idx - 1] if time_idx > 0 else None
        du_bar_dt_snapshot = (
            (u_bar_snapshot - u_bar_prev) / dt
            if u_bar_prev is not None
            else np.zeros_like(u_bar_snapshot)
        )
        du_bar_dt_all.append(du_bar_dt_snapshot)

    input_rows: list[NDArray] = []
    output_rows: list[NDArray] = []

    # Need at least 3 time levels (n, n-1, n-2) → start from index 2
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

        for elem_idx in range(n_elements):
            input_vec = build_input_stencil(
                u_bar_history=u_bar_history,
                du_bar_dt_history=du_bar_dt_history,
                forcing_history=forcing_history,
                element_idx=elem_idx,
                n_les_nodes=n_les_nodes,
            )
            if input_vec is None:
                continue  # skip boundary-adjacent elements

            node_left = elem_idx
            node_right = elem_idx + 1

            output_vec = compute_element_output_terms(
                u_bar_left=float(u_bar_cur[node_left]),
                u_bar_right=float(u_bar_cur[node_right]),
                u_prime_left=float(u_prime_cur[node_left]),
                u_prime_right=float(u_prime_cur[node_right]),
                du_bar_dt_left=float(du_bar_dt_all[time_idx][node_left]),
                du_bar_dt_right=float(du_bar_dt_all[time_idx][node_right]),
                du_prime_dt_left=float(du_prime_dt_cur[node_left]),
                du_prime_dt_right=float(du_prime_dt_cur[node_right]),
                du_prime_dx_left=float(du_prime_dx_cur[node_left]),
                du_prime_dx_right=float(du_prime_dx_cur[node_right]),
                element_size=element_size,
            )

            input_rows.append(input_vec)
            output_rows.append(output_vec)

    x_matrix = np.array(input_rows, dtype=np.float64)  # (n_samples, 20)
    y_matrix = np.array(output_rows, dtype=np.float64)  # (n_samples, 5)

    # Normalise to zero mean, unit std (per feature / per output)
    x_mean = x_matrix.mean(axis=0)
    x_std = x_matrix.std(axis=0)
    x_std[x_std < 1e-12] = 1.0  # guard against constant features

    y_mean = y_matrix.mean(axis=0)
    y_std = y_matrix.std(axis=0)
    y_std[y_std < 1e-12] = 1.0

    x_normalised = (x_matrix - x_mean) / x_std
    y_normalised = (y_matrix - y_mean) / y_std

    normalisation_stats = {
        "X_mean": x_mean,
        "X_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return x_normalised, y_normalised, normalisation_stats


# ---------------------------------------------------------------------------
# Train / validation split and saving
# ---------------------------------------------------------------------------


def split_and_save(
    x_data: NDArray,
    y_data: NDArray,
    normalisation_stats: dict,
    output_dir: Path,
    train_fraction: float = 0.8,
    random_seed: int = 42,
) -> None:
    """Shuffle, split, and save training data to *output_dir*.

    Saves:
        X_train.npy, y_train.npy
        X_val.npy,   y_val.npy
        normalisation_stats.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    n_samples = x_data.shape[0]
    shuffled_indices = rng.permutation(n_samples)

    split_index = int(train_fraction * n_samples)
    train_indices = shuffled_indices[:split_index]
    val_indices = shuffled_indices[split_index:]

    np.save(output_dir / "X_train.npy", x_data[train_indices])
    np.save(output_dir / "y_train.npy", y_data[train_indices])
    np.save(output_dir / "X_val.npy", x_data[val_indices])
    np.save(output_dir / "y_val.npy", y_data[val_indices])
    np.savez(output_dir / "normalisation_stats.npz", **normalisation_stats)

    # Save sequential (unshuffled) val slice for time-series and spatial plots
    np.save(
        output_dir / "X_val_sequential.npy",
        x_data[val_indices[np.argsort(val_indices)]],
    )
    np.save(
        output_dir / "y_val_sequential.npy",
        y_data[val_indices[np.argsort(val_indices)]],
    )

    print(
        f"Saved {len(train_indices)} training and {len(val_indices)} validation samples"
        f" to '{output_dir}'."
    )
    print(f"  Input shape  (X_train): {x_data[train_indices].shape}")
    print(f"  Output shape (y_train): {y_data[train_indices].shape}")


# ---------------------------------------------------------------------------
# Pipeline entry point (called from main pipeline after run_projection)
# ---------------------------------------------------------------------------


def run_training_data_assembly(
    projection_path: Path,
    output_dir: Path,
    dt: float,
    element_size: float,
    train_fraction: float = 0.8,
    random_seed: int = 42,
) -> tuple[NDArray, NDArray, dict]:
    """Load projected data, assemble (X, y), split, and save.

    Parameters
    ----------
    projection_path:
        Directory containing ``solutions_projection.npy`` and
        ``forcings_projection.npy`` (outputs of ``run_projection``).
    output_dir:
        Where to write the split training data.
    dt:
        LES time step size.
    element_size:
        Uniform LES element size h = L / (N_les - 1).

    Returns
    -------
    X_normalised, y_normalised, normalisation_stats
    """
    projection_path = Path(projection_path)

    solutions_les_raw = np.load(projection_path / "solutions_projection.npy")

    # Verify dt against saved timestamps if available
    times_path = projection_path / "times.npy"
    if times_path.exists():
        saved_times = np.load(times_path)
        dt_inferred = float(np.diff(saved_times).mean())
        if not np.isclose(dt, dt_inferred, rtol=1e-3):
            raise ValueError(
                f"Passed dt={dt:.6f} does not match inferred dt={dt_inferred:.6f} "
                "from saved timestamps. Check les_every_n_dns_steps."
            )

    # DNS-projected solution (ū + u') — for 2-scale decomposition u' = u_dns - ū
    # If the projection file only stores ū, we need a separate dns-on-les file.
    # Expected convention: projection stores ū; dns_on_les stores the full projected DNS.
    dns_on_les_path = projection_path / "dns_on_les.npy"
    if dns_on_les_path.exists():
        solutions_dns_projected = np.load(dns_on_les_path)
    else:
        raise FileNotFoundError(
            f"DNS-on-LES snapshot file not found at '{dns_on_les_path}'. "
            "The projection step must save the full (unfiltered) DNS interpolated "
            "onto the LES mesh in addition to the box-filtered ū."
        )

    forcings_path = projection_path / "forcings_projection.npy"
    if forcings_path.exists():
        forcings_les = np.load(forcings_path)
    else:
        n_timesteps, n_les_nodes = solutions_les_raw.shape
        forcings_les = np.zeros((n_timesteps, n_les_nodes))
        print("Warning: no forcing file found; using zeros.")

    x_data, y_data, norm_stats = assemble_training_data(
        solutions_les=solutions_les_raw,
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
