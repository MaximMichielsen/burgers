"""Project DNS data to a coarse LES grid and generate labeled ANN training data.

Feature set follows Rajampeta (2021) FS2 / Lagged Feature Set (LFS):

    Input  : { ū^{n, n-1, n-2}_{i-2:i+1} ; (∂ū/∂t)^n_{i-2:i+1} ; f^{n+1}_{i-1:i+2} }
    Output : IT^{n+1}  (four weak-form fine-scale forcing channels)

The *lagged* stencil uses time levels n, n-1, n-2 — never n+1.
This decouples the ANN from the Newton corrector passes and eliminates
Corrector Pass Instability (CPI) as described in Sections 3.5.3 and 5.3
of the thesis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from constants import DNS_TO_LES_RATIO, INPUT_STENCIL, OUTPUT_STENCIL, NORM_STATS
from functions import implicit_euler_first_order


# ---------------------------------------------------------------------------
# DNS I/O
# ---------------------------------------------------------------------------


def read_dns_data(
    directory: str | Path,
) -> tuple[NDArray, list[float], list[NDArray], list[NDArray]]:
    """Read chronologically sorted DNS snapshots from *directory*.

    Returns
    -------
    x : (N_dns,) spatial coordinate array
    times : sorted list of snapshot times
    solutions : list of solution arrays, each (N_dns,)
    forcings  : list of forcing arrays,  each (N_dns,)
    """
    directory = Path(directory)
    files = sorted(directory.glob("sol_t*.csv"))
    if not files:
        raise FileNotFoundError(f"No sol_t*.csv files found in {directory}")

    times, solutions, forcings = [], [], []
    x = None
    for file in files:
        time = float(file.stem.split("t")[-1])
        times.append(time)
        data = np.loadtxt(file, delimiter=",", skiprows=1)
        if x is None:
            x = data[:, 1]
        solutions.append(data[:, 2])
        try:
            forcings.append(data[:, 3])
        except IndexError:
            forcings.append(np.zeros_like(data[:, 2]))

    times, solutions = zip(*sorted(zip(times, solutions)))
    return x, list(times), list(solutions), list(forcings)


# ---------------------------------------------------------------------------
# Filtering / projection
# ---------------------------------------------------------------------------


def box_filter(solution: NDArray, ratio: int, n_les: int) -> tuple[NDArray, NDArray]:
    """Box-filter a DNS snapshot onto the coarse LES grid.

    Returns
    -------
    u_bar  : filtered velocity (N_les,)
    uu_bar : filtered velocity-squared (N_les,) — needed for τ_sgs
    """
    u_bar_full = uniform_filter(solution, size=ratio, mode="nearest")
    uu_bar_full = uniform_filter(solution**2, size=ratio, mode="nearest")
    indices = np.round(np.linspace(0, len(solution) - 1, n_les)).astype(int)
    return u_bar_full[indices], uu_bar_full[indices]


def compute_tau(u_bar: NDArray, uu_bar: NDArray, snapshot_index: int) -> NDArray:
    """SGS stress τ_sgs = uu_bar − u_bar², clamped to zero."""
    tau = uu_bar - u_bar**2
    if np.any(tau < -1e-10):
        raise ValueError(
            f"Negative τ_sgs at snapshot {snapshot_index}. "
            f"Min value: {tau.min():.3e}. Check the box-filter implementation."
        )
    return np.maximum(tau, 0.0)


def compute_du_bar_dt(
    u_bar_now: NDArray,
    u_bar_prev: NDArray | None,
    dt: float,
) -> NDArray:
    """Backward-Euler ∂ū/∂t.  Returns zeros for the first snapshot."""
    if u_bar_prev is None:
        return np.zeros_like(u_bar_now)
    return (u_bar_now - u_bar_prev) / dt


# ---------------------------------------------------------------------------
# Stencil builder
# ---------------------------------------------------------------------------


def extract_stencil(
    field: NDArray,
    mode: str,
    bc_values: tuple[float, float],
) -> NDArray:
    """Build a 4-point stencil [i-2, i-1, i, i+1] for every node.

    Parameters
    ----------
    field     : 1-D array (N,)
    mode      : ``"periodic"`` or ``"dirichlet"``
    bc_values : (left_bc, right_bc) used only for Dirichlet mode
    """
    if mode == "periodic":
        return np.stack(
            [np.roll(field, 2), np.roll(field, 1), field, np.roll(field, -1)],
            axis=1,
        )

    elif mode == "dirichlet":
        left, right = (
            (bc_values, bc_values) if not isinstance(bc_values, tuple) else bc_values
        )
        padded = np.concatenate([[left, left], field, [right]])
        N = len(field)
        return np.stack(
            [padded[0:N], padded[1 : N + 1], padded[2 : N + 2], padded[3 : N + 3]],
            axis=1,
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'periodic' or 'dirichlet'.")


# ---------------------------------------------------------------------------
# Feature / target assembly  —  LAGGED (LFS / FS2 style)
# ---------------------------------------------------------------------------


def build_features(
    solutions_les: list[NDArray],
    du_bar_dt_list: list[NDArray],
    forcing_list: list[NDArray],
    tau_list: list[NDArray],
    u_prime_t_list: list[NDArray],
    h_les: float,
    bc_mode: str,
    bc_values: tuple[float, float],
) -> tuple[NDArray, NDArray]:
    """Assemble the **lagged** (LFS) feature matrix.

    Input vector per node  — uses time levels n, n-1, n-2  (never n+1):

        X = { ū^n_{i-2:i+1} ; ū^{n-1}_{i-2:i+1} ; ū^{n-2}_{i-2:i+1} ;
              (∂ū/∂t)^n_{i-2:i+1} ; f^{n+1}_{i-2:i+1} }   → 20 columns

    Output vector per node  — four weak-form IT channels at n+1:

        y = { (w̄_x, ū²+0.5τ) ; (w̄_l, u'_t) ; (w̄_r, u'_t) ; (w̄_x, u'_x)·ν }

    The lagging (n instead of n+1 for the velocity stencils) means the ANN
    input is **frozen** before the Newton loop starts, so it cannot distort
    the ‖R_wi‖² space during corrector passes — this is the core mechanism
    of Rajampeta's LFS strategy (Section 5.3).

    Requires at least 4 snapshots (indices 0..3 give the first valid lagged
    triple n-2, n-1, n).

    Returns
    -------
    X_raw : ((T-3)*N_les, 20)
    y_raw : ((T-3)*N_les,  4)
    """
    if len(solutions_les) < 4:
        raise ValueError(
            f"build_features (LFS) requires at least 4 snapshots "
            f"(got {len(solutions_les)}).  Extend the simulation duration."
        )

    X_rows, y_rows = [], []

    # n starts at index 2 so we have n-2 (index 0) and n-1 (index 1) available.
    # The *target* is IT^{n+1}, indexed at n+1 = index n+1 in the list.
    # We therefore iterate n from 2 to len-2 so both n and n+1 are valid.
    for n in range(2, len(solutions_les) - 1):
        # ── Lagged inputs: use n, n-1, n-2  (NOT n+1) ──────────────────
        stencil_n = extract_stencil(solutions_les[n], mode=bc_mode, bc_values=bc_values)
        stencil_nm1 = extract_stencil(
            solutions_les[n - 1], mode=bc_mode, bc_values=bc_values
        )
        stencil_nm2 = extract_stencil(
            solutions_les[n - 2], mode=bc_mode, bc_values=bc_values
        )
        stencil_dudt = extract_stencil(
            du_bar_dt_list[n], mode=bc_mode, bc_values=bc_values
        )
        # Forcing at n+1 is fine — it is not updated by the Newton solver
        stencil_f = extract_stencil(
            forcing_list[n + 1], mode=bc_mode, bc_values=bc_values
        )

        features = np.hstack(
            [
                stencil_n,
                stencil_nm1,
                stencil_nm2,
                stencil_dudt,
                stencil_f,
            ]
        )  # (N_les, 20)

        X_rows.append(features)

        # ── Target: IT^{n+1} ────────────────────────────────────────────
        u_bar = solutions_les[n + 1]
        tau = tau_list[n + 1]
        u_prime_t = u_prime_t_list[n + 1]

        # channel 0 — convective flux: (w̄_x, ū² + 0.5τ)
        term_0 = implicit_euler_first_order(u_bar**2 + 0.5 * tau, h_les)
        # channel 1 — unsteady left:   (w̄_l, u'_t)
        term_1 = np.roll(u_prime_t, 1)
        # channel 2 — unsteady right:  (w̄_r, u'_t)
        term_2 = np.roll(u_prime_t, -1)
        # channel 3 — stress gradient: (w̄_x, u'_x)·ν  via τ proxy
        term_3 = implicit_euler_first_order(tau, h_les)

        y_rows.append(np.stack([term_0, term_1, term_2, term_3], axis=1))

    return np.vstack(X_rows), np.vstack(y_rows)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def compute_normalization_stats(
    input_array: NDArray, target_array: NDArray
) -> dict[str, NDArray]:
    """Z-score statistics from the full training set."""
    X_mean = input_array.mean(axis=0)
    X_std = input_array.std(axis=0)
    X_std[X_std == 0] = 1.0

    y_mean = target_array.mean(axis=0)
    y_std = target_array.std(axis=0)
    y_std[y_std == 0] = 1.0

    return {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}


def apply_normalization(
    input_array: NDArray, target_array: NDArray, stats: dict
) -> tuple[NDArray, NDArray]:
    """Apply pre-computed z-score normalisation."""
    X_norm = (input_array - stats["X_mean"]) / stats["X_std"]
    y_norm = (target_array - stats["y_mean"]) / stats["y_std"]
    return X_norm, y_norm


# ---------------------------------------------------------------------------
# Verification plot
# ---------------------------------------------------------------------------


def verify_global_projection(
    output_dir: str | Path,
    mesh_dns: NDArray,
    u_dns: NDArray,
    mesh_les: NDArray,
    u_projected: NDArray,
    n_dns: int | None = None,
    n_les: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    label_dns = f"DNS (N={n_dns})" if n_dns else "DNS"
    label_les = f"LES (N={n_les})" if n_les else "LES"

    ax.plot(mesh_dns, u_dns, label=label_dns, color="black", alpha=0.5)
    ax.plot(
        mesh_les,
        u_projected,
        "x--",
        label=label_les,
        color="orange",
        markersize=8,
        alpha=0.7,
    )
    ax.set_title("DNS vs. Coarse LES Projection (last snapshot)")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True, alpha=0.2)

    save_path = output_dir / "projection_verification.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved projection verification plot to '{save_path}'.")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_projection(
    directory: str | Path,
    bc_mode: str,
    bc_values: tuple[float, float],
    save: bool = True,
    output_dir: str | Path | None = None,
    verify: bool = True,
) -> tuple[NDArray, NDArray, dict, NDArray]:
    """Project DNS snapshots onto the LES grid and build LFS training data.

    Returns
    -------
    X        : normalised feature matrix  ((T-3)*N_les, 20)
    y        : normalised target matrix   ((T-3)*N_les,  4)
    stats    : normalisation dict
    u_les_last : last projected LES snapshot (for verification)
    """
    directory = Path(directory)
    output_dir = Path(output_dir)
    run_id = directory.name

    mesh_dns, times, solutions_dns, forcings_dns = read_dns_data(directory)

    N_les = len(mesh_dns) // DNS_TO_LES_RATIO
    les_indices = np.round(np.linspace(0, len(mesh_dns) - 1, N_les)).astype(int)
    mesh_les = mesh_dns[les_indices]
    h_les = float(abs(mesh_les[1] - mesh_les[0]))
    print(f"LES grid size: {N_les}")

    dt_array = np.diff(times)
    if not np.allclose(dt_array, dt_array[0], rtol=1):
        raise ValueError("Non-uniform time stepping in DNS data")
    dt = dt_array[0]

    solutions_les = []
    tau_list = []
    du_bar_dt_list = []
    forcing_list = []
    u_prime_t_list = []

    for i, (sol_dns, f_dns) in enumerate(zip(solutions_dns, forcings_dns)):
        u_bar, uu_bar = box_filter(sol_dns, ratio=DNS_TO_LES_RATIO, n_les=N_les)

        if bc_mode == "dirichlet":
            left, right = (
                (bc_values, bc_values)
                if not isinstance(bc_values, tuple)
                else bc_values
            )
            u_bar[0] = left
            u_bar[-1] = right
            uu_bar[0] = left**2
            uu_bar[-1] = right**2

        # Fine-scale time derivative (filtered DNS rate minus coarse-grid rate)
        if i > 0:
            du_dt_dns = (sol_dns - solutions_dns[i - 1]) / dt
            du_dt_bar = uniform_filter(
                du_dt_dns, size=DNS_TO_LES_RATIO, mode="nearest"
            )[les_indices]
        else:
            du_dt_bar = np.zeros_like(u_bar)

        current_du_bar_dt = compute_du_bar_dt(
            u_bar,
            u_bar_prev=solutions_les[-1] if solutions_les else None,
            dt=dt,
        )

        u_prime_t = du_dt_bar - current_du_bar_dt
        tau = compute_tau(u_bar, uu_bar, snapshot_index=i)
        f_bar, _ = box_filter(f_dns, ratio=DNS_TO_LES_RATIO, n_les=N_les)

        solutions_les.append(u_bar)
        tau_list.append(tau)
        du_bar_dt_list.append(current_du_bar_dt)
        forcing_list.append(f_bar)
        u_prime_t_list.append(u_prime_t)

    if verify:
        verify_global_projection(
            output_dir=output_dir,
            u_dns=solutions_dns[-1],
            u_projected=solutions_les[-1],
            mesh_dns=mesh_dns,
            mesh_les=mesh_les,
            n_dns=len(mesh_dns),
            n_les=N_les,
        )

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "solutions_projection.npy", np.array(solutions_les))
        print("Saved global LES projection snapshots.")

    # Build lagged features
    X_raw, y_raw = build_features(
        solutions_les=solutions_les,
        du_bar_dt_list=du_bar_dt_list,
        tau_list=tau_list,
        forcing_list=forcing_list,
        u_prime_t_list=u_prime_t_list,
        h_les=h_les,
        bc_mode=bc_mode,
        bc_values=bc_values,
    )

    stats = compute_normalization_stats(X_raw, y_raw)
    X, y = apply_normalization(X_raw, y_raw, stats)
    print(f"[{run_id}] Dataset shape — X: {X.shape}, y: {y.shape}")

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / INPUT_STENCIL, X)
        np.save(output_dir / OUTPUT_STENCIL, y)
        np.savez(output_dir / NORM_STATS, **stats)
        print(
            f"[{run_id}] Saved training data and normalisation stats to '{output_dir}'."
        )

    return X, y, stats, solutions_les[-1]
