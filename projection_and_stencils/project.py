"""Project DNS data to a coarse LES grid and generate labeled ANN training data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from constants import DNS_TO_LES_RATIO, INPUT_STENCIL, OUTPUT_STENCIL, NORM_STATS
from functions import implicit_euler_first_order


def read_dns_data(
    directory: str | Path,
) -> tuple[NDArray, list[float], list[NDArray], list[NDArray]]:
    """Read chronologically sorted DNS snapshots from *directory*.

    Returns:
    -------
    x:
        Spatial coordinate array (N_dns,).
    times:
        Sorted list of snapshot times.
    solutions:
        Corresponding list of solution arrays, each of shape (N_dns,).
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

        # PLACEHOLDER: Assuming column 3 will be forcing
        # For now, we create a dummy array of zeros matching the DNS shape
        try:
            forcings.append(data[:, 3])
        except IndexError:
            forcings.append(np.zeros_like(data[:, 2]))

    times, solutions = zip(*sorted(zip(times, solutions)))
    return x, list(times), list(solutions), list(forcings)


def box_filter(solution: NDArray, ratio: int, n_les: int) -> tuple[NDArray, NDArray]:
    """Apply a box filter and downsample a DNS snapshot to the LES grid.

    Returns:
    -------
    u_bar:
        Filtered, coarse-grained velocity (N_les,).
    uu_bar:
        Filtered, coarse-grained velocity squared (N_les,) — needed for τ_sgs.
    """
    u_bar_full = uniform_filter(solution, size=ratio, mode="nearest")
    uu_bar_full = uniform_filter(solution**2, size=ratio, mode="nearest")
    indices = np.round(np.linspace(0, len(solution) - 1, n_les)).astype(int)
    return u_bar_full[indices], uu_bar_full[indices]


def compute_tau(u_bar: NDArray, uu_bar: NDArray, snapshot_index: int) -> NDArray:
    """Compute the SGS stress τ_sgs = uu_bar - u_bar², clamped to zero."""
    tau = uu_bar - u_bar**2
    if np.any(tau < -1e-10):  # only raise on meaningfully negative values
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
    """Backward-Euler time derivative of the filtered velocity.

    Returns a zero array for the first snapshot (no previous state available).
    """
    if u_bar_prev is None:
        return np.zeros_like(u_bar_now)
    return (u_bar_now - u_bar_prev) / dt


def extract_stencil(
    field: NDArray,
    mode: str,
    bc_values: tuple[float, float],
) -> NDArray:
    """Build a 4-point stencil [i-2, i-1, i, i+1] for every node.

    Parameters
    ----------
    field : NDArray
        1-D field of shape (N,).
    mode : str
        ``"periodic"``  – wrap edges with np.roll (default, for periodic BCs).
        ``"fixed"``     – pad edges with known Dirichlet boundary values.
    bc_values : tuple[float, float]
        (left_bc, right_bc) used only when mode == "fixed".
        E.g. (1.0, 1.0) if u=1 is enforced at both walls.
    """
    if mode == "periodic":
        return np.stack(
            [
                np.roll(field, 2),  # i-2
                np.roll(field, 1),  # i-1
                field,  # i
                np.roll(field, -1),  # i+1
            ],
            axis=1,
        )

    elif mode == "dirichlet":
        left, right = (
            bc_values if isinstance(bc_values, tuple) else (bc_values, bc_values)
        )
        # Pad: [left, left, f0, f1, ..., f_{N-1}, right]
        # so we have 2 ghost cells on the left, 1 on the right
        padded = np.concatenate([[left, left], field, [right]])
        N = len(field)
        return np.stack(
            [
                padded[0:N],  # i-2
                padded[1 : N + 1],  # i-1
                padded[2 : N + 2],  # i   (= field itself)
                padded[3 : N + 3],  # i+1
            ],
            axis=1,
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'periodic' or 'dirichlet'.")


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
    """Assemble the raw (un-normalized) feature matrix.

    Input vector per node:

        input = { ū^{n, n-1, n-2}_{i-2:i+1} ; (∂ū/∂t)^n_{i-2:i+1} ; f^n_{i-2:i+1} }
        output = { (w_x, u_bar^2 + 0.5 * u'_u') ;  (w_l, u'_t) ;  (w_r, u'_t) ;  (w_x, u'_x) }

    Returns:
    -------
    X_raw:
        Array of shape ((T-2)*N_les, 20).
    y_raw:
        Array of shape ((T-2)*N_les, 4).
    """
    if len(solutions_les) < 4:
        raise ValueError(
            f"build_features requires at least 4 snapshots (got {len(solutions_les)}). "
            f"Increase 'extract_at_times' or extend the simulation duration."
        )

    X_rows, y_rows = [], []

    for n in range(2, len(solutions_les)):
        features = np.hstack(
            [
                extract_stencil(solutions_les[n], mode=bc_mode, bc_values=bc_values),
                extract_stencil(
                    solutions_les[n - 1], mode=bc_mode, bc_values=bc_values
                ),
                extract_stencil(
                    solutions_les[n - 2], mode=bc_mode, bc_values=bc_values
                ),
                extract_stencil(du_bar_dt_list[n], mode=bc_mode, bc_values=bc_values),
                extract_stencil(forcing_list[n], mode=bc_mode, bc_values=bc_values),
            ]
        )

        X_rows.append(features)

        # --- Output Target Matrix (y) ---
        u_bar = solutions_les[n]
        tau = tau_list[n]
        u_prime_t = u_prime_t_list[n]

        # 1. Convective term: (w_x, ū² + 0.5 * τ)
        convective_flux = u_bar**2 + 0.5 * tau
        term_1 = implicit_euler_first_order(convective_flux, h_les)

        # 2 & 3. Unsteady fluctuation terms: (w_l, u'_t) and (w_r, u'_t)
        term_2 = np.roll(u_prime_t, 1)  # w_l (left weighting)
        term_3 = np.roll(u_prime_t, -1)  # w_r (right weighting)

        # 4. Stress gradient term: (w_x, u'_x)
        term_4 = implicit_euler_first_order(tau, h_les)

        y_rows.append(np.stack([term_1, term_2, term_3, term_4], axis=1))

    return np.vstack(X_rows), np.vstack(y_rows)


def compute_normalization_stats(
    input_array: NDArray, target_array: NDArray
) -> dict[str, float]:
    """Compute per-feature mean and standard deviation for z-score normalization.

    Statistics are derived from the full training set and must be saved
    alongside the data so that the *same* transform can be applied at
    inference time without data leakage.

    Returns:
    -------
    dict with keys ``X_mean``, ``X_std``, ``y_mean``, ``y_std``.
    """
    X_mean = input_array.mean(axis=0)
    X_std = input_array.std(axis=0)
    X_std[X_std == 0] = 1.0  # avoid division by zero for constant features

    y_mean = target_array.mean(axis=0)
    y_std = target_array.std(axis=0)
    y_std[y_std == 0] = 1.0

    return {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}


def apply_normalization(
    input_array: NDArray, target_array: NDArray, stats: dict[str, float]
) -> tuple[NDArray, NDArray]:
    """Z-score normalize features and target using pre-computed *stats*."""
    X_norm = (input_array - stats["X_mean"]) / stats["X_std"]
    y_norm = (target_array - stats["y_mean"]) / stats["y_std"]
    return X_norm, y_norm


def verify_global_projection(
    output_dir: str | Path,
    mesh_dns: NDArray,
    u_dns: NDArray,
    mesh_les: NDArray,
    u_projected: NDArray,
    n_dns: int | None = None,
    n_les: int | None = None,
) -> None:
    """Verify projection using markers to visualize the actual LES grid nodes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    label_dns = f"DNS (N: {n_dns})" if n_dns is not None else "DNS"
    label_les = f"LES (N: {n_les})" if n_les is not None else "LES"

    ax.plot(mesh_dns, u_dns, label=label_dns, color="black", alpha=0.5)
    ax.plot(mesh_les, u_projected, "x", label=label_les, color="orange", markersize=8)
    ax.plot(mesh_les, u_projected, "--", color="orange", alpha=0.7)

    ax.set_title("DNS vs. Coarse LES Projection (last t)")
    ax.set_xlabel("Coordinate (x)")
    ax.set_ylabel("Velocity (u)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    save_path = output_dir / "projection_verification.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved projection verification plot to '{save_path}'.")
    plt.show()
    plt.close(fig)


def run_projection(
    directory: str | Path,
    bc_mode: str,
    bc_values: tuple[float, float],
    save: bool = True,
    output_dir: str | Path | None = None,
    verify: bool = True,
) -> tuple[NDArray, NDArray, dict[str, float], NDArray]:
    """Project DNS data onto the LES grid and build ANN training data.

    Returns:
    -------
    X:
        Normalised feature matrix, shape ``((T-2)*N_les, 20)``.
    y:
        Normalised target vector, shape ``((T-2)*N_les,)``.
    stats:
        Normalisation statistics dict (``X_mean``, ``X_std``, ``y_mean``, ``y_std``).
    """
    directory = Path(directory)
    run_id = directory.name  # e.g. "run_DNS_0423_174914"
    output_dir = Path(output_dir)

    # --- read DNS snapshots ---------------------------------------------------
    mesh_dns, times, solutions_dns, forcings_dns = read_dns_data(directory)

    N_les = len(mesh_dns) // DNS_TO_LES_RATIO
    les_indices = np.round(np.linspace(0, len(mesh_dns) - 1, N_les)).astype(int)
    mesh_les = mesh_dns[les_indices]
    h_les = float(abs(mesh_les[1] - mesh_les[0]))

    print(f"LES grid size: {len(mesh_les)}")

    dt_array = np.diff(times)
    if not np.allclose(dt_array, dt_array[0], rtol=1):
        raise ValueError("Non-uniform time stepping in DNS data")

    dt = dt_array[0]

    # --- project each DNS snapshot onto LES grid -----------------------------
    solutions_les = []
    tau_list = []
    du_bar_dt_list = []
    forcing_list = []
    u_prime_t_list = []

    for i, (solution_dns, forcing_dns) in enumerate(zip(solutions_dns, forcings_dns)):
        u_bar, uu_bar = box_filter(solution_dns, ratio=DNS_TO_LES_RATIO, n_les=N_les)

        # Enforce Dirichlet BCs on the filtered field before anything else
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

        if i > 0:
            du_dt_dns = (solution_dns - solutions_dns[i - 1]) / dt
            du_dt_bar = uniform_filter(
                du_dt_dns, size=DNS_TO_LES_RATIO, mode="nearest"
            )[les_indices]

        else:
            du_dt_bar = np.zeros_like(u_bar)

        current_du_bar_dt = compute_du_bar_dt(
            u_bar, u_bar_prev=solutions_les[-1] if solutions_les else None, dt=dt
        )

        u_prime_t = du_dt_bar - current_du_bar_dt
        u_prime_t_list.append(u_prime_t)

        tau = compute_tau(u_bar, uu_bar, snapshot_index=i)
        du_bar_dt = compute_du_bar_dt(
            u_bar,
            u_bar_prev=solutions_les[-1] if solutions_les else None,
            dt=dt,
        )
        f_bar, _ = box_filter(forcing_dns, ratio=DNS_TO_LES_RATIO, n_les=N_les)

        solutions_les.append(u_bar)
        tau_list.append(tau)
        du_bar_dt_list.append(du_bar_dt)
        forcing_list.append(f_bar)

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
        print("Saved global LES projection snapshots for verification.")

    # --- assemble & normalise ------------------------------------------------
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

    print(f"[{run_id}] Dataset shape  — X: {X.shape}, y: {y.shape}")

    # --- persist to disk -----------------------------------------------------
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / INPUT_STENCIL, X)
        np.save(output_dir / OUTPUT_STENCIL, y)
        np.savez(output_dir / NORM_STATS, **stats)
        print(
            f"[{run_id}] Saved training data and normalisation stats to '{output_dir}'."
        )

    return X, y, stats, solutions_les[-1]
