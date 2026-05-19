from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from constants import DNS_TO_LES_RATIO


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


def compute_tau(u_bar: NDArray, uu_bar: NDArray, snapshot_index: int) -> NDArray:
    """Compute the SGS stress τ_sgs = uu_bar - u_bar², clamped to zero."""
    tau = uu_bar - u_bar**2
    if np.any(tau < -1e-10):  # only raise on meaningfully negative values
        raise ValueError(
            f"Negative τ_sgs at snapshot {snapshot_index}. "
            f"Min value: {tau.min():.3e}. Check the box-filter implementation."
        )
    return np.maximum(tau, 0.0)


def run_projection(
    directory: str | Path,
    bc_mode: str,
    bc_values: tuple[float, float],
    save: bool = True,
    output_dir: str | Path | None = None,
    verify: bool = True,
) -> None:
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
