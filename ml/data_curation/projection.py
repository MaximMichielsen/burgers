"""DNS-to-LES projection: box filtering, τ_sgs computation, and snapshot saving."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from constants import DNS_TO_LES_RATIO
from utils.io_utils import read_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enforce_dirichlet_bcs(
    u_bar: NDArray,
    uu_bar: NDArray,
    bc_values: float | tuple[float, float],
) -> tuple[NDArray, NDArray]:
    """Enforce Dirichlet BCs on filtered fields in-place."""
    left, right = (
        (bc_values, bc_values) if not isinstance(bc_values, tuple) else bc_values
    )
    u_bar[0], u_bar[-1] = left, right
    uu_bar[0], uu_bar[-1] = left**2, right**2
    return u_bar, uu_bar


def box_filter(solution: NDArray, ratio: int, n_les: int) -> tuple[NDArray, NDArray]:
    """Box-filter and downsample a DNS snapshot to the LES grid.

    Returns (u_bar, uu_bar) — filtered velocity and filtered velocity squared.
    """
    indices = np.round(np.linspace(0, len(solution) - 1, n_les)).astype(int)
    u_bar_full = uniform_filter(solution, size=ratio, mode="nearest")
    uu_bar_full = uniform_filter(solution**2, size=ratio, mode="nearest")
    return u_bar_full[indices], uu_bar_full[indices]


def compute_du_bar_dt(
    u_bar_now: NDArray,
    u_bar_prev: NDArray | None,
    dt: float,
) -> NDArray:
    """Backward-Euler time derivative of the filtered velocity; zero at first step."""
    if u_bar_prev is None:
        return np.zeros_like(u_bar_now)
    return (u_bar_now - u_bar_prev) / dt


def compute_tau(u_bar: NDArray, uu_bar: NDArray, snapshot_index: int) -> NDArray:
    """SGS stress τ_sgs = uu_bar - u_bar², clamped to zero."""
    tau = uu_bar - u_bar**2
    if np.any(tau < -1e-10):
        raise ValueError(
            f"Negative τ_sgs at snapshot {snapshot_index}. "
            f"Min value: {tau.min():.3e}. Check the box-filter implementation."
        )
    return np.maximum(tau, 0.0)


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
    """Plot DNS vs projected LES solution and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        mesh_dns,
        u_dns,
        label=f"DNS (N={n_dns})" if n_dns else "DNS",
        color="black",
        alpha=0.5,
    )
    ax.plot(
        mesh_les,
        u_projected,
        "x",
        label=f"LES (N={n_les})" if n_les else "LES",
        color="orange",
        markersize=8,
    )
    ax.plot(mesh_les, u_projected, "--", color="orange", alpha=0.7)
    ax.set_title("DNS vs. Coarse LES Projection (last t)")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True, alpha=0.2)

    save_path = output_dir / "projection_verification.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved projection verification plot to '{save_path}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main projection
# ---------------------------------------------------------------------------


def run_projection(
    directory: str | Path,
    bc_mode: str,
    bc_values: float | int | tuple[float | int, float | int] | None,
    output_dir: str | Path | None = None,
    verify: bool = True,
    les_snapshot_indices: np.ndarray | None = None,
    n_nodes_les: int | None = None,
) -> None:
    """Project DNS snapshots onto the LES grid and save filtered arrays."""
    directory = Path(directory)
    output_dir = Path(output_dir)

    mesh_dns, times, solutions_dns, forcings_dns = read_data(directory)

    n_les = (
        n_nodes_les
        if n_nodes_les is not None
        else (len(mesh_dns) - 1) // DNS_TO_LES_RATIO + 1
    )
    les_indices = np.round(np.linspace(0, len(mesh_dns) - 1, n_les)).astype(int)
    mesh_les = mesh_dns[les_indices]

    print(f"LES grid size: {n_les}")

    dt_array = np.diff(times)
    if not np.allclose(dt_array, dt_array[0], rtol=1):
        raise ValueError("Non-uniform time stepping in DNS data.")
    dt = float(dt_array[0])

    enforce_bcs = bc_mode in ("dirichlet", "fixed")

    solutions_les, tau_list, du_bar_dt_list = [], [], []
    forcing_list, u_prime_t_list, dns_on_les_list = [], [], []

    for i, (solution_dns, forcing_dns) in enumerate(zip(solutions_dns, forcings_dns)):
        u_bar, uu_bar = box_filter(solution_dns, ratio=DNS_TO_LES_RATIO, n_les=n_les)

        if enforce_bcs and bc_values is not None:
            u_bar, uu_bar = _enforce_dirichlet_bcs(u_bar, uu_bar, bc_values)

        du_dt_bar = (
            uniform_filter(
                (solution_dns - solutions_dns[i - 1]) / dt,
                size=DNS_TO_LES_RATIO,
                mode="nearest",
            )[les_indices]
            if i > 0
            else np.zeros_like(u_bar)
        )

        current_du_bar_dt = compute_du_bar_dt(
            u_bar,
            u_bar_prev=solutions_les[-1] if solutions_les else None,
            dt=dt,
        )

        u_prime_t_list.append(du_dt_bar - current_du_bar_dt)
        tau_list.append(compute_tau(u_bar, uu_bar, snapshot_index=i))
        du_bar_dt_list.append(current_du_bar_dt)
        f_bar, _ = box_filter(forcing_dns, ratio=DNS_TO_LES_RATIO, n_les=n_les)
        forcing_list.append(f_bar)
        solutions_les.append(u_bar)
        dns_on_les_list.append(solution_dns[les_indices])

    if verify:
        verify_global_projection(
            output_dir=output_dir,
            u_dns=solutions_dns[-1],
            u_projected=solutions_les[-1],
            mesh_dns=mesh_dns,
            mesh_les=mesh_les,
            n_dns=len(mesh_dns),
            n_les=n_les,
        )

    if les_snapshot_indices is None:
        les_snapshot_indices = np.arange(len(solutions_les))

    np.save(
        output_dir / "solutions_projection.npy",
        np.array(solutions_les)[les_snapshot_indices],
    )
    np.save(
        output_dir / "dns_on_les.npy", np.array(dns_on_les_list)[les_snapshot_indices]
    )
    np.save(
        output_dir / "forcings_projection.npy",
        np.array(forcing_list)[les_snapshot_indices],
    )
    np.save(output_dir / "times.npy", np.array(times)[les_snapshot_indices])
