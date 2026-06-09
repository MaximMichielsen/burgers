"""DNS-to-LES projection: nodal or L2 based."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from problems_and_configurations.disc_config import DiscretisationConfig
from utils.io_utils import read_data

_VALID_PROJECTION_MODES: frozenset[str] = frozenset({"l2", "L2", "nodal"})


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


def nodal_project(
    solution_dns: NDArray,
    mesh_dns: NDArray,
    mesh_les: NDArray,
) -> tuple[NDArray, NDArray]:
    """Nodal projection of a DNS snapshot onto the LES mesh.

    Evaluates u_DNS at each LES node via linear interpolation.
    Returns (u_bar, uu_bar) to match the l2_project interface.

    Note: u' = 0 at nodes by construction, which makes the viscous
    interaction term (w_x, ν u'_x) unlearnable. Use L2 projection
    for SGSP training data. Nodal projection is provided for comparison.
    """
    u_bar = np.interp(mesh_les, mesh_dns, solution_dns)
    uu_bar = np.interp(mesh_les, mesh_dns, solution_dns**2)
    return u_bar, uu_bar


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


def run_projection(
    dns_directory: str | Path,
    output_directory: str | Path,
    bc_mode: str,
    bc_values: float | int | tuple[float, float] | None,
    disc_cfg: DiscretisationConfig,
    verify: bool = True,
    projection_mode: str = "l2",
) -> None:
    """Project DNS snapshots onto the LES grid and save filtered arrays."""
    if projection_mode not in _VALID_PROJECTION_MODES:
        raise ValueError(
            f"Choose mode for projection, options: {_VALID_PROJECTION_MODES}"
        )

    _project = l2_project if projection_mode in ("l2", "L2") else nodal_project

    dns_directory = Path(dns_directory)
    output_directory = Path(output_directory)

    mesh_dns, times, solutions_dns, forcings_dns = read_data(dns_directory)
    mesh_les = disc_cfg.mesh_les
    n_les = disc_cfg.n_nodes_les

    les_snapshot_indices = np.arange(
        0, len(solutions_dns), disc_cfg.temporal_refinement
    )

    solutions_proj, forcing_list, dns_on_les_list = [], [], []
    enforce_bcs = bc_mode in ("dirichlet", "fixed")

    for i, (solution_dns, forcing_dns) in enumerate(zip(solutions_dns, forcings_dns)):
        u_bar, uu_bar = _project(solution_dns, mesh_dns=mesh_dns, mesh_les=mesh_les)

        if enforce_bcs and bc_values is not None:
            u_bar, uu_bar = _enforce_dirichlet_bcs(u_bar, uu_bar, bc_values)

        f_bar, _ = _project(forcing_dns, mesh_dns=mesh_dns, mesh_les=mesh_les)
        forcing_list.append(f_bar)
        solutions_proj.append(u_bar)
        dns_on_les_list.append(np.interp(mesh_les, mesh_dns, solution_dns))

    if verify:
        verify_global_projection(
            output_dir=output_directory,
            u_dns=solutions_dns[-1],
            u_projected=solutions_proj[-1],
            mesh_dns=mesh_dns,
            mesh_les=mesh_les,
            n_dns=disc_cfg.n_nodes_dns,
            n_les=n_les,
            mode=projection_mode,
        )

    if les_snapshot_indices is None:
        les_snapshot_indices = np.arange(len(solutions_proj))

    np.save(
        output_directory / "solutions_projection.npy",
        np.array(solutions_proj)[les_snapshot_indices],
    )
    np.save(
        output_directory / "forcings_projection.npy",
        np.array(forcing_list)[les_snapshot_indices],
    )
    np.save(output_directory / "times.npy", np.array(times)[les_snapshot_indices])
    np.save(
        output_directory / "solutions_dns_raw.npy",
        np.array(solutions_dns)[les_snapshot_indices],  # shape (T, n_dns)
    )
    np.save(
        output_directory / "mesh_dns.npy",
        mesh_dns,  # shape (n_dns,) — needed by assembly
    )


def verify_global_projection(
    output_dir: str | Path,
    mesh_dns: NDArray,
    u_dns: NDArray,
    mesh_les: NDArray,
    u_projected: NDArray,
    mode: str,
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
        color="lightgreen",
        markersize=8,
    )
    ax.plot(mesh_les, u_projected, "--", color="orange", alpha=0.7)
    ax.set_title(f"DNS vs. Coarse LES Projection ({mode})")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True, alpha=0.2)

    save_path = output_dir / f"projection_{mode}_verification.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved projection verification plot to '{save_path}'.")
    plt.close(fig)
