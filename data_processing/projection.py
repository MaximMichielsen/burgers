from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

# Paths Setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

PARSED_DATA_DIR = PROJECT_ROOT / "dns_data" / "curated"
DAT_PARSED_DIR = PARSED_DATA_DIR / "dat"
STAT_PARSED_DIR = PARSED_DATA_DIR / "stat"

FORCING_PATH = DAT_PARSED_DIR / "f.npy"
W_FIELD_PATH = DAT_PARSED_DIR / "w.npy"

FILTERED_PATH = PROJECT_ROOT / "dns_data" / "projected"


def filter_forcing(
    forcing_data: NDArray,
    les_resolution: int,
    mode: str = "l2",
    save_path: Path | None = None,
    z_min: float = 0.0,
    z_max: float = 2.0,
) -> NDArray:
    n_timesteps, dns_res = forcing_data.shape
    z_dns = np.linspace(z_min, z_max, dns_res)
    z_les = np.linspace(z_min, z_max, les_resolution)

    length = z_max - z_min
    h_les = length / (les_resolution - 1)
    dz_dns = length / (dns_res - 1)

    # 1. Assemble Matrix M
    diag_main_m = np.full(les_resolution, 2.0 * h_les / 3.0)
    diag_main_m[0] /= 2.0
    diag_main_m[-1] /= 2.0
    diag_off_m = np.full(les_resolution - 1, h_les / 6.0)

    if mode.lower() == "l2":
        m = diags([diag_off_m, diag_main_m, diag_off_m], [-1, 0, 1], format="csc")

    elif mode.lower() in ("h1", "h10"):
        # Stiffness matrix (scaled appropriately)
        diag_main_k = np.full(les_resolution, 2.0 / h_les) * (h_les**2)
        diag_main_k[0] /= 2.0
        diag_main_k[-1] /= 2.0
        diag_off_k = np.full(les_resolution - 1, -1.0 / h_les) * (h_les**2)

        m = diags(
            [
                diag_off_m + diag_off_k,
                diag_main_m + diag_main_k,
                diag_off_m + diag_off_k,
            ],
            [-1, 0, 1],
            format="csc",
        )
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'l2' or 'h1'.")

    # 2. Map DNS points to LES elements
    elem_idx = np.clip(
        np.floor((z_dns - z_min) / h_les).astype(int), 0, les_resolution - 2
    )
    xi = (z_dns - z_les[elem_idx]) / h_les

    n_left = 1.0 - xi
    n_right = xi

    weights = np.full(dns_res, dz_dns)
    weights[0] /= 2.0
    weights[-1] /= 2.0

    # Assemble b_L2
    b = np.zeros((n_timesteps, les_resolution))
    weighted_forcing = forcing_data * weights

    for t in range(n_timesteps):
        np.add.at(b[t], elem_idx, weighted_forcing[t] * n_left)
        np.add.at(b[t], elem_idx + 1, weighted_forcing[t] * n_right)

    # 3. Add Gradient Contribution b_grad for H1 mode
    if mode.lower() in ("h1", "h10"):
        # Compute central spatial derivative df/dz on DNS grid
        df_dz_dns = np.gradient(forcing_data, dz_dns, axis=1)
        weighted_df = df_dz_dns * weights

        # Shape function gradients: dN_left/dz = -1/h_les, dN_right/dz = 1/h_les
        # Scale by h_les^2 to match the scaling of K in the matrix
        d_n_left_dz = -1.0 / h_les * (h_les**2)
        d_n_right_dz = 1.0 / h_les * (h_les**2)

        for t in range(n_timesteps):
            np.add.at(b[t], elem_idx, weighted_df[t] * d_n_left_dz)
            np.add.at(b[t], elem_idx + 1, weighted_df[t] * d_n_right_dz)

    # 4. Solve System
    solve_m = factorized(m)
    filtered_forcing = np.zeros((n_timesteps, les_resolution))
    for t in range(n_timesteps):
        filtered_forcing[t] = solve_m(b[t])

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / f"forcing_{mode.lower()}_{les_resolution}"
        np.save(save_file, filtered_forcing)

    return filtered_forcing


def filter_ic(ic_dns: NDArray, mesh_les: NDArray, mesh_dns: NDArray) -> NDArray:
    """Projects the DNS initial condition onto the LES mesh via linear interpolation."""
    return np.interp(mesh_les, mesh_dns, ic_dns)


def nodal_project(
    fields_dns: NDArray,
    mesh_les: NDArray,
    mesh_dns: NDArray,
) -> NDArray:
    """Project an array DNS fields onto the LES grid using 1D nodal interpolation."""
    return np.asarray([np.interp(mesh_les, mesh_dns, field) for field in fields_dns])


if __name__ == "__main__":
    n_nodes_les = [9, 17, 33, 65]
    forcing_filter = "l2"
    if FORCING_PATH.exists():
        forcing_dns = np.load(FORCING_PATH)
        for n_nodes in n_nodes_les:
            l2_data = filter_forcing(
                forcing_dns,
                mode=forcing_filter,
                les_resolution=n_nodes,
                save_path=FILTERED_PATH,
                z_min=0.0,
                z_max=2.0,
            )
            print(
                f"Forcing filtering complete ({n_nodes}). Outputs saved to {FILTERED_PATH}"
            )
    else:
        print(f"Forcing file not found at {FORCING_PATH}")

    if W_FIELD_PATH.exists():
        w_dns = np.load(W_FIELD_PATH)
        mesh_dns_ = np.linspace(0.0, 2.0, 513)

        for nodes in n_nodes_les:
            mesh_les_ = np.linspace(0.0, 2.0, nodes)

            w_projected = nodal_project(w_dns, mesh_les_, mesh_dns_)
            ic_projected = w_projected[0]

            np.save(FILTERED_PATH / f"w_linear_{nodes}.npy", w_projected)
            np.save(FILTERED_PATH / f"ic_linear_{nodes}.npy", ic_projected)
            print(f"W filtering complete ({nodes}). Outputs saved to {FILTERED_PATH}")
    else:
        print("no w field")
