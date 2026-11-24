from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray


def compute_gauss_strong_residual_snapshots(
    z_les: NDArray,
    w_snapshots: NDArray,
    f_snapshots: NDArray,
    dt: float,
) -> NDArray:
    """Computes the strong residual at 3 Gauss points per element and projects
    them onto the global nodes via L2 projection.
    """
    n_timesteps, n_nodes = w_snapshots.shape
    n_elements = n_nodes - 1  # 1D linear elements

    # 3-Point Gauss-Legendre Quadrature Rules on reference element [-1, 1]
    gp_coords_ref = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
    gp_weights = np.array([5 / 9, 8 / 9, 5 / 9])

    r_nodal_snapshots = np.zeros_like(w_snapshots)

    # Build Global Mass Matrix M once
    M_global = np.zeros((n_nodes, n_nodes))
    element_nodes_list = []

    for e in range(n_elements):
        node1, node2 = e, e + 1
        element_nodes_list.append((node1, node2))
        h_e = z_les[node2] - z_les[node1]
        m_elem = (h_e / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        M_global[node1 : node2 + 1, node1 : node2 + 1] += m_elem

    dz = z_les[1] - z_les[0]

    for t in range(n_timesteps):
        w = w_snapshots[t]
        f = f_snapshots[t] if f_snapshots is not None else np.zeros_like(w)

        # 1. Temporal derivative (BDF1 for step 1, BDF2 thereafter)
        if t == 0:
            w_t = np.zeros_like(w)
        elif t == 1:
            w_t = (w_snapshots[1] - w_snapshots[0]) / dt
        else:
            w_t = (3 * w_snapshots[t] - 4 * w_snapshots[t - 1] + w_snapshots[t - 2]) / (
                2 * dt
            )

        # Nodal convective spatial derivative
        w_z = np.zeros_like(w)
        w_z[1:-1] = (w[2:] - w[:-2]) / (2 * dz)
        w_z[0] = (w[1] - w[0]) / dz
        w_z[-1] = (w[-1] - w[-2]) / dz

        F_proj_global = np.zeros(n_nodes)

        # 2. Loop over elements to evaluate residual at 3 Gauss points and integrate
        for e, (node1, node2) in enumerate(element_nodes_list):
            h_e = z_les[node2] - z_les[node1]
            detJ = h_e / 2.0
            f_elem_local = np.zeros(2)

            for gp_idx, xi in enumerate(gp_coords_ref):
                wgt = gp_weights[gp_idx]
                N = np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])

                # Interpolate fields at Gauss point
                w_gp = w[node1] * N[0] + w[node2] * N[1]
                wt_gp = w_t[node1] * N[0] + w_t[node2] * N[1]

                N_z = np.array([-1.0 / h_e, 1.0 / h_e])
                wz_gp = w[node1] * N_z[0] + w[node2] * N_z[1]
                f_gp = f[node1] * N[0] + f[node2] * N[1]

                # Strong residual at Gauss point: R = w_t + w * w_z - f
                R_gp = wt_gp + (w_gp * wz_gp) - f_gp

                for i in range(2):
                    f_elem_local[i] += N[i] * R_gp * detJ * wgt

            F_proj_global[node1 : node2 + 1] += f_elem_local

        r_nodal_snapshots[t] = np.linalg.solve(M_global, F_proj_global)

    return r_nodal_snapshots


def animate_les_snapshots(
    z_les: NDArray,
    w_snapshots: NDArray,
    f_snapshots: NDArray,
    residual_snapshots: NDArray,
    interval: int = 100,
    save_path: Optional[Path] = None,
) -> FuncAnimation:
    """Animates velocity, forcing, and residual with an interactive slider

    and a Play/Pause button.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), dpi=120, sharex=True)
    fig.subplots_adjust(bottom=0.2)  # Extra room for widgets

    # --- Top Subplot: Velocity & Forcing ---
    (line_w,) = ax1.plot(
        z_les,
        w_snapshots[0],
        "o-",
        color="royalblue",
        lw=1.5,
        markersize=4,
        label="Velocity $w(z, t)$",
    )
    ax1.set_ylabel("Velocity $w(z, t)$", color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.grid(True, linestyle="--", alpha=0.6)

    w_min, w_max = w_snapshots.min(), w_snapshots.max()
    ax1.set_ylim(
        w_min - 0.1 * abs(w_max - w_min) - 1e-5, w_max + 0.1 * abs(w_max - w_min) + 1e-5
    )

    ax1_twin = ax1.twinx()
    (line_f,) = ax1_twin.plot(
        z_les,
        f_snapshots[0],
        "s--",
        color="crimson",
        lw=1.5,
        markersize=4,
        alpha=0.8,
        label="Forcing $f(z, t)$",
    )
    ax1_twin.set_ylabel("Forcing $f(z, t)$", color="crimson")
    ax1_twin.tick_params(axis="y", labelcolor="crimson")

    f_min, f_max = f_snapshots.min(), f_snapshots.max()
    ax1_twin.set_ylim(
        f_min - 0.1 * abs(f_max - f_min) - 1e-5, f_max + 0.1 * abs(f_max - f_min) + 1e-5
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title(
        f"LES Simulation Snapshots & Residuals ($n_{{nodes}} = {len(z_les)}$)"
    )

    # --- Bottom Subplot: Projected Strong Residual ---
    (line_r,) = ax2.plot(
        z_les,
        residual_snapshots[0],
        "^-.",
        color="forestgreen",
        lw=1.5,
        markersize=4,
        label="Projected Residual $\mathcal{R}$",
    )
    ax2.set_xlabel("Domain Coordinate $z$")
    ax2.set_ylabel("Residual $\mathcal{R}$", color="forestgreen")
    ax2.tick_params(axis="y", labelcolor="forestgreen")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    r_min, r_max = residual_snapshots.min(), residual_snapshots.max()
    if abs(r_max - r_min) < 1e-6:
        ax2.set_ylim(-1.0, 1.0)
    else:
        ax2.set_ylim(r_min - 0.1 * abs(r_max - r_min), r_max + 0.1 * abs(r_max - r_min))

    time_text = ax1.text(
        0.02,
        0.82,
        "",
        transform=ax1.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # --- Plot Update Logic ---
    def update_plot(frame: int):
        line_w.set_ydata(w_snapshots[frame])
        line_f.set_ydata(f_snapshots[frame])
        line_r.set_ydata(residual_snapshots[frame])
        time_text.set_text(f"Timestep: {frame} / {len(w_snapshots) - 1}")
        fig.canvas.draw_idle()

    # --- Widgets Setup (Slider & Button) ---
    ax_slider = fig.add_axes([0.2, 0.08, 0.65, 0.03])
    timestep_slider = Slider(
        ax=ax_slider,
        label="Timestep",
        valmin=0,
        valmax=len(w_snapshots) - 1,
        valinit=0,
        valfmt="%d",
    )

    ax_button = fig.add_axes([0.05, 0.07, 0.1, 0.05])
    play_button = Button(
        ax_button, "Pause / Play", color="lightgoldenrodyellow", hovercolor="0.9"
    )

    # State tracking
    is_playing = [True]

    def toggle_play(event):
        is_playing[0] = not is_playing[0]
        if is_playing[0]:
            anim.event_source.start()
            play_button.label.set_text("Pause")
        else:
            anim.event_source.stop()
            play_button.label.set_text("Play")

    play_button.on_clicked(toggle_play)

    def slider_changed(val):
        frame = int(val)
        # Only update if paused or explicitly dragged to avoid recursive loop lag
        if not is_playing[0]:
            update_plot(frame)

    timestep_slider.on_changed(slider_changed)

    # --- Animation Frame Runner ---
    def update(frame: int):
        if is_playing[0]:
            timestep_slider.set_val(frame)  # Moves slider along with play head
            update_plot(frame)
        return [line_w, line_f, line_r, time_text]

    anim = FuncAnimation(
        fig, update, frames=len(w_snapshots), interval=interval, blit=False
    )

    if save_path is not None:
        anim.save(str(save_path))

    plt.show()
    return anim


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    les_run = "n17_dt0.1_t00.0_tf4"
    dt_les = 0.1
    n_nodes_les = 17

    u_field_dir = (
        PROJECT_ROOT
        / "solver_data"
        / "solver_testing"
        / les_run
        / "vms__2_parameter_base"
        / "u_field"
    )

    csv_files = sorted(u_field_dir.glob("sol_t*.csv"))
    z_les = np.linspace(0, 2, n_nodes_les)

    if csv_files:
        w_snapshots_list = []
        f_snapshots_list = []

        print(
            f"Loading {len(csv_files)} velocity & forcing CSV snapshots from {u_field_dir}..."
        )
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            w_snapshots_list.append(df["velocity"].values)
            f_snapshots_list.append(df["forcing"].values)

        w_les_data = np.array(w_snapshots_list)
        f_les_data = np.array(f_snapshots_list)

        print("Computing 3-point Gauss strong residuals and projecting onto nodes...")
        residual_snapshots = compute_gauss_strong_residual_snapshots(
            z_les=z_les, w_snapshots=w_les_data, f_snapshots=f_les_data, dt=dt_les
        )

        print("Launching interactive animation with Play/Pause and Slider...")
        animate_les_snapshots(
            z_les=z_les,
            w_snapshots=w_les_data,
            f_snapshots=f_les_data,
            residual_snapshots=residual_snapshots,
            interval=250,
        )
    else:
        print(f"No CSV files found in {u_field_dir}")
