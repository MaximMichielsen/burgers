import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Publication-Quality Global Settings ---
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    }
)


def plot_profile_comparisons(
    intervals_dir: Path,
    les_base_dir: Path,
    t_start: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
) -> list[tuple[float, float, float, float, float]]:
    """Plot each LES mean profile and Reynolds stress profile side by side (1 row, 2 columns)
    styled with a professional academic color palette and optimized layout.
    """
    mean_profiles_dir = les_base_dir / "mean_profiles"
    reynolds_profiles_dir = les_base_dir / "reynolds_profiles"
    output_dir = les_base_dir / "combined_comparisons"

    if not mean_profiles_dir.exists() or not reynolds_profiles_dir.exists():
        print(f"[Warning] LES profiles directories not found under: {les_base_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    les_files = sorted(mean_profiles_dir.glob("mean_w_*.npy"))
    results = []

    for les_mean_path in les_files:
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", les_mean_path.name)
        if not match:
            continue

        t_end = float(match.group(1))

        les_reynolds_path = (
            reynolds_profiles_dir / f"reynolds_stress_profile_{t_end:.2f}.npy"
        )
        if not les_reynolds_path.exists():
            les_reynolds_path = (
                reynolds_profiles_dir / f"reynolds_stress_profile_{t_end}.npy"
            )
            if not les_reynolds_path.exists():
                continue

        matching_mean_dns = [
            f
            for f in intervals_dir.glob("w_mean_*.npy")
            if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end}.npy")
        ]
        matching_ww_dns = [
            f
            for f in intervals_dir.glob("ww_*.npy")
            if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end}.npy")
        ]

        if not matching_mean_dns or not matching_ww_dns:
            print(f"[Warning] Missing interval DNS references for t_end = {t_end}")
            continue

        dns_mean_path = matching_mean_dns[0]
        dns_ww_path = matching_ww_dns[0]

        mean_profile_dns = np.load(dns_mean_path)
        reynolds_dns = np.load(dns_ww_path)

        les_mean = np.load(les_mean_path)
        les_reynolds = np.load(les_reynolds_path)

        mesh_dns_mean = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))
        mesh_les_mean = np.linspace(z_domain[0], z_domain[1], len(les_mean))
        mesh_dns_ww = np.linspace(z_domain[0], z_domain[1], len(reynolds_dns))
        mesh_les_ww = np.linspace(z_domain[0], z_domain[1], len(les_reynolds))

        # --- Error Metrics: Mean Profile ---
        dns_mean_interp = np.interp(mesh_les_mean, mesh_dns_mean, mean_profile_dns)
        mean_l2 = float(np.linalg.norm(les_mean - dns_mean_interp))
        mean_rmse = float(np.sqrt(np.mean((les_mean - dns_mean_interp) ** 2)))
        dns_mean_norm = np.linalg.norm(dns_mean_interp)
        mean_rel = (mean_l2 / (dns_mean_norm + 1e-12)) * 100.0

        # --- Error Metrics: Reynolds Stress Profile ---
        dns_ww_interp = np.interp(mesh_les_ww, mesh_dns_ww, reynolds_dns)
        ww_l2 = float(np.linalg.norm(les_reynolds - dns_ww_interp))
        ww_rmse = float(np.sqrt(np.mean((les_reynolds - dns_ww_interp) ** 2)))
        dns_ww_norm = np.linalg.norm(dns_ww_interp)
        ww_rel = (ww_l2 / (dns_ww_norm + 1e-12)) * 100.0

        results.append((t_end, mean_l2, mean_rel, ww_l2, ww_rel))

        # --- Create Figure ---
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 5), dpi=300, constrained_layout=True
        )

        for ax in (ax1, ax2):
            ax.axhline(
                0, color="black", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1
            )
            ax.set_xlim(z_domain)
            ax.minorticks_on()
            ax.tick_params(
                which="both", direction="in", top=True, right=True, length=4, width=0.8
            )
            ax.tick_params(which="minor", length=2, width=0.5)

        # Professional Color Palette (Colorblind-safe & publication-refined)
        color_dns_1 = "#1b4965"  # Deep Academic Slate Blue
        color_les_1 = "#e76f51"  # Rich Burnt Coral
        color_dns_2 = "#2a9d8f"  # Muted Teal Green
        color_les_2 = "#b7094c"  # Deep Crimson Red

        # 1. Left Subplot: Mean Velocity Profile
        ax1.plot(
            mesh_dns_mean,
            mean_profile_dns,
            color=color_dns_1,
            linestyle="-",
            linewidth=1.6,
            label=r"DNS Reference ($\langle w \rangle$)",
            zorder=2,
        )
        ax1.plot(
            mesh_les_mean,
            les_mean,
            color=color_les_1,
            linestyle="--",
            linewidth=1.3,
            marker="o",
            markevery=max(1, len(mesh_les_mean) // 16),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=f"LES Model ({len(mesh_les_mean)} pts)",
            zorder=3,
        )

        ax1.set_title(
            r"(a) Mean Velocity Profile $\langle w \rangle$",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=11)

        # Give the y-axis a little breathing room at the top to prevent overlapping
        y_min, y_max = ax1.get_ylim()
        ax1.set_ylim(y_min, y_max * 1.15)

        # Legend moved to lower-right or middle-right where whitespace often exists,
        # or top-center. Let's use center-right or lower right if space permits.
        ax1.legend(
            loc="upper right",
            frameon=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.9,
        )

        # Streamlined, clean metrics text box placed lower down where lines are flatter
        score_text1 = (
            f"$L_2$: {mean_l2:.2e} | RMSE: {mean_rmse:.2e} | Rel. Err: {mean_rel:.1f}%"
        )
        ax1.text(
            0.5,
            0.03,
            score_text1,
            transform=ax1.transAxes,
            fontsize=8.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.4",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
            ),
            zorder=5,
        )

        # 2. Right Subplot: Reynolds Stress Profile
        ax2.plot(
            mesh_dns_ww,
            reynolds_dns,
            color=color_dns_2,
            linestyle="-",
            linewidth=1.6,
            label=r"DNS Reference ($\langle w'w' \rangle$)",
            zorder=2,
        )
        ax2.plot(
            mesh_les_ww,
            les_reynolds,
            color=color_les_2,
            linestyle="--",
            linewidth=1.3,
            marker="s",
            markevery=max(1, len(mesh_les_ww) // 16),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=f"LES Model ({len(mesh_les_ww)} pts)",
            zorder=3,
        )

        ax2.set_title(
            r"(b) Reynolds Stress Profile $\langle w'w' \rangle$",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax2.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax2.set_ylabel(r"$\langle w'w' \rangle$", fontsize=11)

        y_min2, y_max2 = ax2.get_ylim()
        ax2.set_ylim(y_min2, y_max2 * 1.15)

        ax2.legend(
            loc="upper right",
            frameon=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.9,
        )

        score_text2 = (
            f"$L_2$: {ww_l2:.2e} | RMSE: {ww_rmse:.2e} | Rel. Err: {ww_rel:.1f}%"
        )
        ax2.text(
            0.5,
            0.03,
            score_text2,
            transform=ax2.transAxes,
            fontsize=8.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.4",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
            ),
            zorder=5,
        )

        # Save figure cleanly
        save_path = output_dir / f"profile_comparison_{t_end:.2f}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path.name}")

    return results
