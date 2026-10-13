import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_profile_comparisons(
    dns_profile_path: Path,
    les_profiles_dir: Path,
    t_start: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
    dns_frame: int = 0,
) -> list[tuple[float, float, float]]:
    """Plot each LES mean profile against the DNS reference and save one figure per file.

    Returns a list of (t_end, l2_error, relative_l2_error_percent).
    """
    mean_profile_dns = np.load(dns_profile_path)[dns_frame]
    les_files = sorted(les_profiles_dir.glob("mean_w_*.npy"))
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))

    results: list[tuple[float, float, float]] = []

    for les_path in les_files:
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", les_path.name)
        if not match:
            continue

        t_end = float(match.group(1))
        time_interval_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] s"

        les_mean = np.load(les_path)
        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

        # Interpolate DNS onto LES mesh
        dns_interp = np.interp(mesh_les, mesh_dns, mean_profile_dns)

        # L2 error metrics
        l2_error = float(np.linalg.norm(les_mean - dns_interp))
        dns_norm = np.linalg.norm(dns_interp)
        relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0
        results.append((t_end, l2_error, relative_l2_error))

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

        ax.plot(
            mesh_dns,
            mean_profile_dns,
            color="royalblue",
            linestyle="-",
            linewidth=1.2,
            label=f"DNS Reference ({len(mesh_dns)} pts)",
            zorder=2,
        )

        ax.plot(
            mesh_les,
            les_mean,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            marker="o",
            markevery=max(1, len(mesh_les) // 16),
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=f"LES Model ({len(mesh_les)} pts)",
            zorder=3,
        )

        # Dynamic y-limits with 10% padding
        all_data = np.concatenate([mean_profile_dns, les_mean])
        y_min, y_max = all_data.min(), all_data.max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax.set_title(
            r"Mean Velocity Profile Comparison $\langle w \rangle$ "
            + f"({time_interval_str})",
            fontsize=13,
            pad=10,
        )
        ax.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
        ax.set_ylabel(r"Mean Velocity $\langle w \rangle$", fontsize=11)

        ax.set_xlim(z_domain)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)
        ax.minorticks_on()
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

        score_text = (
            rf"$\mathrm{{L}}_2$ Error: {l2_error:.4e}"
            + f"\nRel. Error: {relative_l2_error:.2f}%"
        )
        ax.text(
            0.03,
            0.05,
            score_text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
            zorder=5,
        )

        plt.tight_layout()
        plt.savefig(les_profiles_dir / f"profile_comparison_{t_end:.2f}.png", dpi=300)
        plt.close(fig)

    return results


def plot_reynolds_stress_profile(
    dns_profile_path: Path,
    les_profiles_dir: Path,
    t_start: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
    dns_frame: int = 0,
) -> None:
    """Plot Reynolds stress profile, compared to DNS profile."""
    reynolds_dns = np.load(dns_profile_path)[dns_frame]
    les_files = sorted(les_profiles_dir.glob("reynolds_stress_profile_*.npy"))
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(reynolds_dns))

    for les_path in les_files:
        match = re.search(r"reynolds_stress_profile_(\d+\.\d+)\.npy$", les_path.name)
        if not match:
            continue

        t_end = float(match.group(1))
        time_interval_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] s"

        les_reynolds = np.load(les_path)
        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_reynolds))

        dns_interp = np.interp(mesh_les, mesh_dns, reynolds_dns)

        # L2 error metrics
        l2_error = float(np.linalg.norm(les_reynolds - dns_interp))
        dns_norm = np.linalg.norm(dns_interp)
        relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

        ax.plot(
            mesh_dns,
            reynolds_dns,
            color="mediumseagreen",
            linestyle="-",
            linewidth=1.2,
            label=f"DNS Reference ({len(mesh_dns)} pts)",
            zorder=2,
        )

        ax.plot(
            mesh_les,
            les_reynolds,
            color="crimson",
            linestyle="--",
            linewidth=1.0,
            marker="o",
            markevery=max(1, len(mesh_les) // 16),
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=f"LES Model ({len(mesh_les)} pts)",
            zorder=3,
        )

        # Dynamic y-limits with 10% padding
        all_data = np.concatenate([reynolds_dns, les_reynolds])
        y_min, y_max = all_data.min(), all_data.max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax.set_title(
            r"Mean Velocity Profile Comparison $\langle w \rangle$ "
            + f"({time_interval_str})",
            fontsize=13,
            pad=10,
        )
        ax.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
        ax.set_ylabel(r"Mean Velocity $\langle w \rangle$", fontsize=11)

        ax.set_xlim(z_domain)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)
        ax.minorticks_on()
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9.5)

        score_text = (
            rf"$\mathrm{{L}}_2$ Error: {l2_error:.4e}"
            + f"\nRel. Error: {relative_l2_error:.2f}%"
        )
        ax.text(
            0.03,
            0.05,
            score_text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
            zorder=5,
        )

        plt.tight_layout()
        plt.savefig(les_profiles_dir / f"profile_comparison_{t_end:.2f}.png", dpi=300)
        plt.close(fig)

    return


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    n_nodes = 33
    parameters = 2
    dt = 1e-3

    les_dir = f"n{n_nodes}_p{parameters}_dt{dt}"

    plot_profile_comparisons(
        dns_profile_path=PROJECT_ROOT / "dns_data" / "curated" / "stat" / "mean_w.npy",
        les_profiles_dir=(PROJECT_ROOT / "solver_data" / "base_les" / les_dir / "mean_profiles"),
        t_start=308.01,
        z_domain=(0.0, 2.0),
    )

    plot_reynolds_stress_profile(
        PROJECT_ROOT / "dns_data" / "curated" / "stat" / "mean_reynolds_w.npy",
        les_profiles_dir=(
            PROJECT_ROOT / "solver_data" / "base_les" / les_dir / "reynolds_profiles"
        ),
        t_start=308.01,
        z_domain=(0.0, 2.0),
    )
