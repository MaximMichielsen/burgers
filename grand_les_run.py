from dataclasses import replace
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from ml.reference_scheduler import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECTED_DIR = PROJECT_ROOT / "dns_data" / "projected"
TEMPORAL_PROJECTED_DIR = PROJECT_ROOT / "dns_data" / "temporal" / "projected"
INTERVALS_DIR = PROJECT_ROOT / "dns_data" / "intervals"

# Common configuration parameters
n_nodes_dns = 513
z_length = 2.0
dt = 1e-3
external_coefficients = np.array([1, 1])
simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.TWO_PARAMS

# Data loading dependencies
times_forcing = np.load(TEMPORAL_PROJECTED_DIR / "times.npy")
times_path = PROJECT_ROOT / "dns_data" / "curated" / "dat" / "times.npy"


def discover_intervals(intervals_dir: Path) -> list[tuple[float, float]]:
    """Dynamically discover t_start and t_end pairs from w_mean interval files."""
    intervals = []
    for f in sorted(intervals_dir.glob("w_mean_*.npy")):
        match = re.search(r"w_mean_(\d+\.\d+)_(\d+\.\d+)\.npy$", f.name)
        if match:
            t_start = float(match.group(1))
            t_end = float(match.group(2))
            intervals.append((t_start, t_end))
    return intervals


def plot_profile_comparisons(
        intervals_dir: Path,
        les_profiles_dir: Path,
        t_start: float,
        z_domain: tuple[float, float] = (0.0, 2.0),
) -> None:
    """Plot each LES mean profile against the corresponding interval DNS reference."""
    if not les_profiles_dir.exists():
        return

    les_files = sorted(les_profiles_dir.glob("mean_w_*.npy"))
    for les_path in les_files:
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", les_path.name)
        if not match:
            continue

        t_end = float(match.group(1))
        matching_dns_files = [
            f for f in intervals_dir.glob("w_mean_*.npy")
            if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end}.npy")
        ]

        if not matching_dns_files:
            continue

        dns_profile_path = matching_dns_files[0]
        mean_profile_dns = np.load(dns_profile_path)
        time_interval_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] s"

        les_mean = np.load(les_path)
        mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))
        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

        dns_interp = np.interp(mesh_les, mesh_dns, mean_profile_dns)
        l2_error = float(np.linalg.norm(les_mean - dns_interp))
        dns_norm = np.linalg.norm(dns_interp)
        relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
        ax.plot(mesh_dns, mean_profile_dns, color="royalblue", linestyle="-", linewidth=1.2,
                label=f"DNS Reference ({len(mesh_dns)} pts)", zorder=2)
        ax.plot(mesh_les, les_mean, color="tab:orange", linestyle="--", linewidth=1.0, marker="o",
                markevery=max(1, len(mesh_les) // 16), markersize=5, markerfacecolor="white", markeredgewidth=1.2,
                label=f"LES Model ({len(mesh_les)} pts)", zorder=3)

        all_data = np.concatenate([mean_profile_dns, les_mean])
        y_min, y_max = all_data.min(), all_data.max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax.set_title(r"Mean Velocity Profile Comparison $\langle w \rangle$ " + f"({time_interval_str})", fontsize=13,
                     pad=10)
        ax.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
        ax.set_ylabel(r"Mean Velocity $\langle w \rangle$", fontsize=11)
        ax.set_xlim(z_domain)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.minorticks_on()
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

        score_text = rf"$\mathrm{{L}}_2$ Error: {l2_error:.4e}" + f"\nRel. Error: {relative_l2_error:.2f}%"
        ax.text(0.03, 0.05, score_text, transform=ax.transAxes, fontsize=9.5, verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.85), zorder=5)

        plt.tight_layout()
        plt.savefig(les_profiles_dir / f"profile_comparison_{t_end:.2f}.png", dpi=300)
        plt.close(fig)


def plot_reynolds_stress_profile(
        intervals_dir: Path,
        les_profiles_dir: Path,
        t_start: float,
        z_domain: tuple[float, float] = (0.0, 2.0),
) -> None:
    """Plot Reynolds stress profile compared to the interval DNS profile."""
    if not les_profiles_dir.exists():
        return

    les_files = sorted(les_profiles_dir.glob("reynolds_stress_profile_*.npy"))
    for les_path in les_files:
        match = re.search(r"reynolds_stress_profile_(\d+\.\d+)\.npy$", les_path.name)
        if not match:
            continue

        t_end = float(match.group(1))
        matching_dns_files = [
            f for f in intervals_dir.glob("ww_*.npy")
            if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end:.2f}.npy")
        ]

        if not matching_dns_files:
            continue

        dns_profile_path = matching_dns_files[0]
        reynolds_dns = np.load(dns_profile_path)
        time_interval_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] s"

        les_reynolds = np.load(les_path)
        mesh_dns = np.linspace(z_domain[0], z_domain[1], len(reynolds_dns))
        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_reynolds))

        dns_interp = np.interp(mesh_les, mesh_dns, reynolds_dns)
        l2_error = float(np.linalg.norm(les_reynolds - dns_interp))
        dns_norm = np.linalg.norm(dns_interp)
        relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
        ax.plot(mesh_dns, reynolds_dns, color="mediumseagreen", linestyle="-", linewidth=1.2,
                label=f"DNS Reference ({len(mesh_dns)} pts)", zorder=2)
        ax.plot(mesh_les, les_reynolds, color="crimson", linestyle="--", linewidth=1.0, marker="o",
                markevery=max(1, len(mesh_les) // 16), markersize=5, markerfacecolor="white", markeredgewidth=1.2,
                label=f"LES Model ({len(mesh_les)} pts)", zorder=3)

        all_data = np.concatenate([reynolds_dns, les_reynolds])
        y_min, y_max = all_data.min(), all_data.max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax.set_title(r"Reynolds Stress Profile Comparison $\langle w'w' \rangle$ " + f"({time_interval_str})",
                     fontsize=13, pad=10)
        ax.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
        ax.set_ylabel(r"Reynolds Stress $\langle w'w' \rangle$", fontsize=11)
        ax.set_xlim(z_domain)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.minorticks_on()
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9.5)

        score_text = rf"$\mathrm{{L}}_2$ Error: {l2_error:.4e}" + f"\nRel. Error: {relative_l2_error:.2f}%"
        ax.text(0.03, 0.05, score_text, transform=ax.transAxes, fontsize=9.5, verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.85), zorder=5)

        plt.tight_layout()
        plt.savefig(les_profiles_dir / f"reynolds_profile_comparison_{t_end:.2f}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    intervals = discover_intervals(INTERVALS_DIR)
    n_nodes_list = [17, 33]

    batch_output_dir = PROJECT_ROOT / "solver_data" / "batch_interval_runs"

    for t_start, t_end in intervals:
        timespan = t_end - t_start
        print(f"\n==============================================")
        print(f"Running Interval: t in [{t_start}, {t_end}] (duration: {timespan:.4f}s)")
        print(f"==============================================")

        for n_nodes_les in n_nodes_list:
            print(f"--- Running LES with n_nodes_les = {n_nodes_les} ---")

            forcing_projected = np.load(TEMPORAL_PROJECTED_DIR / f"forcing_l2_{n_nodes_les}.npy")

            problem = Problem(
                name="tcf_1d",
                domain_length=z_length,
                domain_timespan=timespan,
                reynolds=100.0,
                initial_condition=None,
                forcing=forcing_projected,
                forcing_times=times_forcing,
                forcing_is_steady=False,
                boundary_condition_type="fixed",
                boundary_condition_value=0.0,
                t_start=t_start,
            )

            disc_config = DiscretizationConfig(
                n_nodes_les=n_nodes_les,
                n_nodes_dns=n_nodes_dns,
                temporal_refinement=1,
                set_dt_les=dt,
                domain_length=z_length,
                domain_timespan=timespan,
            )

            reference_trajectory = ReferenceTrajectory(
                target_profile_dir=INTERVALS_DIR,
                projected_w_path=PROJECTED_DIR / f"w_linear_{n_nodes_les}.npy",
                times_path=times_path,
                n_les_nodes=n_nodes_les,
                t_start=t_start,
                t_end=t_end,
            )

            problem = replace(problem, initial_condition=reference_trajectory.initial_condition)

            master_path = (
                    batch_output_dir
                    / f"t_{t_start}_{t_end}"
                    / f"n{n_nodes_les}_p{tau_model.output_dimensions}_dt{dt}"
            )

            solver_les = SolverBase(
                problem=problem,
                disc_config=disc_config,
                simulation_mode=simulation_mode,
                master_path=master_path,
                tau_model=tau_model,
                external_correction_coefficients=external_coefficients,
            )

            solver_les.run_simulation()
            solver_les.post_plotting()

            # Generate profile comparisons automatically
            plot_profile_comparisons(
                intervals_dir=INTERVALS_DIR,
                les_profiles_dir=master_path / "mean_profiles",
                t_start=t_start,
                z_domain=(0.0, z_length),
            )

            plot_reynolds_stress_profile(
                intervals_dir=INTERVALS_DIR,
                les_profiles_dir=master_path / "reynolds_profiles",
                t_start=t_start,
                z_domain=(0.0, z_length),
            )
            print(f"[COMPLETED] n_nodes={n_nodes_les} for interval [{t_start}, {t_end}]")

    print("\n[ALL BATCH RUNS COMPLETED SUCCESSFULLY]")