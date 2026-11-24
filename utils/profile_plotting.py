import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray

from setup.constants import Z_LENGTH_DNS, DOUBLE_FIG_SIZE, PLT_PARAMS, STYLE_PROJECTED, STYLE_REFERENCE, \
    STYLE_VERMILLION, STYLE_BLUE
from utils.diagnostics import compute_error_metrics

# --- Publication-Quality Global Settings ---
plt.rcParams.update(PLT_PARAMS)


def plot_profile_comparisons_les_vs_reference(
    ref_intervals_dir: Path,
    les_dir: Path,
    t_start: float,
    z_domain: tuple[float, float] = (0.0, Z_LENGTH_DNS),
) -> None:
    """Plot LES mean profile and Reynolds stress profile side by side (1 row, 2 columns)."""
    output_dir = les_dir / "combined_comparisons"

    mean_profiles_dir = les_dir / "mean_profiles"
    reynolds_profiles_dir = les_dir / "reynolds_profiles"

    if not mean_profiles_dir.exists() or not reynolds_profiles_dir.exists():
        raise FileNotFoundError(
            f"[Warning] LES profiles directories not found under: {les_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    les_files = sorted(mean_profiles_dir.glob("mean_w_*.npy"))
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
                raise FileNotFoundError(
                    f"[Warning] Cannot find corresponding Reynolds stress profile at {les_reynolds_path}"
                )

        dns_mean_path = ref_intervals_dir / f"w_mean_{t_start}_{t_end}.npy"
        dns_ww_path = ref_intervals_dir / f"ww_{t_start}_{t_end}.npy"

        if not dns_mean_path.exists() or not dns_ww_path.exists():
            dns_mean_path = ref_intervals_dir / f"w_mean_{t_start:.2f}_{t_end:.2f}.npy"
            dns_ww_path = ref_intervals_dir / f"ww_{t_start:.2f}_{t_end:.2f}.npy"

        if not dns_mean_path.exists() or not dns_ww_path.exists():
            raise FileNotFoundError(
                f"[Warning] Missing interval DNS references for [{t_start}, {t_end}]"
            )

        dns_mean = np.load(dns_mean_path)
        dns_reynolds = np.load(dns_ww_path)
        les_mean = np.load(les_mean_path)
        les_reynolds = np.load(les_reynolds_path)

        n_dns_nodes = len(dns_mean)
        n_les_nodes = len(les_mean)

        mesh_dns = np.linspace(z_domain[0], z_domain[1], n_dns_nodes)
        mesh_les = np.linspace(z_domain[0], z_domain[1], n_les_nodes)

        # Interpolate / Project DNS reference onto the LES mesh
        proj_dns_mean = np.interp(mesh_les, mesh_dns, dns_mean)
        proj_dns_reynolds = np.interp(mesh_les, mesh_dns, dns_reynolds)

        # Compute error metrics using the exact utility function from training
        mean_l2, mean_rmse, mean_rel = compute_error_metrics(les_mean, proj_dns_mean)
        ww_l2, ww_rmse, ww_rel = compute_error_metrics(les_reynolds, proj_dns_reynolds)

        # --- Create Figure (1 row, 2 columns layout) ---
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=DOUBLE_FIG_SIZE, dpi=300, constrained_layout=True
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

        # --- Left Subplot: Mean Velocity Profile ---
        ax1.plot(
            mesh_dns,
            dns_mean,
            label=f"DNS ({n_dns_nodes} pts)",
            **STYLE_REFERENCE,
        )
        ax1.plot(
            mesh_les,
            proj_dns_mean,
            markevery=max(1, len(mesh_les) // 16),
            label=f"Proj. DNS ({len(mesh_les)} pts)",
            **STYLE_PROJECTED,
        )
        ax1.plot(
            mesh_les,
            les_mean,
            markevery=max(1, len(mesh_les) // 16),
            label=f"LES Model ({len(mesh_les)} pts)",
            **STYLE_VERMILLION,
        )

        time_str = f"$t \\in [{t_start:.2f}, {t_end:.2f}]$"
        ax1.set_title(
            rf"(a) Mean Velocity Profile $\langle w \rangle$ | {time_str}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=11)

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

        # --- Right Subplot: Reynolds Stress Profile ---
        ax2.plot(
            mesh_dns,
            dns_reynolds,
            label=f"DNS ({n_dns_nodes} pts)",
            **STYLE_REFERENCE,
        )
        ax2.plot(
            mesh_les,
            proj_dns_reynolds,
            markevery=max(1, len(mesh_les) // 16),
            label=f"Proj. DNS ({len(mesh_les)} pts)",
            **STYLE_PROJECTED,
        )
        ax2.plot(
            mesh_les,
            les_reynolds,
            markevery=max(1, len(mesh_les) // 16),
            label=f"LES Model ({len(mesh_les)} pts)",
            **STYLE_VERMILLION,
        )
        ax2.set_title(
            rf"(b) Reynolds Stress Profile $\langle w'w' \rangle$ | {time_str}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax2.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax2.set_ylabel(r"$\langle w'w' \rangle$", fontsize=11)

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

        # Shared legend below both panels
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = (h1, l1) if len(h1) >= len(h2) else (h2, l2)
        fig.legend(
            handles,
            labels,
            loc="outside lower center",  # requires matplotlib >= 3.7
            ncol=len(handles),
            fontsize=9,
            frameon=False,
        )

        # Save figure cleanly
        save_path = output_dir / f"profile_comparison_{t_end:.2f}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        save_path = output_dir / f"profile_comparison_{t_end:.2f}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path.name}")


def plot_profile_comparisons_dns_filtered_vs_unfiltered(
    intervals_dir: Path,
    dns_dir: Path,
    t_start: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
) -> None:
    """Plot DNS mean profile and Reynolds stress profile side by side (1 row, 2 columns)."""
    output_dir = dns_dir / "combined_comparisons"
    labels = ["DNS TCF Reference", "DNS Solver (Unfiltered)", "DNS Solver (Filtered)"]

    time_dir = dns_dir.name
    filtered_dns_dir = dns_dir.parent.parent / "filtered" / time_dir

    mean_profiles_dir = dns_dir / "mean_profiles"
    reynolds_profiles_dir = dns_dir / "reynolds_profiles"

    if not mean_profiles_dir.exists() or not reynolds_profiles_dir.exists():
        raise FileNotFoundError(
            f"[Warning] LES profiles directories not found under: {dns_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dns_files = sorted(mean_profiles_dir.glob("mean_w_*.npy"))

    for solver_unfiltered_mean_path in dns_files:
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", solver_unfiltered_mean_path.name)
        if not match:
            continue

        t_end = float(match.group(1))

        solver_unfiltered_reynolds_path = (
            reynolds_profiles_dir / f"reynolds_stress_profile_{t_end:.2f}.npy"
        )
        if not solver_unfiltered_reynolds_path.exists():
            solver_unfiltered_reynolds_path = (
                reynolds_profiles_dir / f"reynolds_stress_profile_{t_end}.npy"
            )
            if not solver_unfiltered_reynolds_path.exists():
                raise FileNotFoundError(
                    f"[Warning] DNS Reynolds stress profile not found under: {solver_unfiltered_reynolds_path}"
                )

        solver_filtered_mean_path = (
            filtered_dns_dir / "mean_profiles" / solver_unfiltered_mean_path.name
        )
        solver_filtered_reynolds_path = (
            filtered_dns_dir
            / "reynolds_profiles"
            / f"reynolds_stress_profile_{t_end:.2f}.npy"
        )

        if (
            not solver_filtered_mean_path.exists()
            and not solver_filtered_reynolds_path.exists()
        ):
            raise FileNotFoundError("Did not found corresponding filtered directories.")

        if t_end == 314.99:
            t_end += 0.01

        tcf_mean_path = intervals_dir / f"w_mean_{t_start}_{t_end}.npy"
        tcf_reynolds_path = intervals_dir / f"ww_{t_start}_{t_end}.npy"

        if not tcf_mean_path.exists() or not tcf_reynolds_path.exists():
            tcf_mean_path = intervals_dir / f"w_mean_{t_start:.2f}_{t_end:.2f}.npy"
            tcf_reynolds_path = intervals_dir / f"ww_{t_start:.2f}_{t_end:.2f}.npy"

        if not tcf_mean_path.exists() or not tcf_reynolds_path.exists():
            raise FileNotFoundError(
                f"[Warning] Missing interval DNS references for [{t_start}, {t_end}]"
            )

        tcf_mean = np.load(tcf_mean_path)
        tcf_reynolds = np.load(tcf_reynolds_path)

        solver_unfiltered_mean = np.load(solver_unfiltered_mean_path)
        solver_unfiltered_reynolds = np.load(solver_unfiltered_reynolds_path)

        solver_filtered_mean = np.load(solver_filtered_mean_path)
        solver_filtered_reynolds = np.load(solver_filtered_reynolds_path)

        n_dns_nodes = len(tcf_mean)
        mesh_dns = np.linspace(z_domain[0], z_domain[1], n_dns_nodes)

        # Compute error metrics using the exact utility function from training
        mean_unfiltered_l2, mean_unfiltered_rmse, mean_unfiltered_rel = (
            compute_error_metrics(solver_unfiltered_mean, tcf_mean)
        )
        ww_unfiltered_l2, ww_unfiltered_rmse, ww_unfiltered_rel = compute_error_metrics(
            solver_unfiltered_reynolds, tcf_reynolds
        )
        mean_filtered_l2, mean_filtered_rmse, mean_filtered_rel = compute_error_metrics(
            solver_filtered_mean, tcf_mean
        )
        ww_filtered_l2, ww_filtered_rmse, ww_filtered_rel = compute_error_metrics(
            solver_filtered_reynolds, tcf_reynolds
        )

        # --- Create Figure (1 row, 2 columns layout) ---
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=DOUBLE_FIG_SIZE, dpi=300, constrained_layout=True
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

        # --- Left Subplot: Mean Velocity Profile ---
        ax1.plot(
            mesh_dns,
            tcf_mean,
            label=labels[0],
            **STYLE_REFERENCE,
        )
        ax1.plot(
            mesh_dns,
            solver_unfiltered_mean,
            markevery=max(1, len(mesh_dns) // 16),
            label=labels[1],
            **STYLE_BLUE,
        )

        ax1.plot(
            mesh_dns,
            solver_filtered_mean,
            markevery=max(1, len(mesh_dns) // 16),
            label=labels[2],
            **STYLE_VERMILLION,
        )

        time_str = f"$t \\in [{t_start:.2f}, {t_end:.2f}]$"
        ax1.set_title(
            rf"(a) Mean Velocity Profile $\langle w \rangle$ | {time_str}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=11)

        mean_score_1 = (r"DNS Solver (Unfiltered) vs. TCF Reference" + "\n") + (
            f"$L_2$: {mean_unfiltered_l2:.2e} | RMSE: {mean_unfiltered_rmse:.2e} | Rel. Err: {mean_unfiltered_rel:.1f}%"
        )
        ax1.text(
            0.5,
            0.025,
            mean_score_1,
            transform=ax1.transAxes,
            fontsize=7.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
                alpha=0.95,
            ),
            zorder=5,
        )

        mean_score_2 = (
            r"DNS Solver (Filtered) vs. TCF reference" + "\n"
            f"$L_2$: {mean_filtered_l2:.2e} | RMSE: {mean_filtered_rmse:.2e} | "
            f"Rel. Err: {mean_filtered_rel:.1f}%"
        )
        ax1.text(
            0.5,
            0.15,
            mean_score_2,
            transform=ax1.transAxes,
            fontsize=7.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
                alpha=0.95,
            ),
            zorder=5,
        )

        # --- Right Subplot: Reynolds Stress Profile ---
        ax2.plot(
            mesh_dns,
            tcf_reynolds,
            label=labels[0],
            **STYLE_REFERENCE,
        )
        ax2.plot(
            mesh_dns,
            solver_unfiltered_reynolds,
            markevery=max(1, len(mesh_dns) // 16),
            label=labels[1],
            **STYLE_BLUE,
        )

        ax2.plot(
            mesh_dns,
            solver_filtered_reynolds,
            markevery=max(1, len(mesh_dns) // 16),
            label=labels[2],
            **STYLE_VERMILLION,
        )

        ax2.set_title(
            rf"(b) Reynolds Stress Profile $\langle w'w' \rangle$ | {time_str}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax2.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax2.set_ylabel(r"$\langle w'w' \rangle$", fontsize=11)

        stress_score_1 = (
            (r"DNS Solver (Unfiltered) vs. TCF reference" + "\n")
            + f"$L_2$: {ww_unfiltered_l2:.2e} | RMSE: {ww_unfiltered_rmse:.2e} | Rel. Err: {ww_unfiltered_rel:.1f}%"
        )
        ax2.text(
            0.5,
            0.025,
            stress_score_1,
            transform=ax2.transAxes,
            fontsize=7.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
                alpha=0.95,
            ),
            zorder=5,
        )

        stress_score_2 = (
            r"DNS Solver (Filtered) vs. TCF reference" + "\n"
            f"$L_2$: {ww_filtered_l2:.2e} | RMSE: {ww_filtered_rmse:.2e} | "
            f"Rel. Err: {ww_filtered_rel:.1f}%"
        )
        ax2.text(
            0.5,
            0.15,
            stress_score_2,
            transform=ax2.transAxes,
            fontsize=7.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
                alpha=0.95,
            ),
            zorder=5,
        )

        # Shared legend below both panels
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = (h1, l1) if len(h1) >= len(h2) else (h2, l2)
        fig.legend(
            handles,
            labels,
            loc="outside lower center",  # requires matplotlib >= 3.7
            ncol=len(handles),
            fontsize=9,
            frameon=False,
        )

        # Save figure cleanly
        save_path = output_dir / f"profile_comparison_{t_end:.2f}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        save_path = output_dir / f"profile_comparison_{t_end:.2f}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path.name}")


def plot_profile_comparisons_dns_filtered_vs_reference(
    intervals_dir: Path,
    dns_dir: Path,
    t_start: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
) -> None:
    """Plot DNS mean profile and Reynolds stress profile side by side (1 row, 2 columns)."""

    mean_profiles_dir = dns_dir / "mean_profiles"
    reynolds_profiles_dir = dns_dir / "reynolds_profiles"
    output_dir = dns_dir / "combined_comparisons"

    labels = ["DNS TCF Reference", "DNS Solver (Filtered)"]

    if not mean_profiles_dir.exists() or not reynolds_profiles_dir.exists():
        raise FileNotFoundError(
            f"[Warning] DNS profiles directories not found under: {dns_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    les_files = sorted(mean_profiles_dir.glob("mean_w_*.npy"))

    for solver_filtered_mean_path in les_files:
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", solver_filtered_mean_path.name)
        if not match:
            continue

        t_end = float(match.group(1))

        solver_filtered_reynolds_path = (
            reynolds_profiles_dir / f"reynolds_stress_profile_{t_end:.2f}.npy"
        )
        if not solver_filtered_reynolds_path.exists():
            solver_filtered_reynolds_path = (
                reynolds_profiles_dir / f"reynolds_stress_profile_{t_end}.npy"
            )
            if not solver_filtered_reynolds_path.exists():
                raise FileNotFoundError(
                    f"[Warning] DNS Reynolds stress profile not found under: {solver_filtered_reynolds_path}"
                )

        if t_end == 314.99:
            t_end += 0.01

        tcf_mean_path = intervals_dir / f"w_mean_{t_start}_{t_end}.npy"
        tcf_reynolds_path = intervals_dir / f"ww_{t_start}_{t_end}.npy"

        if not tcf_mean_path.exists() or not tcf_reynolds_path.exists():
            # Fallback formatting check with explicit float precision
            tcf_mean_path = intervals_dir / f"w_mean_{t_start:.2f}_{t_end:.2f}.npy"
            tcf_reynolds_path = intervals_dir / f"ww_{t_start:.2f}_{t_end:.2f}.npy"

        if not tcf_mean_path.exists() or not tcf_reynolds_path.exists():
            raise FileNotFoundError(
                f"[Warning] Missing interval DNS references for [{t_start}, {t_end}]"
            )

        # Load raw high-resolution DNS profiles
        tcf_mean = np.load(tcf_mean_path)
        tcf_reynolds = np.load(tcf_reynolds_path)

        solver_filtered_mean = np.load(solver_filtered_mean_path)
        solver_filtered_reynolds = np.load(solver_filtered_reynolds_path)

        # Set up meshes matching discretization/reference trajectory
        n_dns_nodes = len(tcf_mean)
        mesh_dns = np.linspace(z_domain[0], z_domain[1], n_dns_nodes)

        # Compute error metrics using the exact utility function from training
        mean_l2, mean_rmse, mean_rel = compute_error_metrics(
            solver_filtered_mean, tcf_mean
        )
        ww_l2, ww_rmse, ww_rel = compute_error_metrics(
            solver_filtered_reynolds, tcf_reynolds
        )

        # --- Create Figure (1 row, 2 columns layout) ---
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=DOUBLE_FIG_SIZE, dpi=300, constrained_layout=True
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

        # --- Left Subplot: Mean Velocity Profile ---
        ax1.plot(
            mesh_dns,
            tcf_mean,
            label=labels[0],
            **STYLE_REFERENCE,
        )
        ax1.plot(
            mesh_dns,
            solver_filtered_mean,
            markevery=max(1, len(mesh_dns) // 16),
            label=labels[1],
            **STYLE_VERMILLION,
        )

        time_str = f"$t \\in [{t_start:.2f}, {t_end:.2f}]$"
        ax1.set_title(
            rf"(a) Mean Velocity Profile $\langle w \rangle$ | {time_str}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=11)

        # When both solver resolutions are available, identify each score box.
        # Otherwise keep the single score box compact and title-free.
        mean_score_1 = (r"DNS Solver (Unfiltered) vs. TCF Reference" + "\n") + (
            f"$L_2$: {mean_l2:.2e} | RMSE: {mean_rmse:.2e} | Rel. Err: {mean_rel:.1f}%"
        )
        ax1.text(
            0.5,
            0.025,
            mean_score_1,
            transform=ax1.transAxes,
            fontsize=7.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
                alpha=0.95,
            ),
            zorder=5,
        )

        # --- Right Subplot: Reynolds Stress Profile ---
        ax2.plot(
            mesh_dns,
            tcf_reynolds,
            label=labels[0],
            **STYLE_REFERENCE,
        )
        ax2.plot(
            mesh_dns,
            solver_filtered_reynolds,
            markevery=max(1, len(mesh_dns) // 16),
            label=labels[1],
            **STYLE_VERMILLION,
        )

        ax2.set_title(
            rf"(b) Reynolds Stress Profile $\langle w'w' \rangle$ | {time_str}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax2.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax2.set_ylabel(r"$\langle w'w' \rangle$", fontsize=11)

        stress_score_1 = (
            r"DNS Solver (Unfiltered) vs. TCF reference" + "\n"
        ) + f"$L_2$: {ww_l2:.2e} | RMSE: {ww_rmse:.2e} | Rel. Err: {ww_rel:.1f}%"
        ax2.text(
            0.5,
            0.025,
            stress_score_1,
            transform=ax2.transAxes,
            fontsize=7.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
                alpha=0.95,
            ),
            zorder=5,
        )

        # Shared legend below both panels
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = (h1, l1) if len(h1) >= len(h2) else (h2, l2)
        fig.legend(
            handles,
            labels,
            loc="outside lower center",  # requires matplotlib >= 3.7
            ncol=len(handles),
            fontsize=9,
            frameon=False,
        )

        # Save figure cleanly
        save_path = output_dir / f"profile_comparison_{t_end:.2f}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        save_path = output_dir / f"profile_comparison_{t_end:.2f}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path.name}")


def plot_profile_comparisons_filtering_les(
    les_base_dir: Path,
    t_start: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
) -> None:
    """Plot each LES mean profile and Reynolds stress profile side by side (1 row, 2 columns)
    styled identically to the TD3 training evaluation pipeline using the exact interval DNS files.
    """
    time_dir = les_base_dir.name
    run_dir = les_base_dir.parent.name
    filtered_les_dir = (
        les_base_dir.parent.parent.parent / "filtered_forcing" / run_dir / time_dir
    )

    mean_profiles_dir = les_base_dir / "mean_profiles"
    reynolds_profiles_dir = les_base_dir / "reynolds_profiles"
    output_dir = les_base_dir / "combined_comparisons"

    if not mean_profiles_dir.exists() or not reynolds_profiles_dir.exists():
        print(f"[Warning] LES profiles directories not found under: {les_base_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    les_files = sorted(mean_profiles_dir.glob("mean_w_*.npy"))

    for solver_unfiltered_mean_path in les_files:
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", solver_unfiltered_mean_path.name)
        if not match:
            continue

        t_end = float(match.group(1))

        solver_unfiltered_reynolds_path = (
            reynolds_profiles_dir / f"reynolds_stress_profile_{t_end:.2f}.npy"
        )
        if not solver_unfiltered_reynolds_path.exists():
            solver_unfiltered_reynolds_path = (
                reynolds_profiles_dir / f"reynolds_stress_profile_{t_end}.npy"
            )
            if not solver_unfiltered_reynolds_path.exists():
                continue

        solver_filtered_mean_path = (
            filtered_les_dir / "mean_profiles" / solver_unfiltered_mean_path.name
        )
        solver_filtered_reynolds_path = (
            filtered_les_dir
            / "reynolds_profiles"
            / f"reynolds_stress_profile_{t_end:.2f}.npy"
        )

    les_mean_filtered = np.load(solver_filtered_mean_path)
    les_mean_unfiltered = np.load(solver_unfiltered_mean_path)

    les_reynolds_filtered = np.load(solver_filtered_reynolds_path)
    les_reynolds_unfiltered = np.load(solver_unfiltered_reynolds_path)

    output_dir = les_base_dir

    t_end = t_start + 1
    style_1 = {
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "o",
        "linewidth": 1.3,
        "markersize": 4.5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.1,
    }

    style_2 = {
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.3,
        "markersize": 4.5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.1,
    }

    labels = ["LES (Unfiltered)", "LES (Filtered)"]

    mesh_les = np.linspace(0, Z_LENGTH_DNS, len(les_mean_filtered))

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Create Figure (1 row, 2 columns layout) ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), dpi=300, constrained_layout=True
    )

    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1)
        ax.set_xlim(z_domain)
        ax.minorticks_on()
        ax.tick_params(
            which="both", direction="in", top=True, right=True, length=4, width=0.8
        )
        ax.tick_params(which="minor", length=2, width=0.5)

    # --- Left Subplot: Mean Velocity Profile ---
    ax1.plot(
        mesh_les,
        les_mean_unfiltered,
        label=labels[0],
        markevery=max(1, len(mesh_les) // 16),
        **style_1,
    )
    ax1.plot(
        mesh_les,
        les_mean_filtered,
        markevery=max(1, len(mesh_les) // 16),
        label=labels[1],
        **style_2,
    )

    ax1.set_title(
        r"(a) Mean Velocity Profile $\langle w \rangle$",
        fontsize=11,
        pad=8,
        loc="left",
    )
    ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
    ax1.set_ylabel(r"$\langle w \rangle$", fontsize=11)

    # --- Right Subplot: Reynolds Stress Profile ---
    ax2.plot(
        mesh_les,
        les_reynolds_unfiltered,
        markevery=max(1, len(mesh_les) // 16),
        label=labels[0],
        **style_1,
    )

    ax2.plot(
        mesh_les,
        les_reynolds_filtered,
        markevery=max(1, len(mesh_les) // 16),
        label=labels[1],
        **style_2,
    )

    ax2.set_title(
        r"(b) Reynolds Stress Profile $\langle w'w' \rangle$",
        fontsize=11,
        pad=8,
        loc="left",
    )
    ax2.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
    ax2.set_ylabel(r"$\langle w'w' \rangle$", fontsize=11)

    # Shared legend below both panels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles, labels = (h1, l1) if len(h1) >= len(h2) else (h2, l2)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",  # requires matplotlib >= 3.7
        ncol=len(handles),
        fontsize=9,
        frameon=False,
    )

    # Save figure cleanly
    save_path = output_dir / f"profile_comparison_{t_end:.2f}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path.name}")


def plot_mms_profile(
    mms_solution: NDArray, les_solution: NDArray, save_dir: Path, z_domain=(0, 2)
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    mesh = np.linspace(z_domain[0], z_domain[1], len(les_solution))
    mean_l2, mean_rmse, mean_rel = compute_error_metrics(les_solution, mms_solution)

    # --- Create Figure (1 row, 2 columns layout) ---
    (
        fig,
        ax,
    ) = plt.subplots(1, 1, figsize=(6.5, 5), dpi=300, constrained_layout=True)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1)
    ax.set_xlim(z_domain)
    ax.minorticks_on()
    ax.tick_params(
        which="both", direction="in", top=True, right=True, length=4, width=0.8
    )
    ax.tick_params(which="minor", length=2, width=0.5)

    # Professional Color Palette (Colorblind-safe & publication-refined)
    color_dns_1, color_les_1 = "#1b4965", "#e76f51"  # Mean Profile

    # --- Left Subplot: Mean Velocity Profile ---
    ax.plot(
        mesh,
        mms_solution,
        color="grey",
        linestyle="-",
        linewidth=1.0,
        label=f"MMS ({len(mesh)} pts)",
        zorder=2,
        alpha=0.8,
    )
    ax.plot(
        mesh,
        les_solution,
        color=color_les_1,
        linestyle="--",
        linewidth=1.3,
        marker="o",
        markevery=max(1, len(mesh) // 16),
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label=f"LES ({len(mesh)} pts)",
        zorder=3,
    )

    ax.set_title(
        r"MMS Solution Profile $w$",
        fontsize=11,
        pad=8,
        loc="left",
    )
    ax.set_xlabel(r"Spatial coordinate $z$", fontsize=11)
    ax.set_ylabel(r"Solution $w$", fontsize=11)
    ax.legend(
        loc="upper right",
        frameon=True,
        edgecolor="none",
        facecolor="white",
        framealpha=0.9,
    )

    score_text1 = (
        f"$L_2$: {mean_l2:.2e} | RMSE: {mean_rmse:.2e} | Rel. Err: {mean_rel:.1f}%"
    )
    ax.text(
        0.5,
        0.03,
        score_text1,
        transform=ax.transAxes,
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
    save_path = save_dir / f"MMS_comparison_{len(mesh):.2f}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    save_path = save_dir / f"MMS_comparison_{len(mesh):.2f}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path.name}")
