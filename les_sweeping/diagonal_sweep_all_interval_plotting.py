from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from setup.constants import N_NODES_LES, DT_LES, PLT_PARAMS, STYLE_VERMILLION, Z_LENGTH_DNS, PATH_TO_DNS_REF_PROFILES
from utils.profile_plotting import STYLE_BLUE

# Publication-quality defaults
plt.rcParams.update(PLT_PARAMS)

# Okabe-Ito colorblind-safe colors
TAU_STYLES = {
    "tau2": {
        "label": r"$\tau_2$ (two-parameter)",
        "color": STYLE_BLUE["color"],
        "cmap": "Blues",
    },
    "tau3": {
        "label": r"$\tau_3$ (three-parameter)",
        "color": STYLE_VERMILLION["color"],
        "cmap": "Oranges",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP_ROOT = PROJECT_ROOT / "runs" / "LES_sweeps" / "diagonal"
DEFAULT_DNS_ROOT = PATH_TO_DNS_REF_PROFILES
Z_DOMAIN = (0.0, Z_LENGTH_DNS)
NO_MODEL_SUFFIX = "no_model"  # pure Galerkin run folder: n<nodes>_dt<dt>_no_model

INTERVAL_RE = re.compile(r"^t0_(?P<t_start>\d+(?:\.\d+)?)_tf_(?P<t_end>\d+(?:\.\d+)?)$")
COEFF_RE = re.compile(r"^c_(\d+(?:\.\d+)?)$")
PROFILE_CONFIG = {
    "mean": {
        "directory": "mean_profiles",
        "pattern": "mean_w_*.npy",
        "regex": re.compile(r"mean_w_(\d+(?:\.\d+)?)\.npy$"),
        "label": r"$\langle w \rangle$",
        "title": "Mean Velocity Profile",
        "prefix": "mean_profile_sweep",
    },
    "reynolds": {
        "directory": "reynolds_profiles",
        "pattern": "reynolds_stress_profile_*.npy",
        "regex": re.compile(r"reynolds_stress_profile_(\d+(?:\.\d+)?)\.npy$"),
        "label": r"$\langle w'w' \rangle$",
        "title": "Reynolds Stress Profile",
        "prefix": "reynolds_profile_sweep",
    },
}


def _classify_tau_folder(name: str) -> str | None:
    """Identify a run folder generated from TauModel.TWO_PARAMS_BASE/THREE_PARAMS_BASE."""
    token = name.lower()
    if "three" in token or "3_params" in token or "3param" in token:
        return "tau3"
    if "two" in token or "2_params" in token or "2param" in token:
        return "tau2"
    return None


def _frame_time(profile_type: str, filename: str) -> float | None:
    match = PROFILE_CONFIG[profile_type]["regex"].search(filename)
    return float(match.group(1)) if match else None


def _collect_model_data(
    interval_dir: Path,
    n_nodes: int,
    dt: float,
    simulation_mode: str,
    profile_type: str,
) -> dict[str, dict]:
    """Return {tau2/tau3: {coeff: {frame_time: filepath}}} for one interval."""
    config = PROFILE_CONFIG[profile_type]
    model_data: dict[str, dict] = {"tau2": {}, "tau3": {}}
    run_prefix = f"n{n_nodes}_dt{dt:g}_{simulation_mode}_"

    for model_dir in interval_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith(run_prefix):
            continue
        tau_key = _classify_tau_folder(model_dir.name)
        if tau_key is None:
            print(f"[Warning] Could not identify tau model from {model_dir.name}")
            continue

        for coeff_dir in model_dir.iterdir():
            if not coeff_dir.is_dir():
                continue
            coeff_match = COEFF_RE.match(coeff_dir.name)
            if not coeff_match:
                continue
            coeff = float(coeff_match.group(1))
            profile_dir = coeff_dir / config["directory"]
            if not profile_dir.is_dir():
                continue

            frames = {}
            for profile_path in profile_dir.glob(config["pattern"]):
                frame_time = _frame_time(profile_type, profile_path.name)
                if frame_time is not None:
                    frames[frame_time] = profile_path
            if frames:
                model_data[tau_key][coeff] = frames

    return model_data


def _find_dns_reference(
    dns_root: Path,
    profile_type: str,
    interval_dir: Path,
) -> Path | None:
    """
    Find the DNS reference for this exact interval.

    The reference directory is flat:
        w_mean_<t_start>_<t_end>.npy
        ww_<t_start>_<t_end>.npy
    """
    interval_match = INTERVAL_RE.match(interval_dir.name)
    if interval_match is None:
        return None

    t_start = interval_match.group("t_start")
    t_end = interval_match.group("t_end")

    if profile_type == "mean":
        filename = f"w_mean_{t_start}_{t_end}.npy"
    else:
        filename = f"ww_{t_start}_{t_end}.npy"

    reference_path = dns_root / filename
    return reference_path if reference_path.is_file() else None


def _select_frame_time(model_data: dict[str, dict]) -> float | None:
    """Prefer the latest frame available for every coefficient in both models."""
    frame_sets = []
    for tau_key in ("tau2", "tau3"):
        for frames in model_data[tau_key].values():
            frame_sets.append(set(frames))

    if not frame_sets:
        return None

    common = set.intersection(*frame_sets) if frame_sets else set()
    if common:
        return max(common)

    # If output times are not identical, use the latest time present in either model.
    all_times = set.union(*frame_sets)
    return max(all_times) if all_times else None


def _relative_l2_error(
    profile: np.ndarray,
    dns_profile: np.ndarray,
    mesh_dns: np.ndarray,
    z_domain: tuple[float, float],
) -> float:
    """Relative L2 error (%) of a profile against the DNS reference."""
    mesh = np.linspace(z_domain[0], z_domain[1], len(profile))
    reference = np.interp(mesh, mesh_dns, dns_profile)
    return float(
        np.linalg.norm(profile - reference)
        / (np.linalg.norm(reference) + 1e-12)
        * 100.0
    )


def _galerkin_error(
    interval_dir: Path,
    n_nodes: int,
    dt: float,
    profile_type: str,
    frame_time: float,
    dns_profile: np.ndarray,
    mesh_dns: np.ndarray,
    z_domain: tuple[float, float],
) -> float | None:
    """Relative L2 error (%) of the pure Galerkin (no-model) run, or None if unavailable."""
    config = PROFILE_CONFIG[profile_type]
    time_dir = interval_dir.name
    galerkin_dir = interval_dir.parent.parent / "galerkin" / time_dir
    run_dir = galerkin_dir / f"n{n_nodes}_dt{dt:g}_{NO_MODEL_SUFFIX}"
    if not run_dir.is_dir():
        print(f"[Warning] Galerkin folder not found: {run_dir}")
        return None

    # Works whether the profiles sit directly in the run folder or in a subfolder.
    profile_dirs = sorted(p for p in run_dir.rglob(config["directory"]) if p.is_dir())
    if not profile_dirs:
        print(f"[Warning] No '{config['directory']}' folder under {run_dir}")
        return None
    if len(profile_dirs) > 1:
        print(
            f"[Warning] {len(profile_dirs)} '{config['directory']}' folders under "
            f"{run_dir}; using {profile_dirs[0]}"
        )

    frames = {}
    for profile_path in profile_dirs[0].glob(config["pattern"]):
        t = _frame_time(profile_type, profile_path.name)
        if t is not None:
            frames[t] = profile_path
    if not frames:
        print(f"[Warning] No {profile_type} frames in {profile_dirs[0]}")
        return None

    selected_time = min(frames, key=lambda value: abs(value - frame_time))
    if abs(selected_time - frame_time) > 0.05:
        print(
            f"[Warning] Galerkin {profile_type} frame t={selected_time} does not match "
            f"sweep frame t={frame_time}; skipping Galerkin line"
        )
        return None

    profile = np.load(frames[selected_time])
    return _relative_l2_error(profile, dns_profile, mesh_dns, z_domain)


def _profile_and_error(
    model_data: dict[str, dict],
    tau_key: str,
    frame_time: float,
    dns_profile: np.ndarray,
    mesh_dns: np.ndarray,
    z_domain: tuple[float, float],
) -> tuple[list, list]:
    """Return coefficient/profile pairs and coefficient/relative-error pairs."""
    profiles, errors = [], []
    for coeff, frames in sorted(model_data[tau_key].items()):
        if not frames:
            continue
        selected_time = min(frames, key=lambda value: abs(value - frame_time))
        # Avoid silently comparing substantially different output times.
        if abs(selected_time - frame_time) > 0.05:
            continue

        profile = np.load(frames[selected_time])
        mesh = np.linspace(z_domain[0], z_domain[1], len(profile))
        relative_error = _relative_l2_error(profile, dns_profile, mesh_dns, z_domain)
        profiles.append((coeff, mesh, profile))
        errors.append((coeff, relative_error))
    return profiles, errors


def _plot_interval_profile(
    interval_dir: Path,
    profile_type: str,
    model_data: dict[str, dict],
    dns_root: Path,
    z_domain: tuple[float, float],
    n_nodes: int,
    dt: float,
) -> None:
    config = PROFILE_CONFIG[profile_type]
    frame_time = _select_frame_time(model_data)
    if frame_time is None:
        print(f"[Warning] No {profile_type} profiles found in {interval_dir}")
        return

    dns_path = _find_dns_reference(dns_root, profile_type, interval_dir)
    if dns_path is None:
        print(
            f"[Warning] No DNS {profile_type} reference for "
            f"{interval_dir.name} in {dns_root}; skipping this interval"
        )
        return

    dns_profile = np.load(dns_path)
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(dns_profile))

    # Explicit layout: [profile | gap | cbar tau2 | label gap | cbar tau3] + error panel.
    # The wide gap after each colorbar leaves room for its tick labels and caption.
    fig = plt.figure(figsize=(10, 3.5), dpi=300)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.5, 1],
        wspace=0.2,
        left=0.06,
        right=0.985,
        bottom=0.14,
        top=0.90,
    )
    inner = outer[0].subgridspec(
        1, 6, width_ratios=[1.0, 0.025, 0.03, 0.10, 0.03, 0.0], wspace=0.0
    )
    ax_profile = fig.add_subplot(inner[0, 0])
    ax_error = fig.add_subplot(outer[1])
    cbar_slots = {"tau2": inner[0, 2], "tau3": inner[0, 4]}  # blue left, orange right
    for ax in (ax_profile, ax_error):
        ax.axhline(0, color="black", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    ax_profile.set_xlim(z_domain)
    ax_profile.plot(
        mesh_dns,
        dns_profile,
        color="0.25",
        linewidth=1.8,
        label="DNS reference",
        zorder=5,
    )

    error_results = {}
    colorbars = []
    for tau_key in ("tau2", "tau3"):
        style = TAU_STYLES[tau_key]
        profiles, errors = _profile_and_error(
            model_data, tau_key, frame_time, dns_profile, mesh_dns, z_domain
        )
        error_results[tau_key] = errors

        if not profiles:
            print(
                f"[Warning] No {tau_key} {profile_type} profiles in {interval_dir.name}"
            )
            continue

        coeffs = np.array([item[0] for item in profiles])
        cmap = mpl.colormaps[style["cmap"]]
        norm = mpl.colors.Normalize(vmin=float(coeffs.min()), vmax=float(coeffs.max()))
        for coeff, mesh, profile in profiles:
            # Keep the lightest shades visible against a white background.
            shade = 0.45 + 0.5 * norm(coeff)
            ax_profile.plot(
                mesh,
                profile,
                color=cmap(shade),
                linewidth=1.0,
                alpha=0.8,
                zorder=3 if tau_key == "tau2" else 2,
            )

        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        colorbars.append((tau_key, scalar_map, style["label"]))

        if errors:
            coeff_values, error_values = zip(*errors)
            ax_error.plot(
                coeff_values,
                error_values,
                color=style["color"],
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linewidth=1.5,
                label=style["label"],
            )
            best_index = int(np.argmin(error_values))
            ax_error.plot(
                coeff_values[best_index],
                error_values[best_index],
                marker="*",
                markersize=9,
                linestyle="",
                color=style["color"],
                markeredgecolor="black",
                markeredgewidth=0.35,
                zorder=6,
            )

    galerkin_error = _galerkin_error(
        interval_dir,
        n_nodes,
        dt,
        profile_type,
        frame_time,
        dns_profile,
        mesh_dns,
        z_domain,
    )
    if galerkin_error is not None:
        ax_error.axhline(
            galerkin_error,
            color="0.25",
            linestyle="--",
            linewidth=1.3,
            label="Galerkin (no model)",
            zorder=1,
        )

    interval_match = INTERVAL_RE.match(interval_dir.name)
    interval_label = (
        rf"$t \in [{float(interval_match.group('t_start')):.2f}, "
        rf"{float(interval_match.group('t_end')):.2f}]$"
        if interval_match
        else interval_dir.name
    )
    ax_profile.set_title(
        f"(a) {config['title']} {config['label']} | {interval_label}",
        loc="left",
        fontsize=9.5,
        pad=6,
    )
    ax_profile.set_xlabel(r"Wall-normal coordinate $z$")
    ax_profile.set_ylabel(config["label"])
    ax_profile.legend(loc="best", frameon=False, fontsize=8)

    # Two independent colorbars: blue shades for tau2, orange shades for tau3.
    for tau_key, scalar_map, label in colorbars:
        cax = fig.add_subplot(cbar_slots[tau_key])
        cbar = fig.colorbar(scalar_map, cax=cax, orientation="vertical")
        cbar.set_label(rf"Coefficient ({label})", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax_error.set_title(
        "(b) Relative Error vs. External Coefficient", loc="left", fontsize=9.5, pad=6
    )
    ax_error.set_xlabel(r"External coefficients $c$")
    ax_error.set_ylabel(r"Relative $L_2$ error (%)")
    ax_error.set_xlim(0.8, 8.2)
    ax_error.legend(loc="best", frameon=False, fontsize=8)

    # Keep error limits data-driven across both tau models.
    all_errors = [error for values in error_results.values() for _, error in values]
    if galerkin_error is not None:
        all_errors.append(galerkin_error)
    if all_errors:
        low, high = min(all_errors), max(all_errors)
        padding = 0.12 * (high - low) if high > low else max(0.5, 0.1 * high)
        ax_error.set_ylim(max(0.0, low - padding), high + padding)

    output_dir = interval_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config['prefix']}_t_{frame_time:.2f}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def create_all_diagonal_sweep_plots(
    sweep_root: Path = DEFAULT_SWEEP_ROOT,
    dns_root: Path = DEFAULT_DNS_ROOT,
    n_nodes: int = N_NODES_LES,
    dt: float = DT_LES,
    simulation_mode: str = "vms",
    z_domain: tuple[float, float] = Z_DOMAIN,
) -> None:
    """Plot both tau models and both profile types for every available interval."""
    if not sweep_root.is_dir():
        raise FileNotFoundError(f"Diagonal sweep root does not exist: {sweep_root}")

    intervals = sorted(
        path
        for path in sweep_root.iterdir()
        if path.is_dir() and INTERVAL_RE.match(path.name)
    )
    if not intervals:
        raise FileNotFoundError(
            f"No t0_<start>_tf_<end> interval folders in {sweep_root}"
        )

    for interval_dir in intervals:
        for profile_type in ("mean", "reynolds"):
            model_data = _collect_model_data(
                interval_dir, n_nodes, dt, simulation_mode, profile_type
            )
            if not model_data["tau2"] and not model_data["tau3"]:
                print(f"[Warning] No {profile_type} sweep data in {interval_dir}")
                continue
            _plot_interval_profile(
                interval_dir,
                profile_type,
                model_data,
                dns_root,
                z_domain,
                n_nodes,
                dt,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all diagonal sweeps for tau2/tau3 and every time interval."
    )
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--dns-root", type=Path, default=DEFAULT_DNS_ROOT)
    parser.add_argument("--nodes", type=int, default=N_NODES_LES)
    parser.add_argument("--dt", type=float, default=DT_LES)
    parser.add_argument("--mode", default="vms")
    args = parser.parse_args()

    create_all_diagonal_sweep_plots(
        sweep_root=args.sweep_root,
        dns_root=args.dns_root,
        n_nodes=args.nodes,
        dt=args.dt,
        simulation_mode=args.mode,
    )


if __name__ == "__main__":
    main()
