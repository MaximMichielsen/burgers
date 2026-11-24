import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# --- Publication-quality global settings ---
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
    }
)

STYLE_MARKER_BASE = dict(
    color="#ca1670",
    marker="d",
    markersize=7,
    markerfacecolor="white",
    markeredgewidth=1.3,
    zorder=4,
    linestyle="",
)

STYLE_MARKER_OPT = dict(
    color="#16CA70",
    marker="D",
    markersize=5.5,
    markerfacecolor="white",
    markeredgewidth=1.3,
    zorder=4,
    linestyle="",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POD_DNS_DIR = PROJECT_ROOT / "runs" / "dns" / "pod_dns"
DEFAULT_SWEEP_DIR = (
    PROJECT_ROOT / "runs" / "les_sweeps" / "diagonal" / "n33_dt0.001_vms_three_base"
)
Z_DOMAIN = (0.0, 2.0)

# Matches coefficient run folders such as "c_0.9" (also "c_1" or "c_1.0").
COEFF_DIR_REGEX = r"^c_(\d+(?:\.\d+)?)$"

# NOTE: the file patterns below are assumed to match the naming used in the old
# LES run folders. Adjust them if the files in mean_profiles/ or
# reynolds_profiles/ are named differently.
SWEEP_CONFIG = {
    "mean": {
        "profile_dir": "mean_profiles",
        "file_pattern": "mean_w_*.npy",
        "file_regex": r"mean_w_(\d+\.\d+)\.npy$",
        "property_label": r"$\langle w \rangle$",
        "profile_title": "Mean Velocity Profiles",
        "output_prefix": "mean_profile_sweep",
        "output_subdir": "p_2_dt_0.001",
    },
    "reynolds": {
        "profile_dir": "reynolds_profiles",
        "file_pattern": "reynolds_stress_profile_*.npy",
        "file_regex": r"reynolds_stress_profile_(\d+\.\d+)\.npy$",
        "property_label": r"$\langle w'w' \rangle$",
        "profile_title": "Reynolds Stress Profiles",
        "output_prefix": "reynolds_profile_sweep",
        "output_subdir": "",
    },
}


def _frame_time(profile_type: str, filename: str) -> float | None:
    """Parse the simulation time from a profile frame filename."""
    match = re.search(SWEEP_CONFIG[profile_type]["file_regex"], filename)
    return float(match.group(1)) if match else None


def _find_dns_reference(
    profile_type: str,
    frame_time: float,
    tol: float = 0.05,
) -> Path | None:
    """Find the POD DNS profile frame closest in time to the LES frame."""
    config = SWEEP_CONFIG[profile_type]
    profile_dir = POD_DNS_DIR / config["profile_dir"]

    if not profile_dir.exists():
        print(f"[Warning] POD DNS directory {profile_dir} does not exist.")
        return None

    candidates = []
    for path in profile_dir.glob(config["file_pattern"]):
        t = _frame_time(profile_type, path.name)
        if t is not None:
            candidates.append((abs(t - frame_time), path))

    if not candidates:
        print(
            f"[Warning] No files matching {config['file_pattern']!r} in {profile_dir}."
        )
        return None

    diff, path = min(candidates, key=lambda c: c[0])
    if diff > tol:
        print(
            f"[Warning] Closest POD DNS frame ({path.name}) is {diff:.4f} "
            f"from t={frame_time:.4f}, exceeding tol={tol}."
        )
        return None
    return path


def load_diagonal_sweep_data(
    profile_type: str,
    base_sweep_dir: Path | None = None,
    frame_time: float | None = None,
) -> tuple[dict, Path] | tuple[None, None]:
    """
    Load the diagonal-sweep runs (one c_<coeff> folder per coefficient).

    The frame to plot is chosen as follows:
      - if frame_time is given: the available frame closest to it;
      - otherwise: the latest frame that exists for every coefficient.
    """
    if profile_type not in SWEEP_CONFIG:
        raise ValueError(
            f"Unknown profile_type {profile_type!r}; expected 'mean' or 'reynolds'."
        )

    config = SWEEP_CONFIG[profile_type]
    if base_sweep_dir is None:
        base_sweep_dir = DEFAULT_SWEEP_DIR

    if not base_sweep_dir.exists():
        print(f"[Warning] Directory {base_sweep_dir} does not exist.")
        return None, None

    coeff_to_dir = {}
    for coeff_dir in base_sweep_dir.iterdir():
        if not coeff_dir.is_dir():
            continue
        match = re.match(COEFF_DIR_REGEX, coeff_dir.name)
        if match:
            coeff_to_dir[float(match.group(1))] = coeff_dir

    if not coeff_to_dir:
        print(
            f"[Warning] No coefficient directories (c_<value>) found under "
            f"{base_sweep_dir}."
        )
        return None, None

    # frame filename -> {coeff: path}
    frame_files: dict[str, dict[float, Path]] = {}
    for coeff, coeff_dir in coeff_to_dir.items():
        profile_dir = coeff_dir / config["profile_dir"]
        if not profile_dir.exists():
            print(f"[Warning] {profile_dir} does not exist.")
            continue
        for profile_path in profile_dir.glob(config["file_pattern"]):
            if _frame_time(profile_type, profile_path.name) is not None:
                frame_files.setdefault(profile_path.name, {})[coeff] = profile_path

    if not frame_files:
        print(
            f"[Warning] No {profile_type} profile frames matching "
            f"{config['file_pattern']!r} found under {base_sweep_dir}."
        )
        return None, None

    times = {name: _frame_time(profile_type, name) for name in frame_files}

    if frame_time is not None:
        filename = min(frame_files, key=lambda n: abs(times[n] - frame_time))
    else:
        n_coeffs = len(coeff_to_dir)
        complete = [n for n in frame_files if len(frame_files[n]) == n_coeffs]
        if complete:
            filename = max(complete, key=lambda n: times[n])
        else:
            filename = max(frame_files, key=lambda n: times[n])
            print(
                "[Warning] No frame exists for every coefficient; using the latest "
                f"frame ({filename}) with missing coefficients skipped."
            )

    sweep = {
        "coeffs": np.array(sorted(coeff_to_dir)),
        "frame_files": frame_files,
        "filename": filename,
        "frame_time": times[filename],
    }
    return sweep, base_sweep_dir


def _plot_diagonal_sweep(
    sweep: dict,
    profile_type: str,
    output_dir: Path,
    z_domain: tuple[float, float],
) -> None:
    """Create and save the profile-comparison and coefficient-error panels."""
    config = SWEEP_CONFIG[profile_type]
    coeffs = sweep["coeffs"]
    frame_files = sweep["frame_files"]
    filename = sweep["filename"]
    frame_time = sweep["frame_time"]

    dns_path = _find_dns_reference(profile_type, frame_time)
    if dns_path is None:
        print(
            f"[Warning] Missing POD DNS {profile_type} reference for "
            f"t={frame_time:.4f}. Skipping."
        )
        return

    dns_profile = np.load(dns_path)
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(dns_profile))

    norm = plt.Normalize(vmin=coeffs.min(), vmax=coeffs.max())
    cmap = cm.viridis

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10, 3.5),
        dpi=300,
        gridspec_kw={"width_ratios": [1.3, 1]},
    )

    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
        ax.minorticks_on()
        ax.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
            length=4,
            width=0.8,
        )
        ax.tick_params(which="minor", length=2, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    ax1.set_xlim(z_domain)
    (dns_line,) = ax1.plot(
        mesh_dns,
        dns_profile,
        color="black",
        linestyle="-",
        linewidth=1.8,
        label="POD DNS Ref.",
        zorder=5,
    )

    l2_errors = []
    for coeff in coeffs:
        profile_path = frame_files[filename].get(coeff)
        if profile_path is None:
            continue

        les_profile = np.load(profile_path)
        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_profile))
        dns_interp = np.interp(mesh_les, mesh_dns, dns_profile)
        rel_err = (
            np.linalg.norm(les_profile - dns_interp)
            / (np.linalg.norm(dns_interp) + 1e-12)
        ) * 100.0
        l2_errors.append((coeff, rel_err))

        ax1.plot(
            mesh_les,
            les_profile,
            color=cmap(norm(coeff)),
            alpha=0.60,
            linewidth=1.1,
            zorder=3,
        )

    time_str = rf"$t = {frame_time:.2f}$"
    ax1.set_title(
        rf"(a) {config['profile_title']} {config['property_label']} | {time_str}",
        fontsize=10,
        pad=6,
        loc="left",
    )
    ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=9)
    ax1.set_ylabel(config["property_label"], fontsize=10)

    y_min, y_max = ax1.get_ylim()
    ax1.set_ylim(y_min, y_max * 1.15)

    ax1.legend(
        handles=[dns_line],
        labels=[dns_line.get_label()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=False,
        fontsize=8.5,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", pad=0.03, shrink=0.85)
    cbar.set_label("Coefficients", fontsize=9)

    opt_info = None
    base_info = None

    if l2_errors:
        coeffs_arr, errs_arr = zip(*l2_errors)
        ax2.plot(
            coeffs_arr,
            errs_arr,
            color="#0072B2",
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            linestyle="-",
            label="Error",
            markevery=2,
        )

        min_idx = int(np.argmin(errs_arr))
        opt_coeff = coeffs_arr[min_idx]
        opt_err = errs_arr[min_idx]
        opt_info = (opt_coeff, opt_err)
        ax2.plot(
            opt_coeff,
            opt_err,
            label=f"Opt. ({opt_coeff:.1f}): {opt_err:.1f}%",
            **STYLE_MARKER_OPT,
        )

        coeffs_list = list(coeffs_arr)
        baseline_indices = [
            i for i, value in enumerate(coeffs_list) if np.isclose(value, 1.0)
        ]
        if baseline_indices:
            base_err = errs_arr[baseline_indices[0]]
            base_info = base_err
            ax2.plot(
                1.0,
                base_err,
                label=f"Baseline (1.0): {base_err:.1f}%",
                **STYLE_MARKER_BASE,
            )

        ax2.set_xlim(coeffs.min() - 0.2, coeffs.max() + 0.2)
        err_min, err_max = min(errs_arr), max(errs_arr)
        padding = (err_max - err_min) * 0.15 if err_max != err_min else 2.0
        ax2.set_ylim(max(0, err_min - padding), err_max + padding)

    ax2.set_title(
        r"(b) Relative Error vs. External Coefficients",
        fontsize=10,
        pad=6,
        loc="left",
    )
    ax2.set_xlabel(r"External Coefficients $(c_1 = c_2)$", fontsize=9)
    ax2.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=10)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="#0072B2",
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.4,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Error curve",
        )
    ]
    if opt_info is not None:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                label=f"Opt. ({opt_info[0]:.1f}): {opt_info[1]:.1f}%",
                **STYLE_MARKER_OPT,
            )
        )
    if base_info is not None:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                label=f"Baseline (1.0): {base_info:.1f}%",
                **STYLE_MARKER_BASE,
            )
        )

    ax2.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config['output_prefix']}_t_{frame_time:.2f}"
    output_path_png = output_dir / f"{stem}.png"
    output_path_pdf = output_dir / f"{stem}.pdf"

    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path_png.name} & .pdf (DNS ref: {dns_path.name})")


def create_diagonal_sweep_plots(
    profile_type: str = "mean",
    z_domain: tuple[float, float] = Z_DOMAIN,
    frame_time: float | None = None,
) -> None:
    """
    Generate the diagonal-sweep profile and L2-error plot for one profile type.

    frame_time: simulation time of the frame to plot. If None, the latest frame
    available for every coefficient is used.
    """
    sweep, base_sweep_dir = load_diagonal_sweep_data(
        profile_type, frame_time=frame_time
    )
    if sweep is None:
        print(f"[Error] Missing diagonal-sweep data for profile type: {profile_type}")
        return

    config = SWEEP_CONFIG[profile_type]
    output_dir = base_sweep_dir / "plots" / config["output_subdir"]

    print(
        f"[Info] Generating {profile_type} diagonal-sweep plot "
        f"(frame {sweep['filename']}, {len(sweep['coeffs'])} coefficients)..."
    )
    _plot_diagonal_sweep(
        sweep=sweep,
        profile_type=profile_type,
        output_dir=output_dir,
        z_domain=z_domain,
    )

    print(f"[Success] {profile_type.capitalize()} plot saved to: {output_dir}")


if __name__ == "__main__":
    create_diagonal_sweep_plots("mean")
    create_diagonal_sweep_plots("reynolds")
