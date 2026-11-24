from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from setup.constants import DT_LES, N_NODES_LES

# --- Publication-Quality Global Settings (Direct Times New Roman + STIX Math) ---
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


STYLE_BASELINE = dict(
    color="#0072B2",
    linestyle="-.",
    marker="^",
    markersize=3,
    markerfacecolor="white",
    markeredgewidth=0.9,
    zorder=1,
    markevery=2,
)

STYLE_2 = dict(
    color="#D55E00",
    linestyle="--",
    marker="o",
    markersize=3,
    markerfacecolor="white",
    markeredgewidth=0.9,
    zorder=2,
    markevery=2,
)

STYLE_3 = dict(
    color="#12b062",
    linestyle="-.",
    marker="v",
    markersize=3,
    markerfacecolor="white",
    markeredgewidth=0.9,
    zorder=3,
    markevery=2,
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

STYLE_GALERKIN = dict(
    color="0.25",
    linestyle="--",
    linewidth=1.3,
    zorder=0,
)

STYLES = (STYLE_BASELINE, STYLE_2, STYLE_3)

# --- Paths (same conventions as the diagonal sweep script) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LES_SWEEPS_ROOT = PROJECT_ROOT / "runs" / "LES_sweeps"
DEFAULT_SWEEP_ROOT = LES_SWEEPS_ROOT / "univariate"
DEFAULT_GALERKIN_ROOT = LES_SWEEPS_ROOT / "galerkin"
DEFAULT_DNS_ROOT = PROJECT_ROOT / "runs" / "dns" / "dns_intervals_dt_0.001"
Z_DOMAIN = (0.0, 2.0)
NO_MODEL_SUFFIX = "no_model"  # pure Galerkin run folder: n<nodes>_dt<dt>_no_model
FRAME_TOLERANCE = 0.05  # max allowed |t_found - t_requested| when matching frames

# Layout:
#   univariate/<t0_X_tf_Y>/<vary_c1|vary_c2>/n<N>_dt<dt>_<mode>_<tau-model>_c<c1>_<c2>/
#       mean_profiles/ , reynolds_profiles/
INTERVAL_RE = re.compile(r"^t0_(?P<t_start>\d+(?:\.\d+)?)_tf_(?P<t_end>\d+(?:\.\d+)?)$")
RUN_COEFF_RE = re.compile(r"_c(?P<c1>\d+(?:\.\d+)?)_(?P<c2>\d+(?:\.\d+)?)$")

# (sweep folder, index of varied coefficient, varied name, fixed name)
SWEEPS = (
    ("vary_c1", 0, "c_1", "c_2"),
    ("vary_c2", 1, "c_2", "c_1"),
)

PROFILE_CONFIG = {
    "mean": {
        "directory": "mean_profiles",
        "pattern": "mean_w_*.npy",
        "regex": re.compile(r"mean_w_(\d+(?:\.\d+)?)\.npy$"),
        "label": r"$\langle w \rangle$",
    },
    "reynolds": {
        "directory": "reynolds_profiles",
        "pattern": "reynolds_stress_profile_*.npy",
        "regex": re.compile(r"reynolds_stress_profile_(\d+(?:\.\d+)?)\.npy$"),
        "label": r"$\langle w'w' \rangle$",
    },
}


# ----------------------------------------------------------------------------
# Data retrieval
# ----------------------------------------------------------------------------
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


def _find_profile_frames(run_dir: Path, profile_type: str) -> dict[float, Path]:
    """Return {frame_time: filepath} for a run folder.

    Works whether the profile folder sits directly in the run folder or in a
    subfolder.
    """
    config = PROFILE_CONFIG[profile_type]
    profile_dirs = sorted(p for p in run_dir.rglob(config["directory"]) if p.is_dir())
    if not profile_dirs:
        return {}
    if len(profile_dirs) > 1:
        print(
            f"[Warning] {len(profile_dirs)} '{config['directory']}' folders under "
            f"{run_dir}; using {profile_dirs[0]}"
        )

    frames = {}
    for profile_path in profile_dirs[0].glob(config["pattern"]):
        frame_time = _frame_time(profile_type, profile_path.name)
        if frame_time is not None:
            frames[frame_time] = profile_path
    return frames


def _collect_sweep_data(
    interval_dir: Path,
    sweep_tag: str,
    varied_index: int,
    n_nodes: int,
    dt: float,
    simulation_mode: str,
    tau_key: str,
    profile_type: str,
    fixed_coeff_vals: tuple[float, ...],
) -> dict[float, dict[float, dict[float, Path]]]:
    """Return {fixed_value: {varied_value: {frame_time: filepath}}} for one sweep.

    Reads <interval_dir>/<sweep_tag>/n<N>_dt<dt>_<mode>_..._c<c1>_<c2>/.
    """
    sweep_dir = interval_dir / sweep_tag
    if not sweep_dir.is_dir():
        print(f"[Warning] Sweep folder not found: {sweep_dir}")
        return {}

    data: dict[float, dict[float, dict[float, Path]]] = {}
    run_prefix = f"n{n_nodes}_dt{dt:g}_{simulation_mode}_"
    fixed_index = 1 - varied_index

    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(run_prefix):
            continue

        run_tau = _classify_tau_folder(run_dir.name)
        if run_tau is None:
            print(f"[Warning] Could not identify tau model from {run_dir.name}")
            continue
        if run_tau != tau_key:
            continue

        coeff_match = RUN_COEFF_RE.search(run_dir.name)
        if coeff_match is None:
            print(f"[Warning] Could not parse coefficients from {run_dir.name}")
            continue
        coefficients = (float(coeff_match.group("c1")), float(coeff_match.group("c2")))

        fixed_value = next(
            (v for v in fixed_coeff_vals if np.isclose(v, coefficients[fixed_index])),
            None,
        )
        if fixed_value is None:
            continue

        frames = _find_profile_frames(run_dir, profile_type)
        if frames:
            data.setdefault(fixed_value, {})[coefficients[varied_index]] = frames

    return data


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
    filename = (
        f"w_mean_{t_start}_{t_end}.npy"
        if profile_type == "mean"
        else f"ww_{t_start}_{t_end}.npy"
    )
    reference_path = dns_root / filename
    return reference_path if reference_path.is_file() else None


def _select_frame_time(sweeps: list[dict]) -> float | None:
    """Prefer the latest frame available for every run in every sweep."""
    frame_sets = [
        set(frames)
        for sweep in sweeps
        for varied_dict in sweep.values()
        for frames in varied_dict.values()
    ]
    if not frame_sets:
        return None

    common = set.intersection(*frame_sets)
    if common:
        return max(common)

    # If output times are not identical, use the latest time present in any run.
    return max(set.union(*frame_sets))


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


def _sweep_errors(
    sweep: dict[float, dict[float, dict[float, Path]]],
    frame_time: float,
    dns_profile: np.ndarray,
    mesh_dns: np.ndarray,
    z_domain: tuple[float, float],
) -> dict[float, dict[float, float]]:
    """Return {fixed_value: {varied_value: relative L2 error (%)}}."""
    errors: dict[float, dict[float, float]] = {}
    for fixed_value, varied_dict in sweep.items():
        for varied_value, frames in varied_dict.items():
            selected_time = min(frames, key=lambda value: abs(value - frame_time))
            # Avoid silently comparing substantially different output times.
            if abs(selected_time - frame_time) > FRAME_TOLERANCE:
                continue
            profile = np.load(frames[selected_time])
            errors.setdefault(fixed_value, {})[varied_value] = _relative_l2_error(
                profile, dns_profile, mesh_dns, z_domain
            )
    return errors


def _galerkin_error(
    galerkin_root: Path,
    interval_dir: Path,
    n_nodes: int,
    dt: float,
    profile_type: str,
    frame_time: float,
    dns_profile: np.ndarray,
    mesh_dns: np.ndarray,
    z_domain: tuple[float, float],
) -> float | None:
    """Relative L2 error (%) of the pure Galerkin (no-model) run, or None if unavailable.

    Expected location:
        <galerkin_root>/<t0_X_tf_Y>/n<N>_dt<dt>_no_model/<profile folder>/
    """
    run_dir = (
        galerkin_root / interval_dir.name / f"n{n_nodes}_dt{dt:g}_{NO_MODEL_SUFFIX}"
    )
    if not run_dir.is_dir():
        print(f"[Warning] Galerkin folder not found: {run_dir}")
        return None

    frames = _find_profile_frames(run_dir, profile_type)
    if not frames:
        print(f"[Warning] No Galerkin {profile_type} frames under {run_dir}")
        return None

    selected_time = min(frames, key=lambda value: abs(value - frame_time))
    if abs(selected_time - frame_time) > FRAME_TOLERANCE:
        print(
            f"[Warning] Galerkin {profile_type} frame t={selected_time} does not match "
            f"sweep frame t={frame_time}; skipping Galerkin line"
        )
        return None

    profile = np.load(frames[selected_time])
    return _relative_l2_error(profile, dns_profile, mesh_dns, z_domain)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _draw_panel(
    ax,
    errors: dict[float, dict[float, float]],
    fixed_coeff_vals: tuple[float, ...],
    fixed_name: str,
    max_allowable_error: float,
) -> list[float]:
    """Draw one sweep panel; returns every plotted error (for axis scaling)."""
    all_errs: list[float] = []
    opt_info = None
    base_info = None

    for idx, fix_val in enumerate(fixed_coeff_vals):
        curve = errors.get(fix_val)
        if not curve:
            continue
        varied_sorted = sorted(curve)
        values = [curve[v] for v in varied_sorted]
        ax.plot(
            varied_sorted,
            values,
            label=rf"${fixed_name} = {fix_val:g}$",
            **STYLES[idx % len(STYLES)],
        )
        all_errs.extend(values)

        for varied_value, error in zip(varied_sorted, values):
            # The allowable-error threshold only applies to optimum selection.
            if error <= max_allowable_error and (
                opt_info is None or error < opt_info[1]
            ):
                opt_info = (varied_value, error)
            if np.isclose(fix_val, 1.0) and np.isclose(varied_value, 1.0):
                base_info = (varied_value, error)

    if opt_info is not None:
        ax.plot(*opt_info, label=f"Opt. {opt_info[1]:.1f}%", **STYLE_MARKER_OPT)
    if base_info is not None:
        ax.plot(*base_info, label=f"Base. {base_info[1]:.1f}%", **STYLE_MARKER_BASE)
    return all_errs


def _plot_interval(
    interval_dir: Path,
    profile_type: str,
    sweeps: list[dict],
    galerkin_root: Path,
    dns_root: Path,
    n_nodes: int,
    dt: float,
    fixed_coeff_vals: tuple[float, ...],
    max_allowable_error: float,
    z_domain: tuple[float, float],
    show_galerkin: bool,
) -> None:
    config = PROFILE_CONFIG[profile_type]

    frame_time = _select_frame_time(sweeps)
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

    galerkin_error = None
    if show_galerkin:
        galerkin_error = _galerkin_error(
            galerkin_root,
            interval_dir,
            n_nodes,
            dt,
            profile_type,
            frame_time,
            dns_profile,
            mesh_dns,
            z_domain,
        )

    # --- 1x2 figure and styling retained from the original script ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=300)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(
            which="both", direction="in", top=True, right=True, length=4, width=0.8
        )
        ax.tick_params(which="minor", length=2, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    for ax, panel, sweep, (_, _, varied_name, fixed_name) in zip(
        axes, "ab", sweeps, SWEEPS
    ):
        errors = _sweep_errors(sweep, frame_time, dns_profile, mesh_dns, z_domain)
        _draw_panel(ax, errors, fixed_coeff_vals, fixed_name, max_allowable_error)

        if galerkin_error is not None:
            ax.axhline(
                galerkin_error,
                label=f"Galerkin {galerkin_error:.1f}%",
                **STYLE_GALERKIN,
            )

        ax.set_title(
            rf"({panel}) {config['label']} Error vs. ${varied_name}$",
            fontsize=10,
            pad=6,
            loc="left",
        )
        ax.set_xlabel(rf"Coefficient ${varied_name}$", fontsize=9)
        ax.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=10)

        # External bottom legend (same placement and formatting as before).
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            frameon=False,
            fontsize=8.5,
        )

    output_dir = interval_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"univariate_sweep_error_lines_{profile_type}_t_{frame_time:.2f}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}  (DNS ref: {dns_path.name})")


def create_all_univariate_sweep_plots(
    sweep_root: Path = DEFAULT_SWEEP_ROOT,
    galerkin_root: Path = DEFAULT_GALERKIN_ROOT,
    dns_root: Path = DEFAULT_DNS_ROOT,
    n_nodes: int = 49,
    dt: float = 1e-3,
    simulation_mode: str = "vms",
    tau_key: str = "tau2",
    fixed_coeff_vals: tuple[float, ...] = (1.0, 2.0),
    max_allowable_error: dict[str, float] | None = None,
    z_domain: tuple[float, float] = Z_DOMAIN,
    profile_types: tuple[str, ...] = ("mean", "reynolds"),
    show_galerkin: bool = True,
) -> None:
    """Plot the c1/c2 univariate sweeps for every available interval."""
    if max_allowable_error is None:
        max_allowable_error = {"mean": 50.0, "reynolds": 12.0}

    if not sweep_root.is_dir():
        raise FileNotFoundError(f"Univariate sweep root does not exist: {sweep_root}")

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
        for profile_type in profile_types:
            sweeps = [
                _collect_sweep_data(
                    interval_dir,
                    sweep_tag,
                    varied_index,
                    n_nodes,
                    dt,
                    simulation_mode,
                    tau_key,
                    profile_type,
                    fixed_coeff_vals,
                )
                for sweep_tag, varied_index, _, _ in SWEEPS
            ]
            if not all(sweeps):
                print(
                    f"[Warning] Missing {profile_type} sweep data in "
                    f"{interval_dir.name} (vary_c1 found: {bool(sweeps[0])}, "
                    f"vary_c2 found: {bool(sweeps[1])}); skipping"
                )
                continue

            _plot_interval(
                interval_dir,
                profile_type,
                sweeps,
                galerkin_root,
                dns_root,
                n_nodes,
                dt,
                fixed_coeff_vals,
                max_allowable_error[profile_type],
                z_domain,
                show_galerkin,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot univariate (c1 / c2) sweep errors for every time interval."
    )
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--galerkin-root", type=Path, default=DEFAULT_GALERKIN_ROOT)
    parser.add_argument("--dns-root", type=Path, default=DEFAULT_DNS_ROOT)
    parser.add_argument("--nodes", type=int, default=N_NODES_LES)
    parser.add_argument("--dt", type=float, default=DT_LES)
    parser.add_argument("--mode", default="vms")
    parser.add_argument("--tau", choices=("tau2", "tau3"), default="tau2")
    parser.add_argument(
        "--fixed",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0],
        help="Values of the non-varied coefficient to draw as separate curves.",
    )
    parser.add_argument("--max-error-mean", type=float, default=50.0)
    parser.add_argument("--max-error-reynolds", type=float, default=12.0)
    parser.add_argument("--no-galerkin", action="store_true")
    args = parser.parse_args()

    create_all_univariate_sweep_plots(
        sweep_root=args.sweep_root,
        galerkin_root=args.galerkin_root,
        dns_root=args.dns_root,
        n_nodes=args.nodes,
        dt=args.dt,
        simulation_mode=args.mode,
        tau_key=args.tau,
        fixed_coeff_vals=tuple(args.fixed),
        max_allowable_error={
            "mean": args.max_error_mean,
            "reynolds": args.max_error_reynolds,
        },
        show_galerkin=not args.no_galerkin,
    )


if __name__ == "__main__":
    main()
