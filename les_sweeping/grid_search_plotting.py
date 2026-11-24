from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, ListedColormap

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

# Colorblind-safe accents (Okabe-Ito palette)
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

STYLE_OPTIMUM = dict(
    color=OKABE_ITO["vermilion"],
    linestyle="none",
    marker="o",
    markersize=5.0,
    markerfacecolor=OKABE_ITO["vermilion"],
    markeredgecolor="white",
    markeredgewidth=0.8,
    zorder=5,
    clip_on=False,
)

# --- Paths (same conventions as the univariate sweep script) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LES_SWEEPS_ROOT = PROJECT_ROOT / "runs" / "LES_sweeps"
DEFAULT_GRID_ROOT = LES_SWEEPS_ROOT / "grid"
DEFAULT_DNS_ROOT = PROJECT_ROOT / "runs" / "dns" / "dns_intervals_dt_0.001"
Z_DOMAIN = (0.0, 2.0)
FRAME_TOLERANCE = 0.05  # max allowed |t_found - t_requested| when matching frames

# Layout:
#   grid/<t0_X_tf_Y>/n<N>_dt<dt>_<mode>_<tau-model>_c<c1>_<c2>/
#       mean_profiles/ , reynolds_profiles/
INTERVAL_RE = re.compile(r"^t0_(?P<t_start>\d+(?:\.\d+)?)_tf_(?P<t_end>\d+(?:\.\d+)?)$")
RUN_COEFF_RE = re.compile(r"_c(?P<c1>\d+(?:\.\d+)?)_(?P<c2>\d+(?:\.\d+)?)$")

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
    """Return {frame_time: filepath} for a run folder."""
    config = PROFILE_CONFIG[profile_type]
    profile_dirs = sorted(p for p in run_dir.rglob(config["directory"]) if p.is_dir())
    if not profile_dirs:
        return {}

    frames = {}
    for profile_path in profile_dirs[0].glob(config["pattern"]):
        frame_time = _frame_time(profile_type, profile_path.name)
        if frame_time is not None:
            frames[frame_time] = profile_path
    return frames


def _collect_grid_data(
    interval_dir: Path,
    n_nodes: int,
    dt: float,
    simulation_mode: str,
    tau_key: str,
    profile_type: str,
) -> dict[tuple[float, float], dict[float, Path]]:
    """Return {(c1, c2): {frame_time: filepath}} for every run in one interval.

    The (c1, c2) pairs are parsed from the folder names, so the grid axes
    (including uneven spacing such as 1.00, 1.20, 1.50, 2.00, ...) are
    discovered from disk instead of being assumed.
    """
    run_prefix = f"n{n_nodes}_dt{dt:g}_{simulation_mode}_"
    data: dict[tuple[float, float], dict[float, Path]] = {}
    n_prefix_matches = 0

    for run_dir in sorted(interval_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(run_prefix):
            continue
        n_prefix_matches += 1

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

        # Keep runs without frames too (empty dict) so the report can name them.
        data[coefficients] = _find_profile_frames(run_dir, profile_type)

    if not data:
        found = sorted(p.name for p in interval_dir.iterdir() if p.is_dir())[:5]
        print(
            f"[Warning] No usable {profile_type} runs in {interval_dir}.\n"
            f"          Expected folders starting with '{run_prefix}' "
            f"({n_prefix_matches} matched the prefix).\n"
            f"          First folders found: {found}"
        )
    return data


def _find_dns_reference(
    dns_root: Path,
    profile_type: str,
    interval_dir: Path,
) -> Path | None:
    """Find the DNS reference for this exact interval (flat directory)."""
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


def _select_frame_time(
    data: dict[tuple[float, float], dict[float, Path]],
) -> float | None:
    """Pick the latest frame time that more than half of the runs have.

    Using the latest frame common to *every* run would let a single unfinished
    run drag the whole plot back to an earlier time, and using the latest frame
    of *any* run would let one stray long run discard everything else. Runs
    lacking the chosen frame are reported as missing instead.
    """
    counts: dict[float, int] = {}
    for frames in data.values():
        for frame_time in frames:
            counts[frame_time] = counts.get(frame_time, 0) + 1
    if not counts:
        return None
    majority = [t for t, n in counts.items() if n > len(data) / 2]
    if majority:
        return max(majority)
    return max(counts, key=lambda frame_time: (counts[frame_time], frame_time))


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


def _grid_errors(
    data: dict[tuple[float, float], dict[float, Path]],
    frame_time: float,
    dns_profile: np.ndarray,
    mesh_dns: np.ndarray,
    z_domain: tuple[float, float],
) -> tuple[dict[tuple[float, float], float], dict[tuple[float, float], str]]:
    """Return ({(c1, c2): relative L2 error (%)}, {(c1, c2): reason it was unusable})."""
    errors: dict[tuple[float, float], float] = {}
    problems: dict[tuple[float, float], str] = {}
    for coefficients, frames in data.items():
        if not frames:
            problems[coefficients] = "run folder found, but no profile frames inside"
            continue
        selected_time = min(frames, key=lambda value: abs(value - frame_time))
        # Avoid silently comparing substantially different output times.
        if abs(selected_time - frame_time) > FRAME_TOLERANCE:
            problems[coefficients] = (
                f"closest frame t={selected_time} is not within {FRAME_TOLERANCE} "
                f"of t={frame_time} (run ended early?)"
            )
            continue
        profile = np.load(frames[selected_time])
        error = _relative_l2_error(profile, dns_profile, mesh_dns, z_domain)
        if not np.isfinite(error):
            problems[coefficients] = (
                f"error is not finite (NaN/inf in {frames[selected_time].name}; "
                "run diverged?)"
            )
            continue
        errors[coefficients] = error
    return errors, problems


def _report_missing(
    c1_grid: np.ndarray,
    c2_grid: np.ndarray,
    z_grid: np.ndarray,
    problems: dict[tuple[float, float], str],
    max_lines: int = 60,
) -> None:
    """Print every grid cell without a value, with the reason."""
    missing = np.argwhere(np.isnan(z_grid))
    if missing.size == 0:
        return
    print(f"[Missing] {len(missing)} grid cell(s) have no value (shown as white gaps):")
    for n, (j, i) in enumerate(missing):
        if n >= max_lines:
            print(f"    ... and {len(missing) - max_lines} more")
            break
        key = (float(c1_grid[j, i]), float(c2_grid[j, i]))
        reason = problems.get(
            key, "no matching run folder (wrong prefix/tau, or not run)"
        )
        print(f"    c1={key[0]:.2f}, c2={key[1]:.2f}: {reason}")


def build_grid_matrix(
    errors: dict[tuple[float, float], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrange {(c1, c2): value} into meshgrid arrays, indexed as Z[j, i].

    Axes are the sorted unique c1 / c2 values found, so uneven spacing is fine.
    Missing (c1, c2) combinations become NaN (gaps in the contour).
    """
    if not errors:
        raise ValueError("No errors to arrange into a grid.")

    c1_vals = np.array(sorted({c1 for c1, _ in errors}))
    c2_vals = np.array(sorted({c2 for _, c2 in errors}))
    c1_grid, c2_grid = np.meshgrid(c1_vals, c2_vals)
    z_grid = np.full(c1_grid.shape, np.nan, dtype=float)

    c1_index = {value: i for i, value in enumerate(c1_vals)}
    c2_index = {value: j for j, value in enumerate(c2_vals)}
    for (c1, c2), value in errors.items():
        z_grid[c2_index[c2], c1_index[c1]] = value

    return c1_grid, c2_grid, z_grid


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _desaturate_cmap(
    cmap: str | Colormap,
    saturation: float = 0.55,
    white_mix: float = 0.22,
) -> Colormap:
    """Scale HSV saturation and optionally blend toward white for a colormap."""
    if not (0.0 <= saturation <= 1.0):
        raise ValueError(f"saturation must be in [0.0, 1.0], got {saturation}.")
    if not (0.0 <= white_mix <= 1.0):
        raise ValueError(f"white_mix must be in [0.0, 1.0], got {white_mix}.")

    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if np.isclose(saturation, 1.0) and np.isclose(white_mix, 0.0):
        return base

    n_colors = getattr(base, "N", 256)
    rgba = base(np.linspace(0.0, 1.0, n_colors)).copy()
    alpha = rgba[:, 3:4]

    if not np.isclose(saturation, 1.0):
        hsv = mcolors.rgb_to_hsv(rgba[:, :3])
        hsv[:, 1] = np.clip(hsv[:, 1] * saturation, 0.0, 1.0)
        rgba[:, :3] = mcolors.hsv_to_rgb(hsv)

    if not np.isclose(white_mix, 0.0):
        rgba[:, :3] = (1.0 - white_mix) * rgba[:, :3] + white_mix * 1.0

    rgba[:, 3:4] = alpha
    name = f"{base.name}_sat{saturation:.2f}_w{white_mix:.2f}"
    return ListedColormap(rgba, name=name)


def _resolve_upper_bound(
    z_grid: np.ndarray,
    vmin: float,
    vmax: float | None,
    vmax_percentile: float | None,
) -> tuple[float, str]:
    """Choose a general upper contour bound across varying error scales."""
    z_max = float(np.nanmax(z_grid))
    candidate = z_max

    if vmax is not None:
        candidate = float(vmax)
    elif vmax_percentile is not None:
        if not (0.0 < vmax_percentile <= 100.0):
            raise ValueError(
                f"vmax_percentile must be in (0, 100], got {vmax_percentile}."
            )
        candidate = float(np.nanpercentile(z_grid, vmax_percentile))

    if candidate > vmin and candidate < z_max - 1e-12:
        return candidate, "max"
    return z_max, "neither"


def plot_grid_contour(
    c1_grid: np.ndarray,
    c2_grid: np.ndarray,
    z_grid: np.ndarray,
    output_path: Path | None = None,
    cmap: str | Colormap = "viridis",
    saturation: float = 0.55,
    white_mix: float = 0.22,
    levels: int | np.ndarray = 12,
    vmax: float | None = None,
    vmax_percentile: float | None = 75.0,
    cbar_label: str = r"Error Metric",
    title: str | None = None,
    baseline_point: tuple[float, float] | None = None,
    mark_minimum: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a publication-style filled + line contour plot for the 2D coefficient sweep."""
    finite = np.isfinite(z_grid)
    if not finite.any():
        raise ValueError("z_grid contains no finite values; nothing to plot.")
    if c1_grid.shape[0] < 2 or c1_grid.shape[1] < 2:
        raise ValueError(
            f"Contour plot needs at least 2 values on each axis, got grid {c1_grid.shape}."
        )
    if np.isclose(np.nanmin(z_grid), np.nanmax(z_grid)):
        raise ValueError("z_grid is constant; contour levels cannot be drawn.")

    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=300)

    # Mask NaNs explicitly so missing runs show up as gaps.
    z_masked = np.ma.masked_invalid(z_grid)
    active_cmap = _desaturate_cmap(cmap, saturation=saturation, white_mix=white_mix)

    vmin = float(np.nanmin(z_grid))
    z_upper, extend = _resolve_upper_bound(z_grid, vmin, vmax, vmax_percentile)

    if isinstance(levels, int):
        contour_levels = np.linspace(vmin, z_upper, levels + 1)
    else:
        contour_levels = np.asarray(levels, dtype=float)

    cf = ax.contourf(
        c1_grid,
        c2_grid,
        z_masked,
        levels=contour_levels,
        cmap=active_cmap,
        extend=extend,
    )
    # Reuse the filled-contour levels so lines and fills always agree.
    cs = ax.contour(
        c1_grid,
        c2_grid,
        z_masked,
        levels=contour_levels,
        colors="black",
        linewidths=0.5,
        alpha=0.45,
    )

    if mark_minimum:
        min_idx = np.nanargmin(z_grid)
        j_min, i_min = np.unravel_index(min_idx, z_grid.shape)
        c1_opt = c1_grid[j_min, i_min]
        c2_opt = c2_grid[j_min, i_min]
        z_opt = z_grid[j_min, i_min]
        ax.plot(
            [c1_opt],
            [c2_opt],
            label=f"Min ({c1_opt:.2f}, {c2_opt:.2f}): {z_opt:.3g}%",
            **STYLE_OPTIMUM,
        )

    ax.set_xlabel(r"$c_1$")
    ax.set_ylabel(r"$c_2$")
    if title:
        ax.set_title(title, pad=8)

    cbar = fig.colorbar(cf, ax=ax, pad=0.025)
    cbar.set_label(cbar_label, size=9.5)
    cbar.ax.tick_params(labelsize=8.5, direction="in")
    cbar.outline.set_linewidth(0.8)

    if mark_minimum:
        ax.legend(
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="0.8",
            framealpha=0.92,
        )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def _plot_interval(
    interval_dir: Path,
    profile_type: str,
    dns_root: Path,
    n_nodes: int,
    dt: float,
    simulation_mode: str,
    tau_key: str,
    z_domain: tuple[float, float],
    levels: int,
    cmap: str | Colormap,
    saturation: float,
    white_mix: float,
    vmax: float | None,
    vmax_percentile: float | None,
) -> None:
    config = PROFILE_CONFIG[profile_type]

    data = _collect_grid_data(
        interval_dir, n_nodes, dt, simulation_mode, tau_key, profile_type
    )
    if not data:
        return

    frame_time = _select_frame_time(data)
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

    errors, problems = _grid_errors(data, frame_time, dns_profile, mesh_dns, z_domain)
    if not errors:
        print(f"[Warning] No {profile_type} errors computed in {interval_dir.name}")
        return

    c1_grid, c2_grid, z_grid = build_grid_matrix(errors)
    _report_missing(c1_grid, c2_grid, z_grid, problems)
    n_missing = int(np.isnan(z_grid).sum())
    print(
        f"[Info] {interval_dir.name} {profile_type}: grid "
        f"{c1_grid.shape[1]} (c1) x {c1_grid.shape[0]} (c2), "
        f"{len(errors)} runs, {n_missing} missing, frame t={frame_time}"
    )

    output_dir = interval_dir.parent / "plots"
    stem = f"grid_contour_error_{profile_type}_t_{frame_time:.2f}"
    png_path = output_dir / f"{stem}.png"

    fig, _ = plot_grid_contour(
        c1_grid=c1_grid,
        c2_grid=c2_grid,
        z_grid=z_grid,
        output_path=png_path,
        cmap=cmap,
        saturation=saturation,
        white_mix=white_mix,
        levels=levels,
        vmax=vmax,
        vmax_percentile=vmax_percentile,
        cbar_label=r"Relative $L_2$ Error (%)",
        title=None,
        baseline_point=None,
        mark_minimum=True,
    )
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}  (DNS ref: {dns_path.name})")


def create_all_grid_plots(
    grid_root: Path = DEFAULT_GRID_ROOT,
    dns_root: Path = DEFAULT_DNS_ROOT,
    n_nodes: int = N_NODES_LES,
    dt: float = DT_LES,
    simulation_mode: str = "vms",
    tau_key: str = "tau2",
    z_domain: tuple[float, float] = Z_DOMAIN,
    profile_types: tuple[str, ...] = ("mean", "reynolds"),
    levels: int = 12,
    cmap: str | Colormap = "viridis",
    saturation: float = 0.55,
    white_mix: float = 0.22,
    vmax: float | None = None,
    vmax_percentile: float | None = 75.0,
) -> None:
    """Plot the c1/c2 grid-search error contours for every available interval."""
    if not grid_root.is_dir():
        raise FileNotFoundError(f"Grid root does not exist: {grid_root}")

    intervals = sorted(
        path
        for path in grid_root.iterdir()
        if path.is_dir() and INTERVAL_RE.match(path.name)
    )
    if not intervals:
        raise FileNotFoundError(
            f"No t0_<start>_tf_<end> interval folders in {grid_root}"
        )

    for interval_dir in intervals:
        for profile_type in profile_types:
            _plot_interval(
                interval_dir,
                profile_type,
                dns_root,
                n_nodes,
                dt,
                simulation_mode,
                tau_key,
                z_domain,
                levels,
                cmap,
                saturation,
                white_mix,
                vmax,
                vmax_percentile,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot c1/c2 grid-search error contours for every time interval."
    )
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID_ROOT)
    parser.add_argument("--dns-root", type=Path, default=DEFAULT_DNS_ROOT)
    parser.add_argument("--nodes", type=int, default=N_NODES_LES)
    parser.add_argument("--dt", type=float, default=DT_LES)
    parser.add_argument("--mode", default="vms")
    parser.add_argument("--tau", choices=("tau2", "tau3"), default="tau2")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_CONFIG),
        default=list(PROFILE_CONFIG),
    )
    parser.add_argument("--levels", type=int, default=12)
    parser.add_argument("--cmap", default="plasma_r")
    parser.add_argument(
        "--saturation",
        type=float,
        default=0.7,
        help="HSV saturation multiplier in [0.0, 1.0] (default: 0.55).",
    )
    parser.add_argument(
        "--white-mix",
        type=float,
        default=0.10,
        help="Blend fraction toward white in [0.0, 1.0] (default: 0.22).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Explicit upper error cap (%%); overrides --vmax-percentile when set.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=75.0,
        help="Percentile of finite error values used as upper bound when --vmax is None (default: 75).",
    )
    args = parser.parse_args()

    create_all_grid_plots(
        grid_root=args.grid_root,
        dns_root=args.dns_root,
        n_nodes=args.nodes,
        dt=args.dt,
        simulation_mode=args.mode,
        tau_key=args.tau,
        profile_types=tuple(args.profiles),
        levels=args.levels,
        cmap=args.cmap,
        saturation=args.saturation,
        white_mix=args.white_mix,
        vmax=args.vmax,
        vmax_percentile=args.vmax_percentile,
    )


if __name__ == "__main__":
    main()
