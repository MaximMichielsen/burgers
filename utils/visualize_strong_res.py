"""Interactive viewer for ``SolverBase2.strong_residual_history``.

Each frame is one Newton-Raphson iterate (time step ``step``, iteration
``iteration``). Top: strong residual at the Gauss points of that iterate (plus
backtracking line-search trials, if any). Middle: space-time map of every
iterate. Bottom: max|R| and RMS history, coloured by NR iteration.

Data layout assumption (checked against ``solver_base_2.py``):
``calculate_elemental_residual_jacobian`` appends one dict per Gauss point
and is only called from ``assemble_global_system``. Every call to the
assembler therefore adds ``n_elements * n_gauss`` consecutive entries, in
element order. Within one (step, iteration) the first assembly is the
current iterate; every later one is a line-search trial with
alpha = 0.5**(trial - 1).
"""

import itertools
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import (
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    NullFormatter,
)
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray

REQUIRED_COLUMNS = ("step", "iteration", "element", "gauss_point", "x", "residual")

HistorySource = Union[pd.DataFrame, Sequence[dict], str, Path]


# --------------------------------------------------------------------------- #
#  Data preparation
# --------------------------------------------------------------------------- #
def load_strong_residual_history(source: HistorySource) -> pd.DataFrame:
    """Accepts the list of dicts, a DataFrame, or a saved .npz / .csv file."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".npz":
            with np.load(path) as data:
                df = pd.DataFrame({key: data[key] for key in data.files})
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported file type {path.suffix!r} (use .npz or .csv)"
            )
    else:
        df = pd.DataFrame(list(source))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"History is missing columns: {missing}")
    if df.empty:
        raise ValueError("Strong-residual history is empty.")
    return df.reset_index(drop=True)


def tag_assembly_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ``call`` (global assembly index) and ``trial`` (0 = iterate,
    k >= 1 = k-th line-search trial of that iteration)."""
    df = df.copy()
    is_call_start = (df["element"] == df["element"].min()) & np.isclose(
        df["gauss_point"], df["gauss_point"].min()
    )
    df["call"] = is_call_start.cumsum() - 1
    first_call = df.groupby(["step", "iteration"])["call"].transform("min")
    df["trial"] = df["call"] - first_call
    return df


def _break_between_elements(values: NDArray, n_gp: int) -> NDArray:
    """Insert NaN after each element so lines are not drawn across the
    element-to-element jumps of the (discontinuous) strong residual."""
    n_full = (len(values) // n_gp) * n_gp
    grouped = values[:n_full].reshape(-1, n_gp)
    grouped = np.hstack([grouped, np.full((grouped.shape[0], 1), np.nan)])
    return grouped.ravel()


def build_frames(
    df: pd.DataFrame,
    last_n_steps: Optional[int] = None,
    stride: int = 1,
) -> list[dict]:
    """One frame per Newton iterate; line-search trials are attached to it."""
    n_gp = int(df["gauss_point"].round(12).nunique())

    call = df["call"].to_numpy()
    bounds = np.flatnonzero(np.diff(call)) + 1
    starts = np.r_[0, bounds]
    x_calls = np.split(df["x"].to_numpy(dtype=float), bounds)
    r_calls = np.split(df["residual"].to_numpy(dtype=float), bounds)
    step_c = df["step"].to_numpy()[starts]
    iter_c = df["iteration"].to_numpy()[starts]
    trial_c = df["trial"].to_numpy()[starts]

    frames: list[dict] = []
    for k in range(len(x_calls)):
        if trial_c[k] == 0:
            frames.append(
                dict(
                    step=int(step_c[k]),
                    iteration=int(iter_c[k]),
                    x=_break_between_elements(x_calls[k], n_gp),
                    r=_break_between_elements(r_calls[k], n_gp),
                    x_raw=x_calls[k],
                    r_raw=r_calls[k],
                    trials=[],
                )
            )
        else:
            frames[-1]["trials"].append(
                dict(
                    alpha=0.5 ** (int(trial_c[k]) - 1),
                    x=_break_between_elements(x_calls[k], n_gp),
                    r=_break_between_elements(r_calls[k], n_gp),
                    r_raw=r_calls[k],
                )
            )

    if last_n_steps is not None:
        step_min = max(f["step"] for f in frames) - last_n_steps + 1
        frames = [f for f in frames if f["step"] >= step_min]
    return frames[:: max(stride, 1)]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _finite(values: NDArray) -> NDArray:
    return values[np.isfinite(values)]


def _max_abs(values: NDArray) -> float:
    """max|R| over the finite entries (NaN if none are finite)."""
    finite = _finite(values)
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _rms(values: NDArray) -> float:
    finite = _finite(values)
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def _n_nonfinite(values: NDArray) -> int:
    return int(np.size(values) - np.isfinite(values).sum())


def _padded_limits(values: NDArray) -> tuple[float, float]:
    finite = _finite(values)
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    pad = 0.1 * abs(hi - lo) + 1e-5
    return lo - pad, hi + pad


# --------------------------------------------------------------------------- #
#  Animation
# --------------------------------------------------------------------------- #
ITER_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]  # NR it. 0..4
ITER_OVERFLOW_COLOR = "0.4"  # NR it. >= 5
C_ITERATE = "#0072B2"
C_TRIAL = "#B9314F"
YLIM_MODES = ("step", "frame", "global")


def _apply_publication_style() -> None:
    """Same defaults as the solver's post_plotting."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )


def _style_axis(ax) -> None:
    ax.minorticks_on()
    ax.tick_params(
        which="both", direction="in", top=True, right=True, length=4, width=0.8
    )
    ax.tick_params(which="minor", length=2, width=0.5)


def _time_positions(steps: NDArray) -> NDArray:
    """x-position of every frame on a time-step axis: iterates of step s are
    spread evenly over [s, s + 1)."""
    t_pos = np.zeros(len(steps))
    i = 0
    while i < len(steps):
        j = i
        while j < len(steps) and steps[j] == steps[i]:
            j += 1
        t_pos[i:j] = steps[i] + np.arange(j - i) / (j - i)
        i = j
    return t_pos


def animate_strong_residual(
    frames: list[dict],
    mesh: Optional[NDArray] = None,
    interval: int = 150,
    start_frame: int = 0,
    autoplay: bool = False,
    line_search: str = "backtracking",
    ylim_mode: str = "step",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> FuncAnimation:
    """Strong residual per Newton iterate.

    Panels: (top) profile at the Gauss points of the selected iterate,
    (middle) space-time map of all iterates, (bottom) max|R| / RMS history.
    Controls: Frame slider, Play/Pause, left/right arrows (step one iterate),
    ``y`` (cycle the profile y-limits).

    ``mesh``        node coordinates (``solver.mesh``) to draw element edges.
    ``line_search`` "backtracking": draw trials only where the line search had
                    to backtrack (alpha < 1 was tried); "all": always draw
                    them (an alpha = 1 trial duplicates the next iterate);
                    "none": never.
    ``ylim_mode``   profile y-limits: "step" (fixed within a time step, so the
                    NR convergence is visible), "frame" (rescale each iterate),
                    "global" (whole run).
    """
    if line_search not in ("backtracking", "all", "none"):
        raise ValueError("line_search must be 'backtracking', 'all' or 'none'")
    if ylim_mode not in YLIM_MODES:
        raise ValueError(f"ylim_mode must be one of {YLIM_MODES}")

    _apply_publication_style()

    n_frames = len(frames)
    start_frame = start_frame % n_frames  # allows -1 = last frame (the crash)
    steps = np.array([f["step"] for f in frames])
    iter_idx = np.array([f["iteration"] for f in frames])
    iterates = [f["r_raw"] for f in frames]
    max_hist = np.array([_max_abs(r) for r in iterates])
    rms_hist = np.array([_rms(r) for r in iterates])
    bad_hist = np.array([_n_nonfinite(r) for r in iterates])
    t_pos = _time_positions(steps)

    by_step: dict[int, list] = {}
    for s_, r_ in zip(steps, iterates):
        by_step.setdefault(int(s_), []).append(r_)
    step_ylim = {s_: _padded_limits(np.concatenate(rs)) for s_, rs in by_step.items()}
    global_ylim = _padded_limits(np.concatenate(iterates))

    # --- Figure layout: profile / space-time map / history (+ colorbar column) ---
    fig = plt.figure(figsize=(10, 9.5), dpi=120)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[50, 1],
        height_ratios=[1.15, 0.9, 0.9],
        hspace=0.42,
        wspace=0.03,
        left=0.08,
        right=0.90,
        top=0.86,
        bottom=0.2,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[2, 0], sharex=ax_map)
    cax = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[2, 1]).axis("off")
    for ax in (ax1, ax_map, ax_hist):
        _style_axis(ax)

    fig.suptitle(
        "Strong residual over time steps and Newton iterations",
        x=0.08,
        ha="left",
        y=0.975,
        fontsize=12,
    )

    # --- (1) Profile ---
    if mesh is not None:
        for node in mesh:
            ax1.axvline(node, color="0.88", lw=0.6, zorder=0)
    ax1.axhline(0.0, color="black", lw=0.5, alpha=0.4, ls=":")
    (line_r,) = ax1.plot(
        frames[0]["x"],
        frames[0]["r"],
        color=C_ITERATE,
        lw=1.6,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        zorder=4,
        label="Current iterate",
    )
    if line_search != "none":
        ax1.plot([], [], color=C_TRIAL, lw=1.0, alpha=0.7, label="Line-search trials")
    ax1.set_xlabel(r"Coordinate $x$")
    ax1.set_ylabel(r"Strong residual $\mathcal{R}$")
    fig.legend(
        *ax1.get_legend_handles_labels(),
        loc="upper right",
        bbox_to_anchor=(0.90, 0.985),
        ncol=2,
        frameon=False,
    )
    x_all = np.concatenate([f["x"][np.isfinite(f["x"])] for f in frames])
    ax1.set_xlim(x_all.min(), x_all.max())
    trial_artists: list = []

    # --- (2) Space-time map of every iterate ---
    n_pts = max(len(f["x_raw"]) for f in frames)
    x_gp = next(f["x_raw"] for f in frames if len(f["x_raw"]) == n_pts)
    matrix = np.full((n_pts, n_frames), np.nan)
    for j, r_ in enumerate(iterates):
        matrix[: len(r_), j] = r_
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    finite_abs = np.abs(matrix[np.isfinite(matrix)])
    vmax = float(np.percentile(finite_abs, 99)) if finite_abs.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.15")  # non-finite entries show up dark
    if n_frames > 1:
        mesh_artist = ax_map.pcolormesh(
            t_pos,
            x_gp,
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            shading="nearest",
            rasterized=True,
        )
        cbar = fig.colorbar(mesh_artist, cax=cax, extend="both")
        cbar.set_label(r"$\mathcal{R}$ (clipped at 99th pct.)", fontsize=9)
        cbar.ax.tick_params(direction="in", labelsize=8)
    else:
        cax.axis("off")
    ax_map.set_ylabel(r"Coordinate $x$")
    ax_map.set_ylim(x_all.min(), x_all.max())
    plt.setp(ax_map.get_xticklabels(), visible=False)
    cursor_map = ax_map.axvline(t_pos[start_frame], color="black", lw=1.0)

    # --- (3) History, coloured by NR iteration ---
    ax_hist.plot(t_pos, max_hist, "-", color="0.6", lw=0.8, zorder=1)
    ax_hist.plot(t_pos, rms_hist, ":", color="black", lw=1.2, zorder=2, label="RMS")
    for k in sorted(set(int(i) for i in iter_idx)):
        sel = iter_idx == k if k < len(ITER_COLORS) else iter_idx >= len(ITER_COLORS)
        color = ITER_COLORS[k] if k < len(ITER_COLORS) else ITER_OVERFLOW_COLOR
        label = (
            f"NR it. {k}"
            if k < len(ITER_COLORS)
            else f"NR it. $\\geq${len(ITER_COLORS)}"
        )
        if k >= len(ITER_COLORS) and k != len(ITER_COLORS):
            label = None  # already labelled by the first overflow group
        ax_hist.scatter(
            t_pos[sel], max_hist[sel], s=16, color=color, zorder=3, label=label
        )
    bad = bad_hist > 0
    if bad.any():
        ax_hist.plot(
            t_pos[bad],
            max_hist[bad],
            "x",
            color="black",
            markersize=7,
            zorder=5,
            label="Non-finite entries",
        )
    (point_now,) = ax_hist.plot(
        [], [], "o", mfc="none", mec="black", ms=9, mew=1.2, zorder=6
    )
    cursor_hist = ax_hist.axvline(t_pos[start_frame], color="black", lw=1.0)
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(steps.min(), steps.max() + 1)
    ax_hist.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_hist.set_xlabel(r"Time step")
    ax_hist.set_ylabel(r"$\max|\mathcal{R}|$, RMS")
    ax_hist.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0), numticks=12))
    ax_hist.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
    ax_hist.yaxis.set_minor_formatter(NullFormatter())
    ax_hist.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=7,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )

    # --- Plot update ---
    ylim_state = {"mode": ylim_mode}

    def update_plot(frame: int) -> None:
        f = frames[frame]
        line_r.set_data(f["x"], f["r"])

        for artist in trial_artists:
            artist.remove()
        trial_artists.clear()
        n_trials = len(f["trials"])
        draw_trials = line_search == "all" or (
            line_search == "backtracking" and n_trials > 1
        )
        if draw_trials:
            for trial in f["trials"]:
                (artist,) = ax1.plot(
                    trial["x"],
                    trial["r"],
                    "-",
                    color=C_TRIAL,
                    lw=0.9,
                    alpha=0.35,
                    zorder=2,
                )
                trial_artists.append(artist)

        mode = ylim_state["mode"]
        if mode == "step":
            ax1.set_ylim(*step_ylim[f["step"]])
        elif mode == "frame":
            ax1.set_ylim(*_padded_limits(f["r_raw"]))
        else:
            ax1.set_ylim(*global_ylim)

        lines = [
            f"Step {f['step']}  |  NR iteration {f['iteration']}  |  "
            f"frame {frame} / {n_frames - 1}",
            f"max|R| = {max_hist[frame]:.3e}   RMS = {rms_hist[frame]:.3e}"
            + (f"   non-finite entries: {bad_hist[frame]}" if bad_hist[frame] else ""),
        ]
        if n_trials == 1:
            lines.append(r"Line search: 1 trial ($\alpha = 1$)")
        elif n_trials > 1:
            trial_max = np.array([_max_abs(t["r_raw"]) for t in f["trials"]])
            finite = _finite(trial_max)
            lo, hi = (finite.min(), finite.max()) if finite.size else (np.nan, np.nan)
            lines.append(
                f"Line search: {n_trials} trials, $\\alpha = 1 \\ldots$ "
                f"{f['trials'][-1]['alpha']:.1e}, trial max|R| in [{lo:.2e}, {hi:.2e}]"
                + (
                    " (non-finite present)"
                    if any(_n_nonfinite(t["r_raw"]) for t in f["trials"])
                    else ""
                )
            )
        ax1.set_title("\n".join(lines), loc="left", pad=6)

        t = t_pos[frame]
        cursor_map.set_xdata([t, t])
        cursor_hist.set_xdata([t, t])
        point_now.set_data([t], [max_hist[frame]])
        fig.canvas.draw_idle()

    # --- Widgets ---
    ax_slider = fig.add_axes([0.30, 0.095, 0.55, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label="Frame",
        valmin=0,
        valmax=max(n_frames - 1, 1),
        valinit=start_frame,
        valfmt="%d",
        color=C_ITERATE,
    )
    ax_button = fig.add_axes([0.08, 0.085, 0.1, 0.05])
    play_button = Button(
        ax_button,
        "Pause" if autoplay else "Play",
        color="lightgoldenrodyellow",
        hovercolor="0.9",
    )
    fig.text(
        0.5,
        0.035,
        r"$\leftarrow$ / $\rightarrow$: previous / next iterate    "
        r"y: cycle y-limits (step / frame / global)",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )

    state = {"frame": int(np.clip(start_frame, 0, n_frames - 1))}
    is_playing = [autoplay]

    def toggle_play(event=None):
        is_playing[0] = not is_playing[0]
        play_button.label.set_text("Pause" if is_playing[0] else "Play")
        fig.canvas.draw_idle()

    play_button.on_clicked(toggle_play)

    def slider_changed(val):
        state["frame"] = int(val)
        update_plot(state["frame"])

    frame_slider.on_changed(slider_changed)

    def on_key(event):
        if event.key in ("left", "right"):
            is_playing[0] = False
            play_button.label.set_text("Play")
            step = -1 if event.key == "left" else 1
            frame_slider.set_val(int(np.clip(state["frame"] + step, 0, n_frames - 1)))
        elif event.key == "y":
            i = YLIM_MODES.index(ylim_state["mode"])
            ylim_state["mode"] = YLIM_MODES[(i + 1) % len(YLIM_MODES)]
            update_plot(state["frame"])

    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Frame runner: advances from the slider position, so Play resumes
    # where the slider was left. ---
    def update(_):
        if is_playing[0]:
            frame_slider.set_val((state["frame"] + 1) % n_frames)
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=itertools.count(),
        interval=interval,
        blit=False,
        cache_frame_data=False,
        save_count=n_frames,
    )

    update_plot(state["frame"])

    if save_path is not None:
        is_playing[0] = True
        state["frame"] = n_frames - 1  # first saved frame wraps to 0
        anim.save(str(save_path))

    if show:
        plt.show()
    return anim


def visualize_strong_residual_history(
    source: HistorySource,
    mesh: Optional[NDArray] = None,
    last_n_steps: Optional[int] = None,
    stride: int = 1,
    **animate_kwargs,
) -> FuncAnimation:
    """Entry point. ``source`` may be ``solver.strong_residual_history``
    (list of dicts) or the saved ``strong_residual_history.npz``."""
    df = tag_assembly_calls(load_strong_residual_history(source))
    frames = build_frames(df, last_n_steps=last_n_steps, stride=stride)
    print(
        f"{len(df)} Gauss-point records -> {len(frames)} NR iterates "
        f"(steps {frames[0]['step']}..{frames[-1]['step']})"
    )
    return animate_strong_residual(frames, mesh=mesh, **animate_kwargs)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Set this to the run directory written by the solver (or pass it as argv[1]).
    history_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT
        / "runs"
        / "les"
        / "n33_p_two_base_dt0.001"
        / "strong_residual_history.npz"
    )
    z_les = np.linspace(0, 2, 33)  # solver.mesh

    visualize_strong_residual_history(
        history_path,
        mesh=z_les,
        last_n_steps=10,  # None = whole run; the crash is at the end
        start_frame=-1,  # open on the last NR iterate
    )
