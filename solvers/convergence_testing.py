"""MMS convergence study for SolverBaseOpt (pure Galerkin).

Place this file next to solver_base_opt.py and run it from the project
environment. It performs independent temporal and spatial refinement studies.

The temporal study holds the mesh fixed and varies dt. The spatial study
holds the number of timesteps fixed and varies the mesh. In both studies the
MMS solution and forcing are regenerated on the corresponding grid.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from numpy.typing import NDArray

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solver_base_opt import DNSLoaderTest, SimulationMode, SolverBaseOpt


# ---------------------------------------------------------------------------
# MMS: keep these expressions consistent with the solver's __main__ section.
# ---------------------------------------------------------------------------


def manufactured_solution(x: NDArray, t: float | NDArray) -> NDArray:
    return np.sin(t) ** 2 * np.sin(x)


def manufactured_forcing(x: NDArray, t: float | NDArray, viscosity: float) -> NDArray:
    u = np.sin(t) ** 2 * np.sin(x)
    u_t = 2.0 * np.sin(t) * np.cos(t) * np.sin(x)
    u_x = np.sin(t) ** 2 * np.cos(x)
    u_xx = -(np.sin(t) ** 2) * np.sin(x)
    return u_t + u * u_x - viscosity * u_xx


@dataclass(frozen=True)
class StudySettings:
    domain_length: float = 2.0 * np.pi
    t_start: float = 0.0
    t_end: float = np.pi / 2.0
    reynolds: float = 180.0

    # Temporal refinement: fixed mesh, varying number of time steps.
    temporal_nodes: int = 513 * 2
    temporal_n_steps: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)

    # Spatial refinement: fixed number of time steps, varying node count.
    spatial_nodes: tuple[int, ...] = (11, 21, 41, 81)
    spatial_n_steps: int = 400

    # Plot/output settings.
    dpi: int = 300
    figure_size: tuple[float, float] = (10.0, 8.0)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
            "figure.autolayout": True,
        }
    )


def _make_problem(settings: StudySettings) -> Problem:
    return Problem(
        domain_length=settings.domain_length,
        t_start=settings.t_start,
        t_end=settings.t_end,
        name="mms_convergence",
        reynolds=settings.reynolds,
    )


def _run_case(
    *,
    settings: StudySettings,
    n_nodes: int,
    n_steps: int,
    output_dir: Path,
    label: str,
) -> dict:
    """Run one pure-Galerkin MMS case and return final-time error data."""
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least 3.")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2 for the BDF2 study.")

    problem = _make_problem(settings)
    dt = (settings.t_end - settings.t_start) / n_steps
    mesh = np.linspace(0.0, settings.domain_length, n_nodes)
    times = np.linspace(settings.t_start, settings.t_end, n_steps + 1)

    # Arrays have shape (time, space), matching DNSLoaderTest's expectations.
    X = mesh[None, :]
    T = times[:, None]
    velocity_data = manufactured_solution(X, T)
    forcing_data = manufactured_forcing(X, T, problem.viscosity)

    reference = DNSLoaderTest(
        velocity_data=velocity_data,
        velocity_times=times,
        forcing_data=forcing_data,
        forcing_times=times.copy(),
        t_start=settings.t_start,
    )
    disc_config = DiscretizationConfig(problem, n_nodes, dt)

    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)

    solver = SolverBaseOpt(
        problem=problem,
        disc_config=disc_config,
        reference_trajectory=reference,
        simulation_mode=SimulationMode.NO_MODEL,
        master_path=case_dir,
        tau_model=None,
        testing_mode=True,
        exact_solution=manufactured_solution,
        comparison_label="Pure Galerkin",
        do_post_processing=False,
    )
    solver.run_simulation()

    # Compare at the prescribed final time, rather than relying on an
    # accumulated floating-point time value for the analytical solution.
    exact = np.asarray(manufactured_solution(mesh, settings.t_end), dtype=float)
    numerical = np.asarray(solver.solution_now, dtype=float)
    error = numerical - exact

    l2 = float(np.sqrt(np.trapezoid(error**2, mesh)))
    exact_l2 = float(np.sqrt(np.trapezoid(exact**2, mesh)))
    relative_l2 = l2 / exact_l2 if exact_l2 > np.finfo(float).eps else l2

    return {
        "label": label,
        "n_nodes": int(n_nodes),
        "n_elements": int(n_nodes - 1),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "dx": float(settings.domain_length / (n_nodes - 1)),
        "mesh": mesh,
        "numerical": numerical,
        "exact": exact,
        "l2": l2,
        "relative_l2": float(relative_l2),
        "final_time": float(settings.t_end),
    }


def _fit_order(x: NDArray, error: NDArray) -> float:
    valid = np.isfinite(x) & np.isfinite(error) & (x > 0.0) & (error > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[valid]), np.log(error[valid]), 1)[0])


def _plot_profile_panel(ax, cases: list[dict], parameter: str, label: str, title: str):
    values = np.asarray([case[parameter] for case in cases], dtype=float)
    norm = LogNorm(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.get_cmap("viridis")

    x_exact = np.linspace(0.0, cases[0]["mesh"][-1], 800)
    ax.plot(
        x_exact,
        manufactured_solution(x_exact, cases[0]["final_time"]),
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="MMS",
        zorder=5,
    )

    for case in cases:
        color = cmap(norm(case[parameter]))
        ax.plot(
            case["mesh"],
            case["numerical"],
            color=color,
            linewidth=1.25,
            label=f"{parameter} = {case[parameter]:.2e}",
        )

    ax.set_xlabel(r"Coordinate $x$")
    ax.set_ylabel(r"Solution $u(x,t_f)$")
    ax.set_title(title, loc="left", pad=6)
    ax.set_xlim(0.0, cases[0]["mesh"][-1])
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.9)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(label)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())


def _plot_convergence_panel(
    ax, cases: list[dict], parameter: str, xlabel: str, title: str
):
    x = np.asarray([case[parameter] for case in cases], dtype=float)
    rel_l2 = np.asarray([case["relative_l2"] for case in cases], dtype=float)
    order_l2 = _fit_order(x, rel_l2)

    ax.loglog(
        x,
        rel_l2,
        color="#0072B2",
        marker="o",
        markersize=4,
        markerfacecolor="white",
        linewidth=1.6,
        label=rf"Relative $L^2$ (order {order_l2:.2f})",
    )
    # Second-order guide, anchored at the finest-grid relative-L2 error.
    if len(x) >= 2 and np.all(rel_l2 > 0.0):
        x_ref = np.asarray([x.min(), x.max()])
        y_ref = rel_l2[np.argmin(x)] * (x_ref / x.min()) ** 2
        ax.loglog(
            x_ref,
            y_ref,
            color="0.35",
            linestyle=":",
            linewidth=1.2,
            label=r"$O(\Delta^2)$ reference",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error")
    ax.set_title(title, loc="left", pad=6)
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.18, linewidth=0.5)


def _write_results(cases: Iterable[dict], path: Path) -> None:
    fields = [
        "label",
        "n_nodes",
        "n_elements",
        "n_steps",
        "dt",
        "dx",
        "final_time",
        "l2",
        "relative_l2",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case[field] for field in fields})


def run_convergence_study(
    output_dir: str | Path | None = None,
    settings: StudySettings | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run temporal and spatial refinement studies and save plots/results."""
    settings = settings or StudySettings()
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else (Path(__file__).resolve().parent.parent / "runs" / "mms_convergence")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    print("\nTEMPORAL CONVERGENCE — fixed spatial mesh")
    temporal_cases = []
    for n_steps in settings.temporal_n_steps:
        label = f"temporal_n{settings.temporal_nodes}_steps{n_steps}"
        print(f"  Running {label}")
        temporal_cases.append(
            _run_case(
                settings=settings,
                n_nodes=settings.temporal_nodes,
                n_steps=n_steps,
                output_dir=output_dir,
                label=label,
            )
        )

    print("\nSPATIAL CONVERGENCE — fixed temporal resolution")
    spatial_cases = []
    for n_nodes in settings.spatial_nodes:
        label = f"spatial_n{n_nodes}_steps{settings.spatial_n_steps}"
        print(f"  Running {label}")
        spatial_cases.append(
            _run_case(
                settings=settings,
                n_nodes=n_nodes,
                n_steps=settings.spatial_n_steps,
                output_dir=output_dir,
                label=label,
            )
        )

    temporal_cases.sort(key=lambda case: case["dt"], reverse=True)
    spatial_cases.sort(key=lambda case: case["dx"], reverse=True)

    _write_results(temporal_cases, output_dir / "temporal_convergence.csv")
    _write_results(spatial_cases, output_dir / "spatial_convergence.csv")
    (output_dir / "settings.json").write_text(
        json.dumps(asdict(settings), indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=settings.figure_size, dpi=settings.dpi)
    for ax in axes.flat:
        ax.minorticks_on()
        ax.tick_params(
            which="both", direction="in", top=True, right=True, length=4, width=0.8
        )
        ax.tick_params(which="minor", length=2, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    _plot_profile_panel(
        axes[0, 0],
        temporal_cases,
        "dt",
        r"Time step $\Delta t$",
        rf"(a) Temporal refinement: final-time profiles ($N_x={settings.temporal_nodes}$)",
    )
    _plot_convergence_panel(
        axes[0, 1],
        temporal_cases,
        "dt",
        r"Time step $\Delta t$",
        rf"(b) Temporal error convergence ($N_x={settings.temporal_nodes}$)",
    )
    _plot_profile_panel(
        axes[1, 0],
        spatial_cases,
        "dx",
        r"Element size $\Delta x$",
        rf"(c) Spatial refinement: final-time profiles ($N_t={settings.spatial_n_steps}$)",
    )
    _plot_convergence_panel(
        axes[1, 1],
        spatial_cases,
        "dx",
        r"Element size $\Delta x$",
        rf"(d) Spatial error convergence ($N_t={settings.spatial_n_steps}$)",
    )

    fig.savefig(
        output_dir / "mms_convergence.png", dpi=settings.dpi, bbox_inches="tight"
    )
    fig.savefig(
        output_dir / "mms_convergence.pdf", dpi=settings.dpi, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"\nSaved convergence figure and CSV results to: {output_dir}")
    print("\nObserved least-squares orders (relative L2):")
    print(
        f"  Temporal: {_fit_order(np.array([c['dt'] for c in temporal_cases]), np.array([c['relative_l2'] for c in temporal_cases])):.3f}"
    )
    print(
        f"  Spatial:  {_fit_order(np.array([c['dx'] for c in spatial_cases]), np.array([c['relative_l2'] for c in spatial_cases])):.3f}"
    )
    print(
        "\nPer-run errors are in temporal_convergence.csv and spatial_convergence.csv."
    )
    return temporal_cases, spatial_cases


if __name__ == "__main__":
    run_convergence_study()
