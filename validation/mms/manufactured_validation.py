"""Manufactured solution verification for the Burgers FEM solver."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray

from fem.burgers import Burgers
from manufactured import (
    set_manufactured_solution,
    set_manufactured_solution_initial,
    manufactured_residual,
)


# --- Simulation ---
def run_simulation(
    coordinates: NDArray,
    initial_solution: NDArray,
    viscosity: float,
    time_step: float,
    length: float,
    times: NDArray,
) -> tuple[list[NDArray], NDArray]:
    """Configure and run the Burgers solver, returning solutions at requested times."""

    def forcing(x: NDArray, t: float) -> NDArray:
        return manufactured_residual(x, t, viscosity=viscosity)

    config = Burgers.create_config(
        node_amount=len(coordinates),
        simulation_type="dns",
        run_objective="manufactured validation",
        solution_initial=initial_solution,
        boundary_conditions="fixed",
        viscosity=viscosity,
        time_step=time_step,
        length=length,
        time=times[-1] + time_step,
        extract_at_times=list(times),
        forcing=forcing,
    )
    solver = Burgers(configuration=config)
    solver.run_simulation()
    solver.post_logging()
    return solver.extracted_solutions, solver.node_cords


# --- Error computation ---
def compute_l2_error(
    solver_solutions: list[NDArray],
    exact_solutions: list[NDArray],
    element_size: float,
) -> float:
    """Compute the L2 norm of the error averaged over all extraction times."""
    errors = [
        np.sqrt(element_size * np.sum((u_h - u_ex) ** 2))
        for u_h, u_ex in zip(solver_solutions, exact_solutions)
    ]
    return float(np.mean(errors))


# --- Convergence study ---
def run_convergence_study(
    node_counts: list[int],
    length: float,
    viscosity: float,
    times: NDArray,
    dt_factor: float = 0.5,  # dt = dt_factor * h
) -> tuple[NDArray, NDArray]:
    mesh_sizes = []
    errors = []

    for n_nodes in node_counts:
        coords = np.linspace(0, length, n_nodes)
        h = length / (n_nodes - 1)
        dt = dt_factor * h**2  # <-- couple dt to h

        # times must be multiples of dt — snap to nearest
        snapped_times = np.array([round(t / dt) * dt for t in times])

        u0 = set_manufactured_solution_initial(coords)
        exact = [set_manufactured_solution(coords, t) for t in snapped_times]

        solver_solutions, _ = run_simulation(
            coords, u0, viscosity, dt, length, snapped_times
        )
        error = compute_l2_error(solver_solutions, exact, h)

        print(f"  n_nodes={n_nodes:4d}  h={h:.4f}  dt={dt:.5f}  L2_error={error:.4e}")
        mesh_sizes.append(h)
        errors.append(error)

    return np.array(mesh_sizes), np.array(errors)


# --- Plotting ---
def plot_solution_comparison(
    mesh: NDArray,
    ic: NDArray,
    exact_solutions: list[NDArray],
    solver_solutions: list[NDArray],
    times: NDArray,
) -> None:
    """Plot solver output against exact solutions and the initial condition."""
    _, ax = plt.subplots(figsize=(9, 5))
    ax: Axes

    ax.plot(mesh, ic, label="Initial condition", linestyle="--", color="gray")

    for i, (u_ex, u_h, t) in enumerate(zip(exact_solutions, solver_solutions, times)):
        label_exact = "Exact" if i == 0 else None
        label_solver = "Solver" if i == 0 else None
        ax.plot(mesh, u_ex, color="royalblue", alpha=0.6, label=label_exact)
        ax.plot(
            mesh, u_h, color="tab:orange", linestyle="--", alpha=0.8, label=label_solver
        )
        ax.text(mesh[-1] * 0.82, u_ex[len(u_ex) // 2], f"t={t:.2f}", fontsize=7)

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Solver vs exact manufactured solution")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_convergence(mesh_sizes: NDArray, errors: NDArray) -> None:
    """Log-log convergence plot with reference slopes."""
    _, ax = plt.subplots(figsize=(6, 5))
    ax: Axes

    ax.loglog(mesh_sizes, errors, "o-", color="royalblue", label="L2 error")

    # Reference slopes anchored at the coarsest point
    h0, e0 = mesh_sizes[0], errors[0]
    ax.loglog(mesh_sizes, e0 * (mesh_sizes / h0) ** 1, "k--", alpha=0.4, label="O(h¹)")
    ax.loglog(mesh_sizes, e0 * (mesh_sizes / h0) ** 2, "k:", alpha=0.4, label="O(h²)")

    # Measure and annotate the observed slope
    slope = float(np.polyfit(np.log(mesh_sizes), np.log(errors), 1)[0])
    ax.set_title(f"Convergence study  (observed slope = {slope:.2f})")
    ax.set_xlabel("h (mesh size)")
    ax.set_ylabel("L2 error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    LENGTH: float = 1.0
    TIME_STEP: float = 0.01
    VISCOSITY: float = 0.1
    TIMES: NDArray = np.linspace(0.1, 1.0, 5)

    # --- Single-resolution qualitative check ---
    N_NODES = 21
    coords = np.linspace(0, LENGTH, N_NODES)
    u0 = set_manufactured_solution_initial(coords)
    exact_solutions = [set_manufactured_solution(coords, t) for t in TIMES]

    solver_solutions, mesh = run_simulation(
        coords, u0, VISCOSITY, TIME_STEP, LENGTH, TIMES
    )

    plot_solution_comparison(mesh, u0, exact_solutions, solver_solutions, TIMES)

    # --- Convergence study ---
    print("\nConvergence study:")
    NODE_COUNTS = [11, 21, 41, 81]
    mesh_sizes, errors = run_convergence_study(
        NODE_COUNTS, LENGTH, VISCOSITY, TIMES, dt_factor=0.5
    )
    plot_convergence(mesh_sizes, errors)
