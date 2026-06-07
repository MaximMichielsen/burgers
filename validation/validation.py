"""Validate the FEM Burgers' solver using exact solutions."""

from math import cos, exp, pi, sin

import numpy as np
from fem.burgers import Burgers
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


# --- Exact solution ---
def initial_condition(x: float, viscosity: float, sigma: float = 2.0) -> float:
    """Initial condition derived from the exact solution of Burgers' equation."""
    return 2 * (viscosity**2 * pi * sin(pi * x)) / (sigma + cos(pi * x))


def exact_solution(x: float, t: float, viscosity: float, sigma: float = 2.0) -> float:
    """Exact solution to Burgers' equation (Example 1)."""
    decay = exp(-(pi**2) * viscosity**2 * t)
    return 2 * (viscosity**2 * pi * decay * sin(pi * x)) / (sigma + decay * cos(pi * x))


def evaluate_on_mesh(fn: callable, cords: np.ndarray, **kwargs: float) -> np.ndarray:
    """Evaluate a scalar function over an array of coordinates."""
    return np.array([fn(x, **kwargs) for x in cords])


# --- Simulation ---
def run_simulation(
    coordinates: np.ndarray,
    initial_solution: np.ndarray,
    viscosity: float,
    time_step: float,
    length: float,
    times: list[float],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Configure and run the Burgers solver, returning solutions at requested times."""
    config = Burgers.create_config(
        node_amount=len(coordinates),
        simulation_type="les",
        run_objective="validation",
        initial_condition=initial_solution,
        viscosity=viscosity,
        time_step=time_step,
        domain_length=length,
        domain_timespan=times[-1] + time_step,
        time_extractions=times,
    )
    solver = Burgers(configuration=config)
    solver.run_simulation()
    solver.post_logging()
    return solver.node_cords, solver.snapshots_solution


# --- Plotting ---
def plot_results(
    mesh: np.ndarray,
    ic: np.ndarray,
    exact_solutions: list[np.ndarray],
    solver_solutions: list[np.ndarray],
    times: list[float],
) -> None:
    """Plot solver output against exact solutions and the initial condition."""
    _, ax = plt.subplots()
    ax: Axes

    # Initial condition
    ax.plot(mesh, ic, label="Initial condition", linestyle="--", color="gray")
    ax.text(0.8, ic[len(ic) // 2], "t = 0")

    # Exact solutions
    exact_labels: list[str] = [f"t = {t}" for t in times]
    exact_styles: list[str] = ["-", "-", "-."]
    for i, (ec, label, style) in enumerate(
        zip(exact_solutions, exact_labels, exact_styles)
    ):
        ax.plot(
            mesh,
            ec,
            linestyle=style,
            color="royalblue",
            label="Exact solution" if i == 0 else None,
        )
        ax.text(0.8 - i * 0.1, ec[len(ec) // 2], label)

    # Solver solutions
    for i, solution in enumerate(solver_solutions):
        ax.plot(mesh, solution, color="tab:orange", label="Solver" if i == 0 else None)

    ax.set_xlabel("Spatial dimension")
    ax.set_ylabel("Velocity")
    ax.set_title("Comparison of solver and exact solution")
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    LENGTH: float = 1.0
    SPACE_STEP: float = 0.05
    TIME_STEP: float = 0.01
    VISCOSITY: float = 1
    TIMES: list[float] = [0.01, 0.1, 1.0]

    cords: np.ndarray = np.linspace(0, LENGTH, int(LENGTH / SPACE_STEP) + 1)
    ic: np.ndarray = evaluate_on_mesh(initial_condition, cords, viscosity=VISCOSITY)

    exact_solutions: list[np.ndarray] = [
        evaluate_on_mesh(exact_solution, cords, t=t, viscosity=VISCOSITY) for t in TIMES
    ]

    mesh: np.ndarray
    solver_solutions: list[np.ndarray]
    mesh, solver_solutions = run_simulation(
        cords, ic, VISCOSITY, TIME_STEP, LENGTH, TIMES
    )

    plot_results(mesh, ic, exact_solutions, solver_solutions, TIMES)
