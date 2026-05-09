"""Manufactured solution to use for solver development and verification."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

OMEGA: float = 1
ALPHA: float = 1
KAPPA: float = 1


def set_manufactured_solution(
    x: float | NDArray,
    t: float,
    omega: float = OMEGA,
    alpha: float = ALPHA,
    k: float = KAPPA,
) -> float | NDArray:
    """Manufactured solution: u(x,t) = Asin(kπx) * cos(ωt)."""
    return alpha * np.sin(k * np.pi * x) * np.cos(omega * t)


def set_manufactured_solution_initial(
    x: float | NDArray, alpha: float = ALPHA, k: float = KAPPA
) -> float | NDArray:
    """Initial condition to the manufactured solution."""
    return alpha * np.sin(k * np.pi * x)


def manufactured_solution_boundary_conditions() -> tuple[float, float]:
    """Boundary conditions to the manufactured solution of length: 1."""
    return 0, 0


def man_sol_dt(
    x: float | NDArray,
    t: float,
    omega: float = OMEGA,
    alpha: float = ALPHA,
    k: float = KAPPA,
) -> float | NDArray:
    """u_t."""
    return -1 * omega * alpha * np.sin(k * np.pi * x) * np.sin(omega * t)


def man_sol_dx(
    x: float | NDArray,
    t: float,
    omega: float = OMEGA,
    alpha: float = ALPHA,
    k: float = KAPPA,
) -> float | NDArray:
    """u_x."""
    return alpha * np.pi * np.cos(k * np.pi * x) * np.cos(omega * t)


def man_sol_dxx(
    x: float | NDArray,
    t: float,
    omega: float = OMEGA,
    alpha: float = ALPHA,
    k: float = KAPPA,
) -> float | NDArray:
    """u_xx."""
    return alpha * -1 * np.pi**2 * np.sin(k * np.pi * x) * np.cos(omega * t)


def manufactured_residual(
    x: float | NDArray, t: float, viscosity: float
) -> float | NDArray:
    """R(u)."""
    return (
        man_sol_dt(x, t)
        + set_manufactured_solution(x, t) * man_sol_dx(x, t)
        - viscosity * man_sol_dxx(x, t)
    )


if __name__ == "__main__":
    mesh_exact = np.linspace(start=0, stop=1, num=100)
    time_mesh_exact = np.linspace(start=0, stop=1, num=10)

    manufactured_solutions = []
    residuals = []

    for time in time_mesh_exact:
        manufactured_solution = set_manufactured_solution(mesh_exact, t=time)
        residual = manufactured_residual(mesh_exact, time, viscosity=1)

        manufactured_solutions.append(manufactured_solution)
        residuals.append(residual)

    print(residuals)

    for index, (solution, residual) in enumerate(
        zip(manufactured_solutions, residuals)
    ):
        plt.plot(
            mesh_exact,
            solution,
            linestyle="-.",
            color="royalblue",
            label="solution" if index == 0 else None,
        )
        plt.plot(
            mesh_exact,
            residual,
            linestyle=":",
            color="tab:orange",
            label="residual" if index == 0 else None,
        )

    plt.legend()
    plt.grid(True)

    plt.show()
