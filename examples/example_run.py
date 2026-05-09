"""Main."""

import numpy as np
from numpy.typing import NDArray

from burgers import Burgers


def initial_condition(x: NDArray) -> NDArray:
    """Sinusoidal initial condition u(x, 0) = sin(x)."""
    return np.sin(x)


def _main() -> None:
    """Example use case of the Burgers solver."""
    # --- Parameters ---
    LENGTH: float = 2 * np.pi
    TIME: float = 1.0
    VISCOSITY: float = 1e-2
    CFL: float = 0.1
    LES_RATIO: int = 2**4  # coarsening factor: DNS -> LES

    # --- Grid ---
    dx_dns: float = VISCOSITY / 2
    dx_les: float = dx_dns * LES_RATIO
    n_nodes_dns: int = int(LENGTH // dx_dns)
    n_nodes_les: int = int(LENGTH // dx_les)

    # --- Initial conditions ---
    cords_dns: NDArray = np.linspace(0, LENGTH, n_nodes_dns)
    cords_les: NDArray = np.linspace(0, LENGTH, n_nodes_les)

    ic_dns: NDArray = initial_condition(cords_dns)
    ic_les: NDArray = initial_condition(cords_les)
    ic_dns[0] = ic_dns[-1]  # enforce periodicity
    ic_les[0] = ic_les[-1]

    # --- Time step ---
    dt_les: float = CFL * dx_les * np.max(ic_dns) * 10

    # --- Run ---
    config_les = Burgers.create_config(
        node_amount=n_nodes_les,
        simulation_type="les",
        initial_condition=ic_les,
        viscosity=VISCOSITY,
        time_step=dt_les,
        domain_timespan=TIME,
        boundary_conditions="fixed_one",
        forcing="uniform",
    )

    solver = Burgers(config_les)
    solver.initial_solution_is_valid()
    solver.run_simulation()
    solver.post_processing()


if __name__ == "__main__":
    _main()
