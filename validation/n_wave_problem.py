import numpy as np

from burgers_pure import BurgersPure
from functions import set_extractions, plot_solutions_from_directory_animated


def initial_condition(nodes):
    term_1 = np.exp(-((nodes - 1) ** 2) / 2)
    term_2 = np.exp(-((nodes + 1) ** 2) / 2)
    return term_1 - term_2


config = BurgersPure.create_config(
    initial_condition=initial_condition,
    simulation_mode="dns",
    node_amount=2**6,
    viscosity=1,
    domain_length=2 * 2 * np.pi,
    domain_timespan=5,
    time_step=0.01,
    boundary_condition_type="periodic",
    external_forcing=None,
    master_path="runs",
    save_path="test_2",
    extract_at_times=set_extractions(
        duration=5, extraction_amount=2000, time_step=0.01
    ),
)

solver = BurgersPure(config)

solver.run_simulation()

plot_solutions_from_directory_animated(directory="runs/test_2")
