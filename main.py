import datetime
from pathlib import Path

import numpy as np

from constants import (
    RUNS_FOLDER,
    DNS_SAVE_PATH,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    SOLVER_DATA_FOLDER,
    TRAINING_DATA_FOLDER,
    PRE_SPLIT_FOLDER,
)
from data_curation.projection import run_projection
from functions import run_config, SolutionConfig, read_data, plot_solution_comparison
from problems_and_configurations.problems import robijns_one
from problems_and_configurations.configurations import (
    create_solver_configs,
)

CURRENT_DIR = Path(__file__).parent.resolve()

# -- pipeline settings ----------------------------


# -- problem --------------------------------------
problem: dict = robijns_one

# -- pathing --------------------------------------
timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
manual_path = ""
master_run_id = (
    f"run_{problem['name']}_{timestamp}" if manual_path == "" else manual_path
)

master_path = CURRENT_DIR / RUNS_FOLDER / master_run_id
master_path.mkdir(parents=True, exist_ok=True)

dns_data_path = master_path / SOLVER_DATA_FOLDER / DNS_SAVE_PATH
les_a_data_path = master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH
les_nm_data_path = master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH
projection_path = master_path / TRAINING_DATA_FOLDER / PRE_SPLIT_FOLDER

# -- configs --------------------------------------
config_dns, config_les, config_les_no_model = create_solver_configs(
    problem, dns_data_path, les_a_data_path, les_nm_data_path
)

# -- data --------------------------------------
run_config(config_dns)
run_config(config_les)
run_config(config_les_no_model)

# projection -------------------------------------
projection_path.mkdir(parents=True, exist_ok=True)
if not dns_data_path.exists():
    raise FileNotFoundError(f"DNS data not found at: {dns_data_path}")

run_projection(
    directory=dns_data_path,
    bc_mode=problem["boundary_condition_type"],
    bc_values=problem["boundary_condition_value"],
    save=True,
    output_dir=projection_path,
    verify=False,
)

# -- plotting ----------------------------------
dns_solution, mesh_dns = read_data(directory=dns_data_path, final_only=True)
_, mesh_les = read_data(les_a_data_path, final_only=True)
projected_solution = np.load(projection_path / "solutions_projection.npy")
projected_solution = projected_solution[-1]

dns_settings = SolutionConfig(
    data_path=dns_data_path,
    label="DNS",
    color="gray",
    linestyle="-",
    marker="",  # no marker for the reference curve
    alpha=0.7,
    mesh=mesh_dns,
    solution=dns_solution,
)
les_a_settings = SolutionConfig(
    data_path=les_a_data_path,
    label="LES - A",
    color="royalblue",
    marker="x",
    mesh=mesh_les,
)
les_nm_settings = SolutionConfig(
    data_path=les_nm_data_path,
    label="LES - no model",
    color="tab:orange",
    marker=".",
    mesh=mesh_les,
)
projection_config = SolutionConfig(
    data_path=dns_data_path,  # unused when mesh/solution are provided
    label="LES - projection",
    color="lightgreen",
    marker="^",
    mesh=mesh_les,
    solution=projected_solution,
)
plotting_settings = [dns_settings, les_a_settings, les_nm_settings, projection_config]
plot_solution_comparison(configs=plotting_settings, output_path=master_path)
