import datetime
from pathlib import Path

from constants import (
    RUNS_FOLDER,
    DNS_SAVE_PATH,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    SOLVER_DATA_FOLDER,
)
from data_curation.stencil_creation import create_stencils
from functions import run_config
from old.projection_and_stencils.project import run_projection
from problems_and_configurations.problems import placeholder_problem
from problems_and_configurations.configurations import (
    create_solver_configs,
)

CURRENT_DIR = Path(__file__).parent.resolve()

# -- pipeline settings ----------------------------


# -- problem --------------------------------------
problem: dict = placeholder_problem

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

# -- configs --------------------------------------
config_dns, config_les, config_les_no_model = create_solver_configs(
    problem, dns_data_path, les_a_data_path, les_nm_data_path
)

# -- data --------------------------------------
run_config(config_dns)
run_config(config_les)
run_config(config_les_no_model)

# projection -------------------------------------

pre_split_path.mkdir(parents=True, exist_ok=True)
if not solver_data_path_dns.exists():
    raise FileNotFoundError(f"DNS data not found at: {solver_data_path_dns}")

run_projection(
    directory=solver_data_path_dns,
    bc_mode=problem["boundary_condition_type"],
    bc_values=problem["boundary_condition_value"],
    save=True,
    output_dir=pre_split_path,
    verify=False,
)

create_stencils(pre_split_path, post_split_path)
