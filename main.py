import datetime
from pathlib import Path

from constants import RUNS_FOLDER
from functions import run_config
from problems_and_configurations.problems import placeholder_problem
from problems_and_configurations.configurations import create_placeholder_config

CURRENT_DIR = Path(__file__).parent.resolve()


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

# -- configs --------------------------------------
config_dns = create_placeholder_config(problem, master_dir=master_path, save_dir="DNS")
config_les = create_placeholder_config(problem, master_path, save_dir="LES")

# -- data --------------------------------------
run_config(config_dns)
run_config(config_les)
