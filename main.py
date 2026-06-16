"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib

from constants import RUNS_FOLDER
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problems, Problem

from solvers.sgsp_training_data_generator import BurgersDataGenerator

CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

# -------------------- Problem and pipeline configuration ------------------------------ #

problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=5.0)

pipeline = PipelineConfig.all(manual_path=r"")
pipeline.run_les_no_model = False
pipeline.clip_pusuluri = False
pipeline.clip_rajampeta = False

n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 0.01

PROJECTION_MODE: str = "nodal"
ALPHA_MAX: float = 100 * problem.viscosity
OUTPUT_SCALE: float = 1
AVC_EPOCHS: int = 20
N_SKIP: int = 5

disc_cfg = DiscretisationConfig(
    n_nodes_les=n_nodes_les,
    temporal_refinement=temporal_refinement,
    courant_les=courant_les,
    domain_length=problem.domain_length,
)

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()

manual_load_dns: str = r""
paths.dns_data = Path(manual_load_dns) if manual_load_dns != "" else paths.dns_data


def run_data_generator(
    problem,
    disc_cfg,
    master_path,
    dns_save_path,
    sgsp_data_training_path,
):
    solver = BurgersDataGenerator(
        problem,
        disc_cfg,
        "dns",
        master_path,
        dns_save_path,
        sgsp_data_training_path,
    )
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()


if __name__ == "__main__":
    # --------------------------------------- DNS & SGSP data --------------------------------------- #
    run_data_generator(problem, disc_cfg, master_path, paths.dns_data, paths.projection)
