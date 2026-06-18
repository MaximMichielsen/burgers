"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from constants import RUNS_FOLDER
from ml.ml_agents.predictor_stash import train_predictor, plot_training_diagnostics
from ml.data_assembly.a_priori_verification_stash import run_apriori_verification
from ml.ml_agents.predictor_stash import evaluate_on_val_set
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problems, Problem
from solvers.burgers_base import BurgersBase
from solvers.burgers_sgsp_stash import BurgersSGSP
from solvers.sgsp_training_data_generator_stash import BurgersDataGenerator
from ml.ml_agents.solver_configs import SGSPConfig
from utils.io_utils import read_data
from utils.plot_utils import plot_solution_comparison, SolutionConfig

CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

# -------------------- Problem and pipeline configuration ------------------------------ #

problem: Problem = Problems.raj_three
problem = replace(problem, domain_timespan=1.0)

pipeline = PipelineConfig.all(manual_path=r"")
pipeline.clip_pusuluri = False
pipeline.clip_rajampeta = False

n_nodes_les: int = 17
temporal_refinement: int = 1
courant_les: float = 0.1

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


# -------------------- Pipeline functions ------------------------------ #


def run_data_generator(
    problem: Problem,
    disc_cfg: DiscretisationConfig,
    master_path: Path,
    dns_save_path: Path,
    sgsp_data_training_path: Path,
) -> None:
    """Run DNS and assemble SGSP training data."""
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


def run_sgsp_training(
    data_path: Path,
    output_dir: Path,
    domain_length: float,
    n_elements: int,
) -> None:
    """Train SGSP predictor and run a priori verification."""
    model, training_stats = train_predictor(
        data_path=data_path,
        output_dir=output_dir,
    )
    plot_training_diagnostics(
        training_stats=training_stats,
        output_dir=output_dir,
        show_fig=False,
    )
    evaluate_on_val_set(
        model=model,
        data_path=data_path,
        output_dir=output_dir,
    )
    run_apriori_verification(
        model=model,
        data_dir=data_path,
        output_dir=output_dir,
        domain_length=domain_length,
        n_elements=n_elements,
    )


def run_sgsp_coupled_solver(
    problem: Problem,
    disc_cfg: DiscretisationConfig,
    master_path: Path,
    sgsp_cfg: SGSPConfig,
) -> None:
    """Run the LES solver with ANN-predicted SGS closure."""
    solver = BurgersSGSP(
        problem,
        disc_cfg,
        "sgsp",
        master_path,
        sgsp_cfg,
    )
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()


# -------------------- Entry point ------------------------------ #

if __name__ == "__main__":
    # --------------------------------------- DNS & SGSP data --------------------------------------- #
    run_data_generator(problem, disc_cfg, master_path, paths.dns_data, paths.projection)

    # --------------------------------------- SGSP training --------------------------------------- #
    run_sgsp_training(
        data_path=paths.projection,
        output_dir=paths.model_output,
        domain_length=problem.domain_length,
        n_elements=n_nodes_les - 1,
    )

    # --------------------------------------- SGSP coupled solver --------------------------------------- #
    sgsp_cfg = SGSPConfig(
        sgsp_model_path=paths.model_output / "sgs_predictor.pt",
        normalization_path=paths.projection,  # contains normalisation_stats.csv
        blown_up_path=paths.les_sgsp_data / "blown_up",
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
        set_off_predictor=False,
    )
    run_sgsp_coupled_solver(problem, disc_cfg, paths.les_sgsp_data, sgsp_cfg)

    les_run = BurgersBase(problem,
                          disc_cfg,
                          "les",
                          paths.les_a_data)
    les_run.run_simulation()
    les_run.post_processing()

    dns_solution, _ = read_data(directory=paths.dns_data, final_only=True)
    projected_solution = np.interp(disc_cfg.mesh_les, disc_cfg.mesh_dns, dns_solution)

    plot_solution_comparison(
        configs=[
            SolutionConfig(
                data_path=paths.dns_data,
                label="DNS",
                color="gray",
                linestyle="-",
                marker="",
                alpha=0.7,
                mesh=disc_cfg.mesh_dns,
                solution=dns_solution,
            ),
            SolutionConfig(
                data_path=paths.dns_data,
                label="LES - projection",
                color="lightgreen",
                marker="x",
                mesh=disc_cfg.mesh_les,
                solution=projected_solution,
            ),
            SolutionConfig(
                data_path=paths.les_a_data,
                label="LES - A",
                color="tab:orange",
                marker="^",
                mesh=disc_cfg.mesh_les,
            ),
            SolutionConfig(
                data_path=paths.les_sgsp_data,
                label="LES - SGSP",
                color="crimson",
                marker="d",
                mesh=disc_cfg.mesh_les,
            ),
        ],
        output_path=master_path,
        filename="comparison_dns_sgsp.png",
    )
