"""Main pipeline: DNS → LES → projection → training data → predictor → a priori verification."""

import datetime
from pathlib import Path

import numpy as np
import torch
from sympy import false

from constants import (
    RUNS_FOLDER,
    DNS_SAVE_PATH,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    SOLVER_DATA_FOLDER,
    TRAINING_DATA_FOLDER,
    PRE_SPLIT_FOLDER,
    POST_SPLIT_FOLDER,
)
from data_curation.a_priori_verificiation import run_apriori_verification
from data_curation.projection import run_projection
from data_curation.training_data_assembly import run_training_data_assembly
from functions import run_config, SolutionConfig, read_data, plot_solution_comparison
from ml_agents.predictor import (
    train_predictor,
    plot_training_diagnostics,
)
from plotting.energy_evolution import plot_energy_comparison
from problems_and_configurations.configurations import create_solver_configs
from problems_and_configurations.problems import robijns_one
from solvers.burgers_coupled import BurgersCoupled
from problems_and_configurations.configurations import create_ann_config

CURRENT_DIR = Path(__file__).parent.resolve()

# -- pipeline settings ----------------------------
RUN_SOLVERS: bool = False
RUN_PROJECTION: bool = False
RUN_TRAINING_ASSEMBLY: bool = False
RUN_TRAINING: bool = False
RUN_APRIORI: bool = False
RUN_COUPLED: bool = True
RUN_PLOTTING: bool = True

RUN_ONLY_COUPLED: bool = False

if RUN_ONLY_COUPLED:
    RUN_SOLVERS: bool = False
    RUN_PROJECTION: bool = False
    RUN_TRAINING_ASSEMBLY: bool = False
    RUN_TRAINING: bool = False
    RUN_APRIORI: bool = False
    RUN_COUPLED: bool = True
    RUN_PLOTTING: bool = True

# -- pathing --------------------------------------
manual_path: str = "run_robijns_one_0520_152921"

# ---------------------------------------------------------------------------
# Problem and pathing
# ---------------------------------------------------------------------------

problem: dict = robijns_one

timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
master_run_id = (
    manual_path if manual_path != "" else f"run_{problem['name']}_{timestamp}"
)

master_path = CURRENT_DIR / RUNS_FOLDER / master_run_id
master_path.mkdir(parents=True, exist_ok=True)

dns_data_path = master_path / SOLVER_DATA_FOLDER / DNS_SAVE_PATH
les_a_data_path = master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH
les_nm_data_path = master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH
projection_path = master_path / TRAINING_DATA_FOLDER / PRE_SPLIT_FOLDER
training_path = master_path / TRAINING_DATA_FOLDER / POST_SPLIT_FOLDER
model_output_path = master_path / "predictor"
apriori_output_path = master_path / "apriori"

# ---------------------------------------------------------------------------
# Step 1: Solver runs — DNS + LES (analytical VMS) + LES (no model)
# ---------------------------------------------------------------------------

config_dns, config_les, config_les_no_model = create_solver_configs(
    problem, dns_data_path, les_a_data_path, les_nm_data_path
)

if RUN_SOLVERS:
    run_config(config_dns)
    run_config(config_les)
    run_config(config_les_no_model)

# ---------------------------------------------------------------------------
# Step 2: Project DNS onto LES grid — produces ū, dns_on_les, f_bar snapshots
# ---------------------------------------------------------------------------

projection_path.mkdir(parents=True, exist_ok=True)
if not dns_data_path.exists():
    raise FileNotFoundError(f"DNS data not found at: {dns_data_path}")

if RUN_PROJECTION:
    run_projection(
        directory=dns_data_path,
        bc_mode=problem["boundary_condition_type"],
        bc_values=problem["boundary_condition_value"],
        save=True,
        output_dir=projection_path,
        verify=False,
    )

# ---------------------------------------------------------------------------
# Step 3: Load mesh info for element size
# ---------------------------------------------------------------------------

dns_solution, mesh_dns = read_data(directory=dns_data_path, final_only=True)
_, mesh_les = read_data(les_a_data_path, final_only=True)

n_les_nodes = len(mesh_les)
element_size_les = float(problem["domain_length"] / (n_les_nodes - 1))

# ---------------------------------------------------------------------------
# Step 4: Assemble training data (X, y) with Rajampeta output stencil
# ---------------------------------------------------------------------------
if RUN_TRAINING_ASSEMBLY:
    _, _, norm_stats = run_training_data_assembly(
        projection_path=projection_path,
        output_dir=training_path,
        dt=config_les["time_step"],
        element_size=element_size_les,
    )

# ---------------------------------------------------------------------------
# Step 5: Train SGS predictor
# ---------------------------------------------------------------------------
if RUN_TRAINING:
    trained_model, training_stats = train_predictor(
        data_path=training_path,
        output_dir=model_output_path,
    )

    plot_training_diagnostics(
        training_stats=training_stats,
        output_dir=model_output_path,
    )

# ---------------------------------------------------------------------------
# Step 6: A priori verification on validation set
# ---------------------------------------------------------------------------
if RUN_APRIORI:
    trained_model.eval()

    def model_predict_fn(x_array: np.ndarray) -> np.ndarray:
        """Wrap trained model for the a priori verification interface."""
        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        with torch.no_grad():
            return trained_model(x_tensor).numpy()

    metrics_val = run_apriori_verification(
        model_predict_fn=model_predict_fn,
        data_dir=training_path,
        output_dir=apriori_output_path,
        domain_length=problem["domain_length"],
        dt=config_les["time_step"],
        dataset_label="Validation",
        n_elements=n_les_nodes - 1,  # add this
    )

# ---------------------------------------------------------------------------
# Step 7: Run coupled-solver
# ---------------------------------------------------------------------------


les_ann_data_path = master_path / SOLVER_DATA_FOLDER / "LES_ANN"

config_ann = create_ann_config(
    problem_definition=problem,
    ann_model_path=model_output_path / "sgs_predictor.pt",
    normalisation_stats_path=training_path / "normalisation_stats.npz",
    les_ann_dir=les_ann_data_path,
)
if RUN_COUPLED:
    solver_ann = BurgersCoupled(config_ann, clip_pusuluri=True, clip_rajampeta=True)
    solver_ann.print_configuration()
    solver_ann.run_simulation()
    solver_ann.post_processing()

# ---------------------------------------------------------------------------
# Step 8: Solution comparison plots (DNS / LES-A / LES-NM / projection)
# ---------------------------------------------------------------------------
if RUN_PLOTTING:
    projected_solution = np.load(projection_path / "solutions_projection.npy")[-1]

    dns_plot_config = SolutionConfig(
        data_path=dns_data_path,
        label="DNS",
        color="gray",
        linestyle="-",
        marker="",
        alpha=0.7,
        mesh=mesh_dns,
        solution=dns_solution,
    )
    les_a_plot_config = SolutionConfig(
        data_path=les_a_data_path,
        label="LES - A",
        color="royalblue",
        marker="x",
        mesh=mesh_les,
    )
    les_nm_plot_config = SolutionConfig(
        data_path=les_nm_data_path,
        label="LES - no model",
        color="tab:orange",
        marker=".",
        mesh=mesh_les,
    )
    projection_plot_config = SolutionConfig(
        data_path=dns_data_path,
        label="LES - projection",
        color="lightgreen",
        marker="^",
        mesh=mesh_les,
        solution=projected_solution,
    )
    les_ann_plot_config = SolutionConfig(
        data_path=les_ann_data_path,
        label="LES - ANN",
        color="salmon",
        marker="d",
        mesh=mesh_les,
    )

    plot_solution_comparison(
        configs=[
            dns_plot_config,
            les_a_plot_config,
            les_nm_plot_config,
            projection_plot_config,
            les_ann_plot_config,
        ],
        output_path=master_path,
    )

    plot_energy_comparison(
        dns_dir=dns_data_path,
        les_a_dir=les_a_data_path,
        les_nm_dir=les_nm_data_path,
        les_ann_dir=les_ann_data_path,
        output_path=master_path,
        viscosity=problem["viscosity"],
        domain_length=problem["domain_length"],
    )
