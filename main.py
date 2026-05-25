"""Main pipeline: DNS → LES → projection → training data → predictor → a priori verification."""

from pathlib import Path

import numpy as np
import torch

from constants import (
    RUNS_FOLDER,
    DNS_SAVE_PATH,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    SOLVER_DATA_FOLDER,
    TRAINING_DATA_FOLDER,
    PRE_SPLIT_FOLDER,
    POST_SPLIT_FOLDER,
    PREDICTOR_AGENT_FOLDER,
    A_PRIORI_FOLDER,
    LES_ANN_SAVE_PATH,
    N_NODES_LES, N_NODES_DNS, DNS_TO_LES_RATIO,
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
from problems_and_configurations.configurations import (
    create_solver_configs,
    create_ann_config,
)
from problems_and_configurations.problems import raj_one, raj_two
from solvers.burgers_coupled import BurgersCoupled
from pipeline_settings import PipelineConfig

CURRENT_DIR = Path(__file__).parent.resolve()

# -- pipeline settings ----------------------------
manual_path: str = "run_raj_two_0524_194829"
pipeline = PipelineConfig.coupled_only(manual_path=manual_path)


# ---------------------------------------------------------------------------
# Problem and pathing
# ---------------------------------------------------------------------------

problem: dict = raj_two

N_NODES_LES_TARGET: int = 9                          # 8 elements, h = 1/8
N_ELEMENTS_LES: int = N_NODES_LES_TARGET - 1         # 8
N_ELEMENTS_DNS: int = N_ELEMENTS_LES * DNS_TO_LES_RATIO  # 8 * 32 = 256 elements
N_NODES_DNS_TARGET: int = N_ELEMENTS_DNS + 1         # 257 nodes

H_LES: float = problem["domain_length"] / N_ELEMENTS_LES        # 0.125
H_DNS: float = problem["domain_length"] / N_ELEMENTS_DNS        # 0.00390625

CO_LES: float = 0.01
TEMPORAL_REFINEMENT: int = 4

DT_LES: float = CO_LES * H_LES                      # 0.00125
DT_DNS: float = DT_LES / TEMPORAL_REFINEMENT        # 0.0003125

master_run_id = pipeline.get_run_id(problem_name=problem["name"])

master_path = CURRENT_DIR / RUNS_FOLDER / master_run_id
master_path.mkdir(parents=True, exist_ok=True)

dns_data_path = master_path / SOLVER_DATA_FOLDER / DNS_SAVE_PATH

les_a_data_path = master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH
les_nm_data_path = master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH
projection_path = master_path / TRAINING_DATA_FOLDER / PRE_SPLIT_FOLDER
training_path = master_path / TRAINING_DATA_FOLDER / POST_SPLIT_FOLDER
model_output_path = master_path / PREDICTOR_AGENT_FOLDER
apriori_output_path = master_path / A_PRIORI_FOLDER

# ---------------------------------------------------------------------------
# Step 1: Solver runs — DNS + LES (analytical VMS) + LES (no model)
# ---------------------------------------------------------------------------

config_dns, config_les, config_les_no_model = create_solver_configs(
    problem,
    dns_data_path,
    les_a_data_path,
    les_nm_data_path,
    dns_time_step=DT_DNS,  # replaces les_time_step_override
    les_time_step=DT_LES,
    n_nodes_dns=N_NODES_DNS_TARGET,  # NEW
    n_nodes_les=N_NODES_LES_TARGET,  # NEW
)

if pipeline.run_solvers:
    if pipeline.run_dns:
        run_config(config_dns)

    run_config(config_les)
    run_config(config_les_no_model)

# ---------------------------------------------------------------------------
# Step 2: Load mesh info for element size
# ---------------------------------------------------------------------------

dns_solution, mesh_dns = read_data(directory=dns_data_path, final_only=True)
_, mesh_les = read_data(les_a_data_path, final_only=True)

n_les_nodes = len(mesh_les)
element_size_les = float(problem["domain_length"] / (n_les_nodes - 1))

# ---------------------------------------------------------------------------
# Step 3: Project DNS onto LES grid — produces ū, dns_on_les, f_bar snapshots
# ---------------------------------------------------------------------------

projection_path.mkdir(parents=True, exist_ok=True)
if not dns_data_path.exists():
    raise FileNotFoundError(f"DNS data not found at: {dns_data_path}")

if pipeline.run_projection:
    from data_curation.projection import read_dns_data

    _, dns_times, _, _ = read_dns_data(dns_data_path)
    dns_times_arr = np.array(dns_times)
    dt_dns_actual = float(dns_times_arr[1] - dns_times_arr[0])
    dt_les = float(config_les["time_step"])



    assert round(dt_les / dt_dns_actual) == TEMPORAL_REFINEMENT, (
        f"DNS saved at dt={dt_dns_actual:.8f}, expected {DT_DNS:.8f}. "
        f"Check that the DNS solver used dns_time_step=DT_DNS={DT_DNS}."
    )

    les_snapshot_indices = np.arange(0, len(dns_times), TEMPORAL_REFINEMENT)

    run_projection(
        directory=dns_data_path,
        bc_mode=problem["boundary_condition_type"],
        bc_values=problem["boundary_condition_value"],
        output_dir=projection_path,
        verify=False,
        les_snapshot_indices=les_snapshot_indices,
        n_nodes_les=N_NODES_LES_TARGET,
    )

# ---------------------------------------------------------------------------
# Step 4: Assemble training data (X, y) with Rajampeta output stencil
# ---------------------------------------------------------------------------

if pipeline.run_training_assembly:
    _, _, norm_stats = run_training_data_assembly(
        projection_path=projection_path,
        output_dir=training_path,
        dt=float(config_les["time_step"]),  # always dt_les, never dt_dns
        element_size=element_size_les,
    )

# ---------------------------------------------------------------------------
# Step 5: Train SGS predictor
# ---------------------------------------------------------------------------

if pipeline.run_training:
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

if pipeline.run_apriori:
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
        n_elements=n_les_nodes - 1,
    )

# ---------------------------------------------------------------------------
# Step 7: Run coupled-solver
# ---------------------------------------------------------------------------
solver_data_dir = master_path / SOLVER_DATA_FOLDER

config_ann, les_ann_stable_path, les_ann_blown_up_path = create_ann_config(
    problem_definition=problem,
    ann_model_path=model_output_path / "sgs_predictor.pt",
    normalisation_stats_path=training_path / "normalisation_stats.npz",
    les_ann_dir=solver_data_dir,  # base only — sub-dirs resolved inside
    time_step_override=DT_LES,
    n_nodes_les=N_NODES_LES_TARGET,
    clip_pusuluri=pipeline.clip_pusuluri,
    clip_rajampeta=pipeline.clip_rajampeta,
    blowup_threshold=1e4,
    blowup_buffer_size=5_000,
)

if pipeline.run_coupled:
    solver_ann = BurgersCoupled(
        config_ann,
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
    )
    solver_ann.print_configuration()
    solver_ann.run_simulation()
    solver_ann.post_processing()

les_ann_data_path = solver_ann.master_path if pipeline.run_coupled else les_ann_stable_path

# ---------------------------------------------------------------------------
# Step 8: Solution comparison plots (DNS / LES-A / LES-NM / projection)
# ---------------------------------------------------------------------------


if pipeline.run_plotting:
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
