"""Entry point: runs the full DNS → LES → projection → training → coupled-solver pipeline."""

from pathlib import Path

import numpy as np
import torch

from constants import RUNS_FOLDER
from data_curation.a_priori_verificiation import run_apriori_verification
from data_curation.projection import run_projection
from data_curation.training_data_assembly import run_training_data_assembly
from ml_agents.predictor import plot_training_diagnostics, train_predictor
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.configurations import (
    build_mesh_config,
    create_ann_config,
    create_solver_configs,
)
from problems_and_configurations.mesh_config import DiscretisationConfig
from problems_and_configurations.problems import (
    Problem,
    Problems,
)
from solvers.burgers_sgsp import BurgersSGSP
from utils.enegy_evolution_utils import plot_energy_comparison
from utils.io_utils import read_data
from utils.plot_utils import build_plot_configs, plot_solution_comparison
from utils.solver_utils import run_config

CURRENT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------

pipeline = PipelineConfig.all_stages(manual_path="")

# ---------------------------------------------------------------------------
# Problem and discretisation
# ---------------------------------------------------------------------------

problem: Problem = Problems.pipeline_test

disc_cfg = DiscretisationConfig(
    n_elements_les=4,
    temporal_refinement=1,
    courant_les=0.01,
    domain_length=problem.domain_length,
)

dns_mesh_cfg = build_mesh_config(
    disc_cfg.n_nodes_dns,
    disc_cfg.domain_length,
    disc_cfg.dt_dns,
    problem.initial_condition,
)
les_mesh_cfg = build_mesh_config(
    disc_cfg.n_nodes_les,
    disc_cfg.domain_length,
    disc_cfg.dt_les,
    problem.initial_condition,
)

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()

# ---------------------------------------------------------------------------
# Step 1: DNS + LES (analytical VMS) + LES (no model)
# ---------------------------------------------------------------------------

config_dns, config_les, config_les_no_model = create_solver_configs(
    problem_definition=problem,
    dns_mesh=dns_mesh_cfg,
    les_mesh=les_mesh_cfg,
    dns_dir=paths.dns_data,
    les_a_dir=paths.les_a_data,
    les_nm_dir=paths.les_nm_data,
)

if pipeline.run_solvers:
    if pipeline.run_dns:
        run_config(config_dns)
    run_config(config_les)
    run_config(config_les_no_model)

# ---------------------------------------------------------------------------
# Step 2: Project DNS onto LES grid
# ---------------------------------------------------------------------------

paths.projection.mkdir(parents=True, exist_ok=True)
if not paths.dns_data.exists():
    raise FileNotFoundError(f"DNS data not found at: {paths.dns_data}")

if pipeline.run_projection:
    _, dns_times, _, _ = read_data(paths.dns_data)
    run_projection(
        directory=paths.dns_data,
        bc_mode=problem.boundary_condition_type,
        bc_values=problem.boundary_condition_value,
        output_dir=paths.projection,
        verify=False,
        les_snapshot_indices=np.arange(0, len(dns_times), disc_cfg.temporal_refinement),
        n_nodes_les=disc_cfg.n_nodes_les,
    )

# ---------------------------------------------------------------------------
# Step 3: Assemble (X, y) training data
# ---------------------------------------------------------------------------

if pipeline.run_training_assembly:
    _, _, norm_stats = run_training_data_assembly(
        projection_path=paths.projection,
        output_dir=paths.training,
        dt=les_mesh_cfg.time_step,
        element_size=les_mesh_cfg.element_size,
    )

# ---------------------------------------------------------------------------
# Step 4: Train SGS predictor
# ---------------------------------------------------------------------------

if pipeline.run_training:
    trained_model, training_stats = train_predictor(
        data_path=paths.training,
        output_dir=paths.model_output,
    )
    plot_training_diagnostics(
        training_stats=training_stats,
        output_dir=paths.model_output,
    )

# ---------------------------------------------------------------------------
# Step 5: A priori verification
# ---------------------------------------------------------------------------

if pipeline.run_apriori:
    trained_model.eval()

    def model_predict_fn(x_array: np.ndarray) -> np.ndarray:
        """Wrap trained model for the a priori verification interface."""
        with torch.no_grad():
            return trained_model(torch.tensor(x_array, dtype=torch.float32)).numpy()

    metrics_val = run_apriori_verification(
        model_predict_fn=model_predict_fn,
        data_dir=paths.training,
        output_dir=paths.apriori,
        domain_length=problem.domain_length,
        dt=les_mesh_cfg.time_step,
        dataset_label="Validation",
        n_elements=les_mesh_cfg.n_nodes - 1,
    )

# ---------------------------------------------------------------------------
# Step 6: Coupled ANN solver
# ---------------------------------------------------------------------------

config_ann, les_ann_stable_path, les_ann_blown_up_path = create_ann_config(
    problem_definition=problem,
    les_mesh=les_mesh_cfg,
    ann_model_path=paths.model_output / "sgs_predictor.pt",
    normalisation_stats_path=paths.training / "normalisation_stats.npz",
    data_dir=paths.solver_data,
    clip_pusuluri=pipeline.clip_pusuluri,
    clip_rajampeta=pipeline.clip_rajampeta,
    blowup_threshold=1e4,
    blowup_buffer_size=5_000,
)

if pipeline.run_coupled:
    solver_ann = BurgersSGSP(
        configuration=config_ann,
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
    )
    solver_ann.print_configuration()
    solver_ann.run_simulation()
    solver_ann.post_processing()

les_ann_data_path = (
    solver_ann.master_path if pipeline.run_coupled else les_ann_stable_path
)

# ---------------------------------------------------------------------------
# Step 7: Plots
# ---------------------------------------------------------------------------

if pipeline.run_plotting:
    dns_solution, _ = read_data(directory=paths.dns_data, final_only=True)
    projected_solution = np.load(paths.projection / "solutions_projection.npy")[-1]

    plot_solution_comparison(
        configs=build_plot_configs(
            paths=paths,
            dns_mesh=dns_mesh_cfg,
            les_mesh=les_mesh_cfg,
            dns_solution=dns_solution,
            projected_solution=projected_solution,
            les_ann_data_path=les_ann_data_path,
        ),
        output_path=paths.master,
    )
    plot_energy_comparison(
        dns_dir=paths.dns_data,
        les_a_dir=paths.les_a_data,
        les_nm_dir=paths.les_nm_data,
        les_ann_dir=les_ann_data_path,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )
