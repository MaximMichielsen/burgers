"""Entry point: runs the full DNS → LES → projection → training → coupled-solver pipeline."""

from pathlib import Path

import numpy as np

from constants import RUNS_FOLDER, BLOWUP_THRESHOLD, BLOWUP_BUFFER_SIZE
from pipeline_settings import PipelineConfig, RunPaths
from pipeline_stages import register_stages
from problems_and_configurations.configurations import (
    create_sgsp_config,
    create_solver_configs,
)
from problems_and_configurations.mesh_config import DiscretisationConfig
from problems_and_configurations.problems import Problem, Problems
from solvers.burgers_avc import BurgersAVC
from solvers.burgers_sgsp import BurgersSGSP
from utils.enegy_evolution_utils import plot_energy_comparison
from utils.io_utils import read_data
from utils.plot_utils import (
    build_plot_configs,
    plot_solution_comparison,
    _is_viable_solution_path,
)

CURRENT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------

pipeline = PipelineConfig.all_stages(manual_path="")
pipeline.debug_sgsp = True
pipeline.clip_pusuluri = True

problem: Problem = Problems.raj_two

disc_cfg = DiscretisationConfig(
    n_elements_les=16,
    temporal_refinement=1,
    courant_les=0.04,
    domain_length=problem.domain_length,
    initial_condition_fn=problem.initial_condition,
)

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()

# ---------------------------------------------------------------------------
# Build solver configs and register all stages onto pipeline
# ---------------------------------------------------------------------------

config_dns, config_les, config_les_no_model = create_solver_configs(
    problem_definition=problem,
    disc_cfg=disc_cfg,
    dns_dir=paths.dns_data,
    les_a_dir=paths.les_a_data,
    les_nm_dir=paths.les_nm_data,
)

config_sgsp, les_sgsp_stable_path, _ = create_sgsp_config(
    problem_definition=problem,
    disc_cfg=disc_cfg,
    sgsp_model_path=paths.model_output / "sgs_predictor.pt",
    normalisation_stats_path=paths.training / "normalisation_stats.npz",
    data_dir=paths.solver_data,
    clip_pusuluri=pipeline.clip_pusuluri,
    clip_rajampeta=pipeline.clip_rajampeta,
    blowup_threshold=BLOWUP_THRESHOLD,
    blowup_buffer_size=BLOWUP_BUFFER_SIZE,
)

register_stages(
    pipeline=pipeline,
    paths=paths,
    problem=problem,
    disc_cfg=disc_cfg,
    config_dns=config_dns,
    config_les=config_les,
    config_les_no_model=config_les_no_model,
    config_sgsp=config_sgsp,
    les_sgsp_stable_path=les_sgsp_stable_path,
)
# ---------------------------------------------------------------------------
# Step 1 · Step 2 · Step 3 · Step 4
# ---------------------------------------------------------------------------

paths.projection.mkdir(parents=True, exist_ok=True)

pipeline.run_solvers_stage()

if (
    pipeline.run_projection and not paths.dns_data.exists()
):  # ← after solvers, gated on projection
    raise FileNotFoundError(f"DNS data not found at: {paths.dns_data}")

pipeline.run_projection_stage()
pipeline.run_sgsp_training_assembly()
trained_model = pipeline.run_sgsp_training()

# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------

pipeline.verify_sgsp_apriori(trained_model)

# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------

solver_sgsp: BurgersSGSP | None = pipeline.run_sgsp_model()

if pipeline.run_avc_online_training and solver_sgsp is None:
    raise RuntimeError(
        "Step 6 requires solver_sgsp — enable pipeline.run_sgsp before run_avc_online_training."
    )

les_sgsp_data_path = (
    solver_sgsp.master_path if pipeline.run_sgsp else les_sgsp_stable_path
)

# ---------------------------------------------------------------------------
# Step 6b: SGSP diagnostics (debug only)
# ---------------------------------------------------------------------------

if pipeline.run_sgsp and pipeline.debug_sgsp:
    from ml.diagnostics.sgsp_diagnostics import (
        diagnose_sgsp_predictions,
        diagnose_training_label_scale,
    )
    from solvers.burgers_sgsp import BurgersSGSP

    diagnose_training_label_scale(training_data_path=paths.training)
    solver_ann_debug = BurgersSGSP(
        configuration=config_sgsp,
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
    )
    diagnose_sgsp_predictions(solver=solver_ann_debug, n_steps=10)

# ---------------------------------------------------------------------------
# Step 7
# ---------------------------------------------------------------------------

config_avc, avc_stable_path = pipeline.run_avc_training(solver_sgsp=solver_sgsp)

# ---------------------------------------------------------------------------
# Step 8a
# ---------------------------------------------------------------------------

if pipeline.run_avc and (config_avc is None or avc_stable_path is None):
    raise RuntimeError(
        "Step 8a requires config_avc — enable pipeline.run_avc_online_training before run_avc."
    )

solver_avc: BurgersAVC | None = pipeline.run_avc_model(config_avc)
les_avc_data_path = (
    solver_avc.master_path
    if pipeline.run_avc and solver_avc is not None
    else avc_stable_path
)

# ---------------------------------------------------------------------------
# Step 8b
# ---------------------------------------------------------------------------

if pipeline.run_avc and solver_avc is not None:
    av_history_values = solver_avc.av_history
    av_mean_value = float(np.mean(av_history_values))
else:
    av_mean_value = 0.0

fixed_av_stable_path = paths.solver_data / "LES_AVC_fixed_mean" / "stable"
fixed_av_stable_path.mkdir(parents=True, exist_ok=True)

config_avc_fixed_mean: dict | None = (
    {
        **config_avc,
        "master_path": str(fixed_av_stable_path),
        "run_objective": "avc_fixed_mean_baseline",
    }
    if pipeline.run_avc and config_avc is not None
    else None
)

solver_avc_fixed_mean: BurgersAVC | None = pipeline.run_fixed_av_baseline(
    config_avc_fixed_mean, av_mean_value
)
les_avc_fixed_mean_path = (
    solver_avc_fixed_mean.master_path
    if solver_avc_fixed_mean is not None
    else fixed_av_stable_path
)

# ---------------------------------------------------------------------------
# Save timings to txt.
# ---------------------------------------------------------------------------

pipeline.report_timings(output_path=master_path)

# ---------------------------------------------------------------------------
# Step 9: Plots
# ---------------------------------------------------------------------------

if pipeline.run_plotting:
    dns_solution, _ = read_data(directory=paths.dns_data, final_only=True)
    projected_solution = np.load(paths.projection / "solutions_projection.npy")[-1]

    plot_configs_all = build_plot_configs(
        paths=paths,
        disc_cfg=disc_cfg,
        dns_solution=dns_solution,
        projected_solution=projected_solution,
        les_sgsp_data_path=les_sgsp_data_path,
        les_avc_data_path=les_avc_data_path,
        les_avc_fixed_mean_path=les_avc_fixed_mean_path,
    )

    # Drop any solver that blew up without producing usable output.
    plot_configs_viable = [
        cfg
        for cfg in plot_configs_all
        if cfg.solution is not None or _is_viable_solution_path(cfg.data_path)
    ]

    plot_solution_comparison(
        configs=plot_configs_viable,
        output_path=paths.master,
        filename="comparison_solvers.png",
    )

    plot_configs_no_sgsp = [
        cfg
        for cfg in plot_configs_viable
        if "SGSP" not in cfg.label and "ANN" not in cfg.label
    ]
    if plot_configs_no_sgsp:
        plot_solution_comparison(
            configs=plot_configs_no_sgsp,
            output_path=paths.master,
            filename="comparison_solvers_no_sgsp.png",
            title="Comparison of DNS and LES Solutions (excl. SGSP)",
        )

    plot_energy_comparison(
        dns_dir=paths.dns_data,
        les_a_dir=paths.les_a_data,
        les_nm_dir=paths.les_nm_data,
        les_sgsp_dir=les_sgsp_data_path
        if _is_viable_solution_path(les_sgsp_data_path)
        else None,
        les_avc_dir=les_avc_data_path
        if _is_viable_solution_path(les_avc_data_path)
        else None,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )
