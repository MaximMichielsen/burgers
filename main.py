"""Entry point: runs the full DNS → LES → projection → training → coupled-solver pipeline."""

from pathlib import Path

import numpy as np
import torch

from constants import RUNS_FOLDER
from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
from ml.data_curation.a_priori_verificiation import run_apriori_verification
from ml.data_curation.projection import run_projection
from ml.data_curation.training_data_assembly import run_training_data_assembly
from ml.ml_agents.corrector import AVCorrector, save_corrector
from ml.ml_agents.predictor import plot_training_diagnostics, train_predictor
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.configurations import (
    build_mesh_config,
    create_ann_config,
    create_solver_configs,
    _resolve_avc_output_paths,
)
from problems_and_configurations.mesh_config import DiscretisationConfig
from problems_and_configurations.problems import (
    Problem,
    Problems,
)
from solvers.burgers_avc import BurgersAVC
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

problem: Problem = Problems.raj_three

disc_cfg = DiscretisationConfig(
    n_elements_les=16,
    temporal_refinement=1,
    courant_les=0.5,
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
    if not pipeline.run_training:
        from ml.ml_agents.predictor import load_predictor

        trained_model = load_predictor(paths.model_output / "sgs_predictor.pt")

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

if pipeline.run_sgsp:
    solver_ann = BurgersSGSP(
        configuration=config_ann,
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
    )
    solver_ann.print_configuration()
    solver_ann.run_simulation()
    solver_ann.post_processing()

les_ann_data_path = solver_ann.master_path if pipeline.run_sgsp else les_ann_stable_path

# ---------------------------------------------------------------------------
# Step 7: AVC Training
# ---------------------------------------------------------------------------

if pipeline.run_sgsp:
    _, dns_positive_spectrum = solver_ann.get_positive_spectrum(
        *solver_ann.compute_energy_spectrum(solver_ann.solution)
    )
    dns_dissipation_ref = solver_ann.dissipation_history[-1]
else:
    buf = np.load(les_ann_stable_path / "buffer_clean_*.npz")
    dns_positive_spectrum = buf["energy_spectra"][-1]
    dns_dissipation_ref = float(buf["dissipation_values"][-1])

avc_stable_path, avc_blown_up_path = _resolve_avc_output_paths(paths.solver_data)

n_wavenumber_bins = len(dns_positive_spectrum)
model = AVCorrector(
    alpha_max=10 * problem.viscosity,
    n_wavenumber_bins=n_wavenumber_bins,
)
save_corrector(model, paths.model_output / "av_corrector.pt")

config_avc = BurgersAVC.create_avc_config(
    avc_model_path=paths.model_output / "av_corrector.pt",
    dns_energy_spectrum=dns_positive_spectrum,
    dns_dissipation=dns_dissipation_ref,
    ann_model_path=paths.model_output / "sgs_predictor.pt",
    normalisation_stats_path=paths.training / "normalisation_stats.npz",
    blown_up_path=str(avc_blown_up_path),
    run_objective="avc_run",
    simulation_mode="avc",
    **{
        k: v
        for k, v in config_ann.items()
        if k
        not in (
            "simulation_mode",
            "ann_model_path",
            "normalisation_stats_path",
            "blown_up_path",
            "objective",
            "ann_warmup_steps",
            "blowup_threshold",
            "blowup_buffer_size",
            "run_objective",
            "master_path",
        )
    },
    master_path=avc_stable_path,
)

if pipeline.run_avc_online_training:
    from ml.corrector_training.online_trainer import (
        BurgersAVCEnvironment,
        OnlineAVTrainer,
        SACAgent,
        SACConfig,
    )

    sac_config = SACConfig(
        n_skip_steps=5,
        warmup_steps=500,
        batch_size=64,
    )

    # Build time-varying DNS reference from the stored DNS snapshot CSVs.
    # This ensures the reward at control step n compares against the DNS
    # state at the same physical time, not just the terminal snapshot.
    dns_reference_schedule = DNSReferenceSchedule.from_directory(
        dns_dir=paths.dns_data,
        domain_length=problem.domain_length,
        viscosity=problem.viscosity,
        n_wavenumber_bins=n_wavenumber_bins,
    )

    # dns_reference_schedule.plot_schedule(
    #     query_times=np.linspace(0, problem.domain_timespan, 10),
    #     output_path=paths.master / "dns_schedule_preview.png",
    # )

    environment = BurgersAVCEnvironment(
        solver_config=config_avc,
        sac_config=sac_config,
        dns_reference_schedule=dns_reference_schedule,  # time-varying reference
    )
    sac_agent = SACAgent(
        av_corrector=model,
        state_dim=environment.state_dim,
        sac_config=sac_config,
    )
    trainer = OnlineAVTrainer(
        environment=environment,
        sac_agent=sac_agent,
        sac_config=sac_config,
        output_dir=paths.model_output / "avc_checkpoints",
    )
    trainer.train(n_episodes=250)

elif pipeline.run_avc_offline_training:
    print("offline training not implement!")
else:
    print("hello no training")

if pipeline.run_avc_eval:
    config_avc_trained = {
        **config_avc,
        "avc_model_path": str(
            paths.model_output / "avc_checkpoints" / "av_corrector_final.pt"
        ),
    }
    solver_avc = BurgersAVC(configuration=config_avc_trained)
    solver_avc.run_simulation()
    solver_avc.post_processing()

# ---------------------------------------------------------------------------
# Step 8a: AVC Run
# ---------------------------------------------------------------------------

if pipeline.run_avc:
    solver_avc = BurgersAVC(configuration=config_avc)
    solver_avc.run_simulation()
    solver_avc.post_processing()

les_avc_data_path = (
    solver_avc.master_path
    if (pipeline.run_avc or pipeline.run_avc_eval)
    else avc_stable_path
)

# ---------------------------------------------------------------------------
# Step 8b: Fixed-mean-AV baseline run
# ---------------------------------------------------------------------------

av_history_values = solver_avc.av_history
av_mean_value = float(np.mean(av_history_values))
print(f"Fixed AV baseline: α = {av_mean_value:.6e}  (mean of {len(av_history_values)} steps)")

fixed_av_stable_path = paths.solver_data / "LES_AVC_fixed_mean" / "stable"
fixed_av_stable_path.mkdir(parents=True, exist_ok=True)

config_avc_fixed_mean = {
    **config_avc,
    "master_path": str(fixed_av_stable_path),
    "run_objective": "avc_fixed_mean_baseline",
}

solver_avc_fixed_mean = BurgersAVC(
    configuration=config_avc_fixed_mean,
    correction_is_fixed=True,   # bypasses policy; holds av_correction constant
    clip_pusuluri=pipeline.clip_pusuluri,
    clip_rajampeta=pipeline.clip_rajampeta,
)
solver_avc_fixed_mean.av_correction = av_mean_value  # inject mean α before run
solver_avc_fixed_mean.run_simulation()
solver_avc_fixed_mean.post_processing()

les_avc_fixed_mean_path = solver_avc_fixed_mean.master_path

# ---------------------------------------------------------------------------
# Step 9: Plots
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
            les_avc_data_path=les_avc_data_path,
            les_avc_fixed_mean_path=les_avc_fixed_mean_path,
        ),
        output_path=paths.master,
    )
    plot_energy_comparison(
        dns_dir=paths.dns_data,
        les_a_dir=paths.les_a_data,
        les_nm_dir=paths.les_nm_data,
        les_ann_dir=les_ann_data_path,
        les_avc_dir=les_avc_data_path,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )
