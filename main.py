"""Entry point: runs the full DNS → LES → projection → training → coupled-solver pipeline."""

from pathlib import Path

import numpy as np

from constants import RUNS_FOLDER, BLOWUP_THRESHOLD, BLOWUP_BUFFER_SIZE
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.configurations import (
    create_sgsp_config,
    create_solver_configs,
    create_avc_config,
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
    is_viable_solution_path,
)
from utils.solver_utils import run_config

CURRENT_DIR = Path(__file__).parent.resolve()

pipeline = PipelineConfig.all_stages(manual_path="")

pipeline.clip_pusuluri = True
pipeline.clip_rajampeta = False
pipeline.debug_sgsp = True

problem: Problem = Problems.raj_three

alpha_max_var: float = 10
output_max_var: float = 2
spectral_pen_only: bool = False

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

# ---------------------------------------------------------------------------
# Step 1: DNS + LES solvers
# ---------------------------------------------------------------------------

if pipeline.run_solvers:
    if pipeline.run_dns:
        run_config(config_dns)
    run_config(config_les)
    run_config(config_les_no_model)

# ---------------------------------------------------------------------------
# Step 2: DNS → LES projection
# ---------------------------------------------------------------------------

paths.projection.mkdir(parents=True, exist_ok=True)

if pipeline.run_projection:
    if not paths.dns_data.exists():
        raise FileNotFoundError(f"DNS data not found at: {paths.dns_data}")
    from ml.data_curation.projection import run_projection
    from utils.io_utils import read_data as _read

    _, dns_times, _, _ = _read(paths.dns_data)
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
# Step 3: Training data assembly
# ---------------------------------------------------------------------------

if pipeline.run_training_assembly:
    from ml.data_curation.training_data_assembly import run_training_data_assembly

    run_training_data_assembly(
        projection_path=paths.projection,
        output_dir=paths.training,
        dt=disc_cfg.dt_les,
        element_size=disc_cfg.element_size_les,
    )

# ---------------------------------------------------------------------------
# Step 4: Train SGS predictor
# ---------------------------------------------------------------------------

trained_model = None
if pipeline.run_training_sgsp:
    from ml.ml_agents.predictor import train_predictor, plot_training_diagnostics

    trained_model, training_stats = train_predictor(
        data_path=paths.training,
        output_dir=paths.model_output,
    )
    plot_training_diagnostics(
        training_stats=training_stats, output_dir=paths.model_output
    )

# ---------------------------------------------------------------------------
# Step 5: A priori verification
# ---------------------------------------------------------------------------

if pipeline.verify_apriori:
    import torch
    from ml.ml_agents.predictor import load_predictor
    from ml.data_curation.a_priori_verificiation import run_apriori_verification
    from numpy.typing import NDArray

    apriori_model = trained_model or load_predictor(
        paths.model_output / "sgs_predictor.pt"
    )
    apriori_model.eval()

    def _model_predict(x_array: np.ndarray) -> NDArray:
        with torch.no_grad():
            return apriori_model(torch.tensor(x_array, dtype=torch.float32)).numpy()

    run_apriori_verification(
        model_predict_fn=_model_predict,
        data_dir=paths.training,
        output_dir=paths.apriori,
        domain_length=problem.domain_length,
        dt=disc_cfg.dt_les,
        dataset_label="Validation",
        n_elements=disc_cfg.n_elements_les,
    )

# ---------------------------------------------------------------------------
# Step 6: SGSP coupled solver
# ---------------------------------------------------------------------------

solver_sgsp: BurgersSGSP | None = None
if pipeline.run_sgsp:
    solver_sgsp = BurgersSGSP(configuration=config_sgsp)
    assert solver_sgsp is not None
    solver_sgsp.print_configuration()
    solver_sgsp.run_simulation()
    solver_sgsp.post_processing()

les_sgsp_data_path = (
    solver_sgsp.master_path if solver_sgsp is not None else les_sgsp_stable_path
)

# ---------------------------------------------------------------------------
# Step 6b: SGSP diagnostics
# ---------------------------------------------------------------------------

if pipeline.run_sgsp and pipeline.debug_sgsp:
    from ml.diagnostics.sgsp_diagnostics import (
        diagnose_sgsp_predictions,
        diagnose_training_label_scale,
    )

    diagnose_training_label_scale(training_data_path=paths.training)
    solver_ann_debug = BurgersSGSP(configuration=config_sgsp)
    diagnose_sgsp_predictions(solver=solver_ann_debug, n_steps=10)

# ---------------------------------------------------------------------------
# Step 7a: AVC (global) online training
# ---------------------------------------------------------------------------

config_avc_global: dict | None = None
avc_stable_path_global: Path | None = None

if pipeline.run_avc_online_training:
    if solver_sgsp is None:
        raise RuntimeError(
            "Step 7 requires solver_sgsp — enable pipeline.run_sgsp first."
        )

    from ml.corrector_training.online_trainer import (
        BurgersAVCEnvironment,
        OnlineAVTrainer,
        SACAgent,
        SACConfig,
    )
    from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
    from ml.ml_agents.corrector import AVCorrector, save_corrector

    _, dns_positive_spectrum = solver_sgsp.get_positive_spectrum(
        *solver_sgsp.compute_energy_spectrum(solver_sgsp.solution)
    )
    dns_dissipation_ref = float(solver_sgsp.dissipation_history[-1])
    n_wavenumber_bins = len(dns_positive_spectrum)

    config_avc_global, avc_stable_path_global, _ = create_avc_config(
        config_sgsp=config_sgsp,
        avc_model_path=paths.model_output / "av_global_corrector.pt",
        dns_energy_spectrum=dns_positive_spectrum,
        dns_dissipation=dns_dissipation_ref,
        data_dir=paths.solver_data,
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
        is_global=True,
    )

    av_corrector_global_model = AVCorrector(
        alpha_max=alpha_max_var * problem.viscosity,
        output_scale=problem.viscosity * 2,
        n_wavenumber_bins=n_wavenumber_bins,
    )
    save_corrector(
        av_corrector_global_model, paths.model_output / "av_global_corrector.pt"
    )

    sac_config = SACConfig(n_skip_steps=5, warmup_steps=100, batch_size=64)
    dns_reference_schedule = DNSReferenceSchedule.from_directory(
        dns_dir=paths.dns_data,
        domain_length=problem.domain_length,
        viscosity=problem.viscosity,
        n_wavenumber_bins=n_wavenumber_bins,
    )
    assert isinstance(config_avc_global, dict)
    environment_global = BurgersAVCEnvironment(
        solver_config=config_avc_global,
        sac_config=sac_config,
        dns_reference_schedule=dns_reference_schedule,
        spectral_penalty_only=spectral_pen_only,
    )
    sac_agent_global = SACAgent(
        av_corrector=av_corrector_global_model,
        state_dim=environment_global.state_dim,
        sac_config=sac_config,
    )
    trainer_global = OnlineAVTrainer(
        environment=environment_global,
        sac_agent=sac_agent_global,
        sac_config=sac_config,
        output_dir=paths.model_output / "avcg_checkpoints",
    )
    trainer_global.train(n_episodes=50)

# ---------------------------------------------------------------------------
# Step 7b: Run AVC model
# ---------------------------------------------------------------------------

solver_avc_global: BurgersAVC | None = None
if pipeline.run_avc:
    if config_avc_global is None or avc_stable_path_global is None:
        raise RuntimeError(
            "Step 8a requires config_avc — enable pipeline.run_avc_online_training first."
        )
    solver_avc_global = BurgersAVC(configuration=config_avc_global)
    assert solver_avc_global is not None
    solver_avc_global.print_configuration()
    solver_avc_global.run_simulation()
    solver_avc_global.post_processing()

les_avc_data_path_global = (
    solver_avc_global.master_path
    if solver_avc_global is not None
    else avc_stable_path_global
)

# ---------------------------------------------------------------------------
# Step 8a: AVC (local) online training
# ---------------------------------------------------------------------------

if pipeline.run_avc_online_training:
    if solver_sgsp is None:
        raise RuntimeError(
            "Step 7 requires solver_sgsp — enable pipeline.run_sgsp first."
        )

    config_avc_local, avc_stable_path_local, _ = create_avc_config(
        config_sgsp=config_sgsp,
        avc_model_path=paths.model_output / "av_local_corrector.pt",
        dns_energy_spectrum=dns_positive_spectrum,
        dns_dissipation=dns_dissipation_ref,
        data_dir=paths.solver_data,
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
        is_global=False,
    )

    av_corrector_local_model = AVCorrector(
        alpha_max=alpha_max_var * problem.viscosity,
        output_scale=problem.viscosity * output_max_var,
        n_wavenumber_bins=n_wavenumber_bins,
        correction_mode="local",
        n_output_nodes=disc_cfg.n_nodes_les,
    )
    save_corrector(
        av_corrector_local_model, paths.model_output / "av_local_corrector.pt"
    )

    sac_config = SACConfig(n_skip_steps=5, warmup_steps=100, batch_size=64)
    dns_reference_schedule = DNSReferenceSchedule.from_directory(
        dns_dir=paths.dns_data,
        domain_length=problem.domain_length,
        viscosity=problem.viscosity,
        n_wavenumber_bins=n_wavenumber_bins,
    )
    assert isinstance(config_avc_local, dict)
    environment_local = BurgersAVCEnvironment(
        solver_config=config_avc_local,
        sac_config=sac_config,
        dns_reference_schedule=dns_reference_schedule,
        spectral_penalty_only=spectral_pen_only,
    )
    sac_agent_local = SACAgent(
        av_corrector=av_corrector_local_model,
        state_dim=environment_local.state_dim,
        sac_config=sac_config,
    )
    trainer_local = OnlineAVTrainer(
        environment=environment_local,
        sac_agent=sac_agent_local,
        sac_config=sac_config,
        output_dir=paths.model_output / "avcl_checkpoints",
    )
    trainer_local.train(n_episodes=50)

# ---------------------------------------------------------------------------
# Step 8b: Run AVC model
# ---------------------------------------------------------------------------

if pipeline.run_avc:
    if config_avc_local is None or avc_stable_path_local is None:
        raise RuntimeError(
            "Step 8a requires config_avc — enable pipeline.run_avc_online_training first."
        )
    solver_avc_local = BurgersAVC(configuration=config_avc_local)
    assert solver_avc_local is not None
    solver_avc_local.print_configuration()
    solver_avc_local.run_simulation()
    solver_avc_local.post_processing()

les_avc_data_path_local = (
    solver_avc_local.master_path
    if solver_avc_local is not None
    else avc_stable_path_local
)

# ---------------------------------------------------------------------------
# Step 7c: Fixed mean AV baseline
# ---------------------------------------------------------------------------

solver_avc_fixed_mean: BurgersAVC | None = None
if pipeline.run_avc and solver_avc_global is not None and config_avc_global is not None:
    av_mean_value = (
        float(np.mean(solver_avc_global.av_history))
        if solver_avc_global.av_history
        else 0.0
    )

    fixed_av_stable_path = paths.solver_data / "LES_AVC_fixed_mean" / "stable"
    fixed_av_stable_path.mkdir(parents=True, exist_ok=True)

    config_avc_fixed_mean = {
        **config_avc_global,
        "master_path": str(fixed_av_stable_path),
        "run_objective": "avc_fixed_mean_baseline",
    }
    solver_avc_fixed_mean = BurgersAVC(
        configuration=config_avc_fixed_mean, correction_is_fixed=True
    )
    assert solver_avc_fixed_mean is not None
    solver_avc_fixed_mean.av_correction = av_mean_value
    solver_avc_fixed_mean.run_simulation()
    solver_avc_fixed_mean.post_processing()

les_avc_fixed_mean_path = (
    solver_avc_fixed_mean.master_path
    if solver_avc_fixed_mean is not None
    else paths.solver_data / "LES_AVC_fixed_mean" / "stable"
)

# ---------------------------------------------------------------------------
# Step 9: Plots
# ---------------------------------------------------------------------------

if pipeline.run_plotting:
    dns_solution, _ = read_data(directory=paths.dns_data, final_only=True)
    projected_solution = np.load(paths.projection / "solutions_projection.npy")[-1]

    les_avc_data_path_global = (
        solver_avc_global.master_path
        if solver_avc_global is not None
        else avc_stable_path_global or paths.solver_data / "LES_AVC" / "stable"
    )

    plot_configs_all = build_plot_configs(
        paths=paths,
        disc_cfg=disc_cfg,
        dns_solution=dns_solution,
        projected_solution=projected_solution,
        les_sgsp_data_path=les_sgsp_data_path,
        les_avcg_data_path=les_avc_data_path_global,
        les_avcl_data_path=les_avc_data_path_local,
        les_avc_fixed_mean_path=les_avc_fixed_mean_path,
    )

    plot_configs_viable = [
        cfg
        for cfg in plot_configs_all
        if cfg.solution is not None or is_viable_solution_path(cfg.data_path)
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
        if is_viable_solution_path(les_sgsp_data_path)
        else None,
        les_avcg_dir=les_avc_data_path_global
        if is_viable_solution_path(les_avc_data_path_global)
        else None,
        les_avcl_dir=les_avc_data_path_local
        if is_viable_solution_path(les_avc_data_path_local)
        else None,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )
