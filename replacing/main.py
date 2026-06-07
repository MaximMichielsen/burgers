"""Entry point: DNS → LES → projection → training → SGSP → AVC pipeline."""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ml.data_curation.training_data_assembly import run_training_data_assembly
from replacing.constants import RUNS_FOLDER, NORM_STATS
from replacing.input_settings.disc_config import DiscretisationConfig
from replacing.input_settings.problems import Problems, Problem
from replacing.input_settings.solver_configs import AVCConfig, SGSPConfig
from replacing.pipeline_settings import PipelineConfig, RunPaths
from replacing.solvers.burgers_base import BurgersBase

from utils.enegy_evolution_utils import plot_energy_comparison
from utils.io_utils import read_data as _read_plot
from utils.plot_utils import (
    SolutionConfig,
    build_plot_configs,
    is_viable_solution_path,
    plot_solution_comparison,
)

from ml.data_curation.a_priori_verificiation import run_apriori_verification
from ml.data_curation.projection import run_projection
from ml.ml_agents.predictor import (
    load_predictor,
    plot_training_diagnostics,
    train_predictor,
)
from solvers.burgers_avc import BurgersAVC
from solvers.burgers_sgsp import BurgersSGSP
from utils.io_utils import read_data as _read


CURRENT_DIR = Path(__file__).parent.resolve()

# ------------------------------------------------------------------ #
#  Problem and pipeline configuration
# ------------------------------------------------------------------ #

problem: Problem = Problems.pipeline_test
pipeline = PipelineConfig.all_stages(manual_path="")

pipeline.clip_pusuluri = True
pipeline.clip_rajampeta = False

disc_cfg = DiscretisationConfig(
    n_nodes_les=8,
    temporal_refinement=1,
    courant_les=0.01,
    domain_length=problem.domain_length,
    initial_condition_fn=problem.initial_condition,
)

ALPHA_MAX: float = 100
OUTPUT_SCALE: float = 10

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()

manual_load_dns = ""
paths.dns_data = Path(manual_load_dns) if manual_load_dns != "" else paths.dns_data

# ------------------------------------------------------------------ #
#  DNS
# ------------------------------------------------------------------ #

if pipeline.run_dns:
    solver_dns = BurgersBase(
        problem,
        disc_cfg,
        simulation_mode="dns",
        master_path=paths.dns_data,
        snapshot_factor=1,
    )
    solver_dns.run_simulation()
    solver_dns.post_processing()

# ------------------------------------------------------------------ #
#  Base LES models
# ------------------------------------------------------------------ #

if pipeline.run_base_models:
    solver_les_vms = BurgersBase(
        problem,
        disc_cfg,
        simulation_mode="les",
        master_path=paths.les_a_data,
        snapshot_factor=1,
    )
    solver_les_vms.run_simulation()
    solver_les_vms.post_processing()

    solver_les_nm = BurgersBase(
        problem,
        disc_cfg,
        simulation_mode="no_model",
        master_path=paths.les_nm_data,
        snapshot_factor=1,
    )
    solver_les_nm.run_simulation()
    solver_les_nm.post_processing()

# ------------------------------------------------------------------ #
#  SGSP block: projection → training → a priori → coupled solver
# ------------------------------------------------------------------ #

if pipeline.run_sgsp_block:
    paths.projection.mkdir(parents=True, exist_ok=True)
    if not paths.dns_data.exists():
        raise FileNotFoundError(f"DNS data not found at: {paths.dns_data}")

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

    run_training_data_assembly(
        projection_path=paths.projection,
        output_dir=paths.training,
        dt=disc_cfg.dt_les,
        element_size=disc_cfg.h_les
    )

    trained_model, training_stats = train_predictor(
        data_path=paths.training,
        output_dir=paths.model_output,
    )
    plot_training_diagnostics(
        training_stats=training_stats, output_dir=paths.model_output
    )

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

    sgsp_cfg = SGSPConfig(
        sgsp_model_path=paths.model_output / "sgs_predictor.pt",
        normalization_path=paths.training / NORM_STATS,
        blown_up_path=paths.les_sgsp_data / "blown_up",
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
    )

    solver_sgsp = BurgersSGSP(
        problem,
        disc_cfg,
        "sgsp",
        paths.les_sgsp_data,
        sgsp_cfg,
        snapshot_factor=1
    )
    solver_sgsp.run_simulation()
    solver_sgsp.post_processing()

# ------------------------------------------------------------------ #
#  AVC block
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  AVC block: online RL training → run global + local correctors
# ------------------------------------------------------------------ #

if pipeline.run_avc_block:
    sgsp_model_path = paths.model_output / "sgs_predictor.pt"
    if not sgsp_model_path.exists():
        raise FileNotFoundError(
            f"SGSP model not found at {sgsp_model_path}. Run the SGSP block first."
        )

    from ml.corrector_training.online_trainer import (
        BurgersAVCEnvironment,
        OnlineAVTrainer,
        SACAgent,
        SACConfig,
    )
    from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
    from ml.ml_agents.corrector import AVCorrector, save_corrector
    from utils.io_utils import read_data as _read_dns

    _, _, dns_solutions, _ = _read_dns(paths.dns_data)
    dns_solver_ref = BurgersBase(
        problem,
        disc_cfg,
        simulation_mode="dns",
        master_path=paths.dns_data,
    )
    _, dns_positive_spectrum = dns_solver_ref.get_positive_spectrum(
        *dns_solver_ref.compute_energy_spectrum(dns_solutions[-1])
    )
    dns_dissipation_ref = float(dns_solver_ref.compute_dissipation(dns_solutions[-1]))
    n_wavenumber_bins = len(dns_positive_spectrum)

    sgsp_cfg_avc = SGSPConfig(
        sgsp_model_path=sgsp_model_path,
        normalization_path=paths.training / NORM_STATS,
        blown_up_path=paths.les_avc_data / "blown_up",
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
    )

    dns_reference_schedule = DNSReferenceSchedule.from_directory(
        dns_dir=paths.dns_data,
        domain_length=problem.domain_length,
        viscosity=problem.viscosity,
        n_wavenumber_bins=n_wavenumber_bins,
    )

    sac_config = SACConfig(n_skip_steps=5, warmup_steps=100, batch_size=64)

    # -------------------------------------------------- #
    #  Global corrector training
    # -------------------------------------------------- #

    if pipeline.train_avc_online:
        av_corrector_global = AVCorrector(
            alpha_max=ALPHA_MAX * problem.viscosity,
            output_scale=problem.viscosity * 2,
            n_wavenumber_bins=n_wavenumber_bins,
        )
        save_corrector(
            av_corrector_global, paths.model_output / "av_global_corrector.pt"
        )

        avc_cfg_global = AVCConfig(
            avc_model_path=paths.model_output / "av_global_corrector.pt",
            dns_energy_spectrum=dns_positive_spectrum,
            dns_dissipation=dns_dissipation_ref,
        )

        environment_global = BurgersAVCEnvironment(
            problem=problem,
            disc_cfg=disc_cfg,
            sgsp_cfg=sgsp_cfg_avc,
            avc_cfg=avc_cfg_global,
            master_path=paths.les_avc_data / "training_global",
            sac_config=sac_config,
            dns_reference_schedule=dns_reference_schedule,
        )
        sac_agent_global = SACAgent(
            av_corrector=av_corrector_global,
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

    # -------------------------------------------------- #
    #  Local corrector training
    # -------------------------------------------------- #

    if pipeline.train_avc_online:
        av_corrector_local = AVCorrector(
            alpha_max=ALPHA_MAX * problem.viscosity,
            output_scale=problem.viscosity * OUTPUT_SCALE,
            n_wavenumber_bins=n_wavenumber_bins,
            correction_mode="local",
            n_output_nodes=disc_cfg.n_nodes_les,
        )
        save_corrector(
            av_corrector_local, paths.model_output / "av_local_corrector.pt"
        )

        avc_cfg_local = AVCConfig(
            avc_model_path=paths.model_output / "av_local_corrector.pt",
            dns_energy_spectrum=dns_positive_spectrum,
            dns_dissipation=dns_dissipation_ref,
        )

        environment_local = BurgersAVCEnvironment(
            problem=problem,
            disc_cfg=disc_cfg,
            sgsp_cfg=sgsp_cfg_avc,
            avc_cfg=avc_cfg_local,
            master_path=paths.les_avc_data / "training_local",
            sac_config=sac_config,
            dns_reference_schedule=dns_reference_schedule,
        )
        sac_agent_local = SACAgent(
            av_corrector=av_corrector_local,
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

    # -------------------------------------------------- #
    #  Run global corrector
    # -------------------------------------------------- #

    global_corrector_path = paths.model_output / "av_global_corrector.pt"
    if global_corrector_path.exists():
        avc_cfg_global_run = AVCConfig(
            avc_model_path=global_corrector_path,
            dns_energy_spectrum=dns_positive_spectrum,
            dns_dissipation=dns_dissipation_ref,
        )
        solver_avc_global = BurgersAVC(
            problem,
            disc_cfg,
            "avc",
            paths.les_avc_data / "global",
            sgsp_cfg_avc,
            avc_cfg_global_run,
            snapshot_factor=1,
        )
        solver_avc_global.run_simulation()
        solver_avc_global.post_processing()

    # -------------------------------------------------- #
    #  Run local corrector
    # -------------------------------------------------- #

    local_corrector_path = paths.model_output / "av_local_corrector.pt"
    if local_corrector_path.exists():
        avc_cfg_local_run = AVCConfig(
            avc_model_path=local_corrector_path,
            dns_energy_spectrum=dns_positive_spectrum,
            dns_dissipation=dns_dissipation_ref,
        )
        solver_avc_local = BurgersAVC(
            problem,
            disc_cfg,
            "avc",
            paths.les_avc_data / "local",
            sgsp_cfg_avc,
            avc_cfg_local_run,
            snapshot_factor=1,
        )
        solver_avc_local.run_simulation()
        solver_avc_local.post_processing()


# ------------------------------------------------------------------ #
#  Plotting block
# ------------------------------------------------------------------ #




    dns_solution, _ = _read_plot(directory=paths.dns_data, final_only=True)
    projected_solution = np.load(paths.projection / "solutions_projection.npy")[-1]

    # Build AVC extra configs only for solvers that actually ran
    avc_extra_configs: list[SolutionConfig] = []

    global_avc_path = paths.les_avc_data / "global"
    if is_viable_solution_path(global_avc_path):
        avc_extra_configs.append(
            SolutionConfig(
                data_path=global_avc_path,
                label="LES - AVC (global)",
                color="mediumorchid",
                linestyle=":",
                marker="s",
                mesh=disc_cfg.mesh_les,
            )
        )

    local_avc_path = paths.les_avc_data / "local"
    if is_viable_solution_path(local_avc_path):
        avc_extra_configs.append(
            SolutionConfig(
                data_path=local_avc_path,
                label="LES - AVC (local)",
                color="gold",
                linestyle="-.",
                marker="^",
                mesh=disc_cfg.mesh_les,
            )
        )

    plot_configs_all = build_plot_configs(
        paths=paths,
        disc_cfg=disc_cfg,
        dns_solution=dns_solution,
        projected_solution=projected_solution,
        les_sgsp_data_path=paths.les_sgsp_data,
        extra_configs=avc_extra_configs,
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
        les_sgsp_dir=paths.les_sgsp_data if is_viable_solution_path(paths.les_sgsp_data) else None,
        les_avcg_dir=global_avc_path if is_viable_solution_path(global_avc_path) else None,
        les_avcl_dir=local_avc_path if is_viable_solution_path(local_avc_path) else None,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )
