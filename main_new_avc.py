"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib

from ml.corrector_training.DNS_snapshot_converter import ProjectionReferenceSchedule
from ml.corrector_training.online_trainer import (
    SACConfig,
    BurgersAVCEnvironment,
    SACAgent,
    OnlineAVTrainer,
)
from ml.ml_agents.corrector import AVController, save_corrector
from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problems, Problem
from solvers.burgers_base import BurgersBase
from ml.ml_agents.solver_configs import SGSPConfig, AVCRunConfig, AVCTrainerConfig
from utils.plotting.dissipation_evolution import plot_dissipation_comparison
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.pipeline_utils import (
    run_dns,
    run_sgsp_training,
    run_sgsp_coupled_solver,
    resolve_pathing,
    load_manual_models,
)
from utils.plotting.velocity_profiles import (
    plot_solution_comparison,
    create_velocity_plot_configs,
)
from solvers.burgers_avc import BurgersAVC


CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

# -------------------- Problem and pipeline configuration ------------------------------ #
problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=2.0)

clip_pusuluri: bool = False
clip_rajampeta: bool = False

run_analytical_les: bool = False
run_no_model_les: bool = False

# general simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 0.1

# AVC (hyper-)parameters
AVC_ALPHA_MAX: float = 1/20
AVC_OUTPUT_SCALE = problem.viscosity
AVC_EPOCHS: int = 50
AVC_N_SKIP: int = 5
AVC_WARMUP_STEPS: int = 64
AVC_INCLUDE_DISS_REWARD: bool = False
AVC_ZERO_RUN: bool = False
TAU_REWARD_WARMUP: float = 0.1

# discretization config
disc_cfg = DiscretizationConfig(
    n_nodes_les,
    temporal_refinement,
    courant_les,
    problem.domain_length,
)

# pathing
paths = resolve_pathing(problem.name, CURRENT_DIR)

manual_load_sgsp_model: str = r""
training_path: str = r""
manual_load_avcg_model: str = r""

load_manual_models(paths, manual_load_sgsp_model, training_path, manual_load_avcg_model)

if __name__ == "__main__":
    # --------------------------------------- DNS & SGSP data --------------------------------------- #
    DNS_CACHE_ROOT = CURRENT_DIR / "dns_cache"
    run_dns(DNS_CACHE_ROOT, problem, disc_cfg, paths)

    # --------------------------------------- SGSP training --------------------------------------- #
    if not paths.sgsp_model.exists() and paths.training is not None:
        run_sgsp_training(
            data_path=paths.training,
            output_dir=paths.agents,
            domain_length=problem.domain_length,
            n_elements=n_nodes_les - 1,
        )

    # --------------------------------------- SGSP coupled solver --------------------------------------- #
    if paths.training is not None:
        sgsp_cfg = SGSPConfig(
            sgsp_model_path=paths.sgsp_model,
            normalization_path=paths.training,
            blown_up_path=paths.les_sgsp_data / "blown_up",
            clip_pusuluri=clip_pusuluri,
            clip_rajampeta=clip_rajampeta,
            turn_off_predictor=False,
        )
        run_sgsp_coupled_solver(problem, disc_cfg, paths.les_sgsp_data, sgsp_cfg)

    # --------------------------------------- Bare LES solvers --------------------------------------- #
    if run_analytical_les:
        les_run = BurgersBase(problem, disc_cfg, "les", paths.les_a_data)
        les_run.run_simulation()

    if run_no_model_les:
        no_model_run = BurgersBase(problem, disc_cfg, "no_model", paths.les_nm_data)
        no_model_run.run_simulation()

    # --------------------------------------- GAVC training --------------------------------------- #
    if not paths.avcg_model.exists() and paths.projection is not None:
        proj_reference_schedule = ProjectionReferenceSchedule.from_projection_directory(
            projection_dir=paths.projection,
            domain_length=problem.domain_length,
            viscosity=problem.viscosity,
            n_wavenumber_bins=(n_nodes_les + 1) // 2,
        )

        avc_trainer_cfg = AVCTrainerConfig(
            avc_model_path=paths.avcg_model,
            simulation_mode="avc",
            correction_mode="global",
            n_skip_steps=AVC_N_SKIP,
            perform_zero_run=AVC_ZERO_RUN
        )

        av_corrector_global = AVController(
            alpha_max=AVC_ALPHA_MAX,
            output_scale=AVC_OUTPUT_SCALE,
            n_wavenumber_bins=(n_nodes_les + 1) // 2,
            correction_mode="global",
            n_output_nodes=1,
        )
        paths.agents.mkdir(parents=True, exist_ok=True)
        save_corrector(av_corrector_global, paths.avcg_model)

        sac_cfg = SACConfig(
            n_skip_steps=AVC_N_SKIP,
            warmup_steps=AVC_WARMUP_STEPS,
            batch_size=128,
            include_diss_from_reward=AVC_INCLUDE_DISS_REWARD,
            tau_transient_warmup=TAU_REWARD_WARMUP,

        )

        environment_global = BurgersAVCEnvironment(
            problem=problem,
            disc_cfg=disc_cfg,
            sgsp_cfg=sgsp_cfg,
            avc_cfg=avc_trainer_cfg,
            master_path=paths.les_avc_data / "training_global",
            sac_config=sac_cfg,
            proj_ref_schedule=proj_reference_schedule,
        )
        sac_agent_global = SACAgent(
            av_corrector=av_corrector_global,
            state_dim=environment_global.state_dim,
            sac_config=sac_cfg,
        )
        trainer_global = OnlineAVTrainer(
            environment=environment_global,
            sac_agent=sac_agent_global,
            sac_config=sac_cfg,
            output_dir=paths.agents / "avcg_checkpoints",
        )
        trainer_global.train(n_episodes=AVC_EPOCHS)
        save_corrector(sac_agent_global.policy, paths.avcg_model)

    # --------------------------------------- GAVC run --------------------------------------- #
    avc_run_cfg = AVCRunConfig(avc_model_path=paths.avcg_model)
    solver_avc_global = BurgersAVC(
        problem,
        disc_cfg,
        "avc",
        paths.les_avc_data / "global",
        sgsp_cfg,
        avc_run_cfg,
    )
    solver_avc_global.run_simulation()
    solver_avc_global.post_processing()

    # -------------------------------------- Plotting --------------------------------------- #
    plot_solution_comparison(
        configs=create_velocity_plot_configs(paths, disc_cfg),
        output_path=paths.master,
        filename="comparison_dns_sgsp.png",
    )

    plot_energy_comparison(
        paths=paths,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )

    plot_dissipation_comparison(paths,
                                paths.master,
                                problem.viscosity,
                                problem.domain_length)
