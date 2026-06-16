"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray

from ml.data_assembly.training_data_assembly import run_training_data_assembly
from constants import RUNS_FOLDER
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problems, Problem
from ml.ml_agents.solver_configs import AVCTrainerConfig, SGSPConfig, AVCRunConfig

from ml.corrector_training.online_trainer import (
    BurgersAVCEnvironment,
    OnlineAVTrainer,
    SACAgent,
    SACConfig,
)
from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
from ml.ml_agents.corrector import AVController, save_corrector
from solvers.burgers_base import BurgersBase

from utils.enegy_evolution_utils import plot_energy_comparison
from utils.io_utils import read_data as _read_plot
from utils.plot_utils import (
    SolutionConfig,
    build_plot_configs,
    is_viable_solution_path,
    plot_solution_comparison,
)

from ml.data_assembly.a_priori_verificiation import run_apriori_verification
from ml.data_assembly.projection import run_projection
from ml.ml_agents.predictor import (
    plot_training_diagnostics,
    train_predictor,
)
from solvers.burgers_avc import BurgersAVC
from solvers.burgers_sgsp import BurgersSGSP


CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")

# -------------------- Problem and pipeline configuration ------------------------------ #

problem: Problem = Problems.raj_one

problem = replace(problem, domain_timespan=5.0)

pipeline = PipelineConfig.all(manual_path=r"")
pipeline.run_les_no_model = True

pipeline.clip_pusuluri = True
pipeline.clip_rajampeta = False
exclude_diss = True
set_off_predictor = False

disc_cfg = DiscretisationConfig(
    n_nodes_les=9,
    temporal_refinement=1,
    courant_les=0.01,
    domain_length=problem.domain_length,
)

PROJECTION_MODE: str = "nodal"
ALPHA_MAX: float = 100 * problem.viscosity
OUTPUT_SCALE: float = 1
AVC_EPOCHS: int = 20
N_SKIP: int = 5

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()

manual_load_dns: str = r""
paths.dns_data = Path(manual_load_dns) if manual_load_dns != "" else paths.dns_data


if __name__ == "__main__":
    # --------------------------------------- DNS --------------------------------------- #
    if pipeline.run_dns:
        solver_dns = BurgersBase(
            problem=problem,
            disc_cfg=disc_cfg,
            simulation_mode="dns",
            master_path=paths.dns_data,
        )
        solver_dns.print_configuration()
        solver_dns.run_simulation()
        solver_dns.post_processing()
    # --------------------------------------- LES --------------------------------------- #
    if pipeline.run_base_models:
        solver_les_vms = BurgersBase(
            problem=problem,
            disc_cfg=disc_cfg,
            simulation_mode="les",
            master_path=paths.les_a_data,
        )
        solver_les_vms.run_simulation()
        solver_les_vms.post_processing()

        if pipeline.run_les_no_model:
            solver_les_nm = BurgersBase(
                problem=problem,
                disc_cfg=disc_cfg,
                simulation_mode="no_model",
                master_path=paths.les_nm_data,
            )
            solver_les_nm.run_simulation()
            solver_les_nm.post_processing()
    # -------------------------------------- SGSP --------------------------------------- #
    if pipeline.run_sgsp_block:
        paths.projection.mkdir(parents=True, exist_ok=True)
        if not paths.dns_data.exists():
            raise FileNotFoundError(f"DNS data not found at: {paths.dns_data}")

        run_projection(
            dns_directory=paths.dns_data,
            output_directory=paths.projection,
            bc_mode=problem.boundary_condition_type,
            bc_values=problem.boundary_condition_value,
            disc_cfg=disc_cfg,
            verify=True,
            projection_mode=PROJECTION_MODE,
        )
        run_training_data_assembly(
            projection_path=paths.projection,
            output_dir=paths.training,
            disc_cfg=disc_cfg,
        )
        trained_model, training_stats = train_predictor(
            data_path=paths.training,
            output_dir=paths.model_output,
        )
        plot_training_diagnostics(
            training_stats=training_stats, output_dir=paths.model_output
        )
        trained_model.eval()

        def _model_predict(x_array: np.ndarray) -> NDArray:
            with torch.no_grad():
                return trained_model(torch.tensor(x_array, dtype=torch.float32)).numpy()

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
            sgsp_model_path=paths.sgsp_model,
            normalization_path=paths.normalization,
            blown_up_path=paths.les_sgsp_data / "blown_up",
            clip_pusuluri=pipeline.clip_pusuluri,
            clip_rajampeta=pipeline.clip_rajampeta,
            set_off_predictor=set_off_predictor,
        )
        solver_sgsp = BurgersSGSP(
            problem=problem,
            disc_cfg=disc_cfg,
            simulation_mode="sgsp",
            master_path=paths.les_sgsp_data,
            sgsp_cfg=sgsp_cfg,
        )
        solver_sgsp.run_simulation()
        solver_sgsp.post_processing()
    # --------------------------------------- AVC -------------------------------------- #
    if pipeline.run_avc_block:
        if not paths.sgsp_model.exists():
            raise FileNotFoundError(
                f"SGSP model not found at {paths.sgsp_model}. Run the SGSP block first."
            )

        projected_solutions = np.load(paths.projection / "solutions_projection.npy")
        dns_solution_on_les = projected_solutions[0]

        dns_solver_ref = BurgersBase(
            problem,
            disc_cfg,
            simulation_mode="no_model",
            master_path=paths.dns_data,
        )
        _, dns_positive_spectrum = dns_solver_ref.get_positive_spectrum(
            *dns_solver_ref.compute_energy_spectrum(dns_solution_on_les)
        )
        dns_dissipation_ref = float(
            dns_solver_ref.compute_dissipation(dns_solution_on_les)
        )
        n_wavenumber_bins = len(dns_positive_spectrum)

        sgsp_cfg_avc = SGSPConfig(
            sgsp_model_path=paths.sgsp_model,
            normalization_path=paths.normalization,
            blown_up_path=paths.les_avc_data / "blown_up",
            clip_pusuluri=pipeline.clip_pusuluri,
            clip_rajampeta=pipeline.clip_rajampeta,
        )
        dns_reference_schedule = DNSReferenceSchedule.from_projection_directory(
            projection_dir=paths.projection,
            domain_length=problem.domain_length,
            viscosity=problem.viscosity,
            n_wavenumber_bins=n_wavenumber_bins,
        )
        sac_config = SACConfig(
            n_skip_steps=N_SKIP,
            warmup_steps=500,  # ~5 full episodes of random exploration
            batch_size=64 * 2,  # fill faster — fine for this problem size
        )
        avc_cfg_trainer_global = AVCTrainerConfig(
            avc_model_path=paths.avcg_model,
            dns_energy_spectrum=dns_positive_spectrum,
            dns_dissipation=dns_dissipation_ref,
            correction_mode="global",
            n_skip_steps=sac_config.n_skip_steps,
            exclude_diss_from_reward=exclude_diss,
            simulation_mode="avc",
        )
        # --------------------------------------- GAVC -------------------------------------- #
        if pipeline.train_avc_online:
            av_corrector_global = AVController(
                alpha_max=ALPHA_MAX * problem.viscosity,
                output_scale=problem.viscosity * OUTPUT_SCALE,
                n_wavenumber_bins=n_wavenumber_bins,
                correction_mode=avc_cfg_trainer_global.correction_mode,
                n_output_nodes=1,
            )
            save_corrector(av_corrector_global, paths.avcg_model)
            environment_global = BurgersAVCEnvironment(
                problem=problem,
                disc_cfg=disc_cfg,
                sgsp_cfg=sgsp_cfg_avc,
                avc_cfg=avc_cfg_trainer_global,
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
            trainer_global.train(n_episodes=AVC_EPOCHS)

        # --------------------------------------- GAVC Run -------------------------------------- #

        if paths.avcg_model.exists():
            avc_run_config = AVCRunConfig(avc_model_path=paths.avcg_model)
            solver_avc_global = BurgersAVC(
                problem,
                disc_cfg,
                "avc",
                paths.les_avc_data / "global",
                sgsp_cfg_avc,
                avc_run_config,
            )
            solver_avc_global.run_simulation()
            solver_avc_global.post_processing()

        # --------------------------------------- Plotting -------------------------------------- #
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
                    color="royalblue",
                    linestyle="--",
                    marker="s",
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

        plot_energy_comparison(
            dns_dir=paths.dns_data,
            les_a_dir=paths.les_a_data,
            les_nm_dir=paths.les_nm_data
            if is_viable_solution_path(paths.les_nm_data)
            else None,
            les_sgsp_dir=paths.les_sgsp_data
            if is_viable_solution_path(paths.les_sgsp_data)
            else None,
            les_avcg_dir=global_avc_path
            if is_viable_solution_path(global_avc_path)
            else None,
            output_path=paths.master,
            viscosity=problem.viscosity,
            domain_length=problem.domain_length,
            projection_dir=paths.projection,
        )
