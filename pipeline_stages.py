"""Pipeline stage functions for the Burgers DNS → LES → AVC pipeline."""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from constants import AVC_EPOCHS
from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
from ml.data_curation.a_priori_verificiation import run_apriori_verification
from ml.data_curation.projection import run_projection
from ml.data_curation.training_data_assembly import run_training_data_assembly
from ml.ml_agents.corrector import AVCorrector, save_corrector
from ml.ml_agents.predictor import (
    SGSPredictor,
    plot_training_diagnostics,
    train_predictor,
)
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.configurations import _resolve_avc_output_paths
from problems_and_configurations.mesh_config import MeshConfig
from problems_and_configurations.problems import Problem
from solvers.burgers_avc import BurgersAVC
from solvers.burgers_sgsp import BurgersSGSP
from utils.solver_utils import run_config


def register_stages(
    pipeline: PipelineConfig,
    paths: RunPaths,
    problem: Problem,
    disc_cfg,
    les_mesh_cfg: MeshConfig,
    config_dns: dict,
    config_les: dict,
    config_les_no_model: dict,
    config_sgsp: dict,
    les_sgsp_stable_path: Path,
) -> None:
    """Attach all decorated stage functions to *pipeline* and expose them as attributes.

    Each stage is stored on the pipeline object so main.py can call them by
    name without importing individual functions.
    """

    # ------------------------------------------------------------------
    # Step 1: DNS + LES solvers
    # ------------------------------------------------------------------

    @pipeline.stage("1 · DNS + LES solvers", enabled=pipeline.run_solvers)
    def run_solvers_stage() -> None:
        """Run DNS, analytical-VMS LES, and no-model LES solvers."""
        if pipeline.run_dns:
            run_config(config_dns)
        run_config(config_les)
        run_config(config_les_no_model)

    pipeline.run_solvers_stage = run_solvers_stage

    # ------------------------------------------------------------------
    # Step 2: DNS → LES projection
    # ------------------------------------------------------------------

    @pipeline.stage("2 · DNS → LES projection", enabled=pipeline.run_projection)
    def run_projection_stage() -> None:
        """Project DNS snapshots onto the LES grid."""
        from utils.io_utils import read_data

        _, dns_times, _, _ = read_data(paths.dns_data)
        run_projection(
            directory=paths.dns_data,
            bc_mode=problem.boundary_condition_type,
            bc_values=problem.boundary_condition_value,
            output_dir=paths.projection,
            verify=False,
            les_snapshot_indices=np.arange(
                0, len(dns_times), disc_cfg.temporal_refinement
            ),
            n_nodes_les=disc_cfg.n_nodes_les,
        )

    pipeline.run_projection_stage = run_projection_stage

    # ------------------------------------------------------------------
    # Step 3: Training data assembly
    # ------------------------------------------------------------------

    @pipeline.stage("3: SGSP training assembly", enabled=pipeline.run_training_assembly)
    def run_sgsp_training_assembly() -> None:
        """Assemble (X, y) training pairs from projected DNS snapshots."""
        run_training_data_assembly(
            projection_path=paths.projection,
            output_dir=paths.training,
            dt=les_mesh_cfg.time_step,
            element_size=les_mesh_cfg.element_size,
        )

    pipeline.run_sgsp_training_assembly = run_sgsp_training_assembly

    # ------------------------------------------------------------------
    # Step 4: Train SGS predictor
    # ------------------------------------------------------------------

    @pipeline.stage("4: Train SGSP model", enabled=pipeline.run_training_sgsp)
    def run_sgsp_training() -> SGSPredictor:
        """Train the SGS predictor and save diagnostics."""
        trained_model_result, training_stats = train_predictor(
            data_path=paths.training,
            output_dir=paths.model_output,
        )
        plot_training_diagnostics(
            training_stats=training_stats,
            output_dir=paths.model_output,
        )
        return trained_model_result

    pipeline.run_sgsp_training = run_sgsp_training

    # ------------------------------------------------------------------
    # Step 5: A priori verification
    # ------------------------------------------------------------------

    @pipeline.stage("5: Verify SGSP a priori", enabled=pipeline.verify_apriori)
    def verify_sgsp_apriori(trained_model: SGSPredictor | None) -> None:
        """Verify the trained SGS predictor a priori against projection data."""
        if trained_model is None:
            from ml.ml_agents.predictor import load_predictor

            trained_model_local = load_predictor(
                paths.model_output / "sgs_predictor.pt"
            )
        else:
            trained_model_local = trained_model

        trained_model_local.eval()

        def model_predict_fn(x_array: np.ndarray) -> NDArray:
            """Wrap trained model for the a priori verification interface."""
            with torch.no_grad():
                return trained_model_local(
                    torch.tensor(x_array, dtype=torch.float32)
                ).numpy()

        run_apriori_verification(
            model_predict_fn=model_predict_fn,
            data_dir=paths.training,
            output_dir=paths.apriori,
            domain_length=problem.domain_length,
            dt=les_mesh_cfg.time_step,
            dataset_label="Validation",
            n_elements=les_mesh_cfg.n_nodes - 1,
        )

    pipeline.verify_sgsp_apriori = verify_sgsp_apriori

    # ------------------------------------------------------------------
    # Step 6: SGSP coupled solver
    # ------------------------------------------------------------------

    @pipeline.stage("6: SGSP run", enabled=pipeline.run_sgsp)
    def run_sgsp_model() -> BurgersSGSP:
        """Run the ANN-coupled SGS predictor solver."""
        solver = BurgersSGSP(
            configuration=config_sgsp,
            clip_pusuluri=pipeline.clip_pusuluri,
            clip_rajampeta=pipeline.clip_rajampeta,
        )
        solver.print_configuration()
        solver.run_simulation()
        solver.post_processing()
        return solver

    pipeline.run_sgsp_model = run_sgsp_model

    # ------------------------------------------------------------------
    # Step 7: AVC online training
    # ------------------------------------------------------------------

    @pipeline.stage(
        "7: Train AVC model (online)", enabled=pipeline.run_avc_online_training
    )
    def run_avc_training(solver_sgsp: BurgersSGSP) -> tuple[dict, Path]:
        """Train the AV corrector via online SAC reinforcement learning."""
        from ml.corrector_training.online_trainer import (
            BurgersAVCEnvironment,
            OnlineAVTrainer,
            SACAgent,
            SACConfig,
        )

        _, dns_positive_spectrum = solver_sgsp.get_positive_spectrum(
            *solver_sgsp.compute_energy_spectrum(solver_sgsp.solution)
        )
        dns_dissipation_ref = solver_sgsp.dissipation_history[-1]

        avc_stable_path_local, avc_blown_up_path = _resolve_avc_output_paths(
            paths.solver_data
        )
        n_wavenumber_bins = len(dns_positive_spectrum)

        av_corrector_model = AVCorrector(
            alpha_max=1 * problem.viscosity,
            n_wavenumber_bins=n_wavenumber_bins,
        )
        save_corrector(av_corrector_model, paths.model_output / "av_corrector.pt")

        config_local = BurgersAVC.create_avc_config(
            avc_model_path=paths.model_output / "av_corrector.pt",
            dns_energy_spectrum=dns_positive_spectrum,
            dns_dissipation=dns_dissipation_ref,
            sgsp_model_path=paths.model_output / "sgs_predictor.pt",
            normalisation_stats_path=paths.training / "normalisation_stats.npz",
            blown_up_path=str(avc_blown_up_path),
            run_objective="avc_run",
            simulation_mode="avc",
            **{
                k: v
                for k, v in config_sgsp.items()
                if k
                not in (
                    "simulation_mode",
                    "sgsp_model_path",
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
            master_path=avc_stable_path_local,
        )

        sac_config = SACConfig(n_skip_steps=5, warmup_steps=500, batch_size=64)
        dns_reference_schedule = DNSReferenceSchedule.from_directory(
            dns_dir=paths.dns_data,
            domain_length=problem.domain_length,
            viscosity=problem.viscosity,
            n_wavenumber_bins=n_wavenumber_bins,
        )
        environment = BurgersAVCEnvironment(
            solver_config=config_local,
            sac_config=sac_config,
            dns_reference_schedule=dns_reference_schedule,
        )
        sac_agent = SACAgent(
            av_corrector=av_corrector_model,
            state_dim=environment.state_dim,
            sac_config=sac_config,
        )
        trainer = OnlineAVTrainer(
            environment=environment,
            sac_agent=sac_agent,
            sac_config=sac_config,
            output_dir=paths.model_output / "avc_checkpoints",
        )
        trainer.train(n_episodes=AVC_EPOCHS)

        if pipeline.run_avc_eval:
            config_avc_trained = {
                **config_local,
                "avc_model_path": str(
                    paths.model_output / "avc_checkpoints" / "av_corrector_final.pt"
                ),
            }
            solver_eval = BurgersAVC(configuration=config_avc_trained)
            solver_eval.run_simulation()
            solver_eval.post_processing()

        return config_local, avc_stable_path_local

    pipeline.run_avc_training = run_avc_training

    # ------------------------------------------------------------------
    # Step 8a: AVC run
    # ------------------------------------------------------------------

    @pipeline.stage("8a: Run AVC model", enabled=pipeline.run_avc)
    def run_avc_model(config_avc: dict) -> BurgersAVC:
        """Run the trained AV corrector solver."""
        solver = BurgersAVC(configuration=config_avc)
        solver.run_simulation()
        solver.post_processing()
        return solver

    pipeline.run_avc_model = run_avc_model

    # ------------------------------------------------------------------
    # Step 8b: Fixed-mean AV baseline
    # ------------------------------------------------------------------

    @pipeline.stage("8b: Fixed mean AV baseline", enabled=pipeline.run_avc)
    def run_fixed_av_baseline(
        config_avc_fixed_mean: dict, av_mean_value: float
    ) -> BurgersAVC:
        """Run a fixed-mean-AV solver as a baseline for the AVC."""
        solver_fixed = BurgersAVC(
            configuration=config_avc_fixed_mean,
            correction_is_fixed=True,
            clip_pusuluri=pipeline.clip_pusuluri,
            clip_rajampeta=pipeline.clip_rajampeta,
        )
        solver_fixed.av_correction = av_mean_value
        solver_fixed.run_simulation()
        solver_fixed.post_processing()
        return solver_fixed

    pipeline.run_fixed_av_baseline = run_fixed_av_baseline
