"""Main pipeline for thesis."""

from pathlib import Path

import numpy as np
import torch

import logging

from agents.predictor import (
    evaluate_test_performance,
    plot_training_diagnostics,
    train_and_diagnose,
)
from constants import (
    RUNS_FOLDER,
    SOLVER_DATA_FOLDER,
    PREDICTOR_FOLDER,
    PRE_SPLIT_FOLDER,
    POST_SPLIT_FOLDER,
    AGENTS_FOLDER,
    INPUT_STENCIL,
    OUTPUT_STENCIL,
    TRAINING_DATA_FOLDER,
    NORM_STATS,
)
from data_generation.configurations import (
    create_code_test_config,
    create_solver_configs,
)
from functions import run_config, read_data, SolutionConfig, plot_solution_comparison
from problems.problems import (
    periodic_sin_forcing_high_visc,
)
from projection_and_stencils.project import run_projection
from projection_and_stencils.split_training_data import (
    save_splits,
    split_data_shuffled,
    verify_splits,
)

from burgers_ann_coupled import Burgers as BurgersCoupled
from agents.predictor import SGSPredictor

logging.getLogger().setLevel(logging.DEBUG)  # see ANN SGS channel norms


# ── Pipeline flags ────────────────────────────────────────────────────────────
test_pipeline: bool = False

generate_data_dns: bool = True
run_les_models: bool = True

create_projection: bool = True
perform_split: bool = True
perform_training: bool = True
ann_diagnostics: bool = False

run_predictor_model: bool = True

compare_solvers: bool = True

# ── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent.resolve()


if __name__ == "__main__":
    problem: dict = periodic_sin_forcing_high_visc

    # ── DNS data generation ───────────────────────────────────────────────────
    if generate_data_dns or run_les_models:
        if not test_pipeline:
            config_dns, config_les_analytical, config_les_no_model = (
                create_solver_configs(problem_definition=problem)
            )
        else:
            config_dns = create_code_test_config()

    if generate_data_dns:
        solver_data_path, run_folder = run_config(config_dns)

    else:
        run_folder = "run_dns_n128_0511_145239"  # hard-coded directory
        solver_data_path = CURRENT_DIR / RUNS_FOLDER / run_folder / SOLVER_DATA_FOLDER

    if run_les_models:
        solver_data_path_les_analytical, run_folder_les_analytical = run_config(
            config_les_analytical
        )
        solver_data_path_les_no_model, run_folder_no_model = run_config(
            config_les_no_model
        )

    # ── Derived paths ─────────────────────────────────────────────────────────
    training_data_path = CURRENT_DIR / RUNS_FOLDER / run_folder / TRAINING_DATA_FOLDER
    pre_split_path = training_data_path / PRE_SPLIT_FOLDER
    post_split_path = training_data_path / POST_SPLIT_FOLDER

    # ── Projection / stencils ─────────────────────────────────────────────────
    if create_projection:
        if not solver_data_path.exists():
            raise FileNotFoundError(f"DNS data not found at: {solver_data_path}")

        X, y, stats, projected_solution = run_projection(
            solver_data_path, save=True, output_dir=pre_split_path, verify=False
        )

    # ── Train/test split ──────────────────────────────────────────────────────
    if perform_split:
        if not create_projection:
            X = np.load(pre_split_path / INPUT_STENCIL)
            y = np.load(pre_split_path / OUTPUT_STENCIL)

        splits = split_data_shuffled(x_input=X, y_target=y)
        save_splits(output_dir=post_split_path, splits=splits)
        verify_splits(post_split_path)

    # ── Predictor training ────────────────────────────────────────────────────
    if perform_training:
        model_save_path = (
            CURRENT_DIR / RUNS_FOLDER / run_folder / AGENTS_FOLDER / PREDICTOR_FOLDER
        )
        model_save_path.mkdir(parents=True, exist_ok=True)

        model, train_stats, test_data = train_and_diagnose(post_split_path)
        test_preds = evaluate_test_performance(
            model, test_data, output_dir=model_save_path
        )

        torch.save(model.state_dict(), model_save_path / "sgs_mlp_model.pth")
        import shutil

        stats_src = pre_split_path / (
            Path(NORM_STATS).stem + ".npz"
        )  # e.g. norm_stats.npz
        shutil.copy(stats_src, model_save_path / stats_src.name)
        print(f"Norm stats copied to: {model_save_path / stats_src.name}")

        np.save(model_save_path / "training_history.npy", train_stats)
        print(f"Model saved to: {model_save_path / 'sgs_mlp_model.pth'}")

        if ann_diagnostics:
            plot_training_diagnostics(model_save_path, train_stats)

    # ── Predictor Run ──────────────────────────────────────────────────────────
    if run_predictor_model:
        config_predictor = create_solver_configs(
            problem_definition=problem, create_predictor_config=True
        )

        if not perform_training:
            agent_folder = run_folder
            model_save_path = Path(agent_folder)
            model_save_path = "runs" / model_save_path / "agents/predictor/"

            model_path = Path(model_save_path / "sgs_mlp_model.pth")

        solver_predictor = BurgersCoupled(
            config_predictor,
            ann_model_path=model_save_path,
            ann_model_class=SGSPredictor,
        )

        # ── Run the simulation directly on the coupled solver ──
        solver_predictor.print_configuration()
        solver_predictor.run_simulation()
        solver_predictor.post_logging()

        # The solver already writes to its run_dir; expose the same paths
        # that run_config would have returned.
        predictor_data_path = solver_predictor.run_dir  # the solver_data subfolder
        predictor_folder_name = solver_predictor.run_dir.parent.name

        # ── DEBUG ──
        print("simulation_type:", solver_predictor.simulation_type)
        print("ann_model loaded:", solver_predictor.ann_model is not None)
        print(
            "use_vms would be:",
            solver_predictor.simulation_type not in ("dns", "les_ann"),
        )

    # ── Compare models ────────────────────────────────────────────────────────

    if compare_solvers:
        # Pre-load DNS once (its style differs enough to justify explicit setup)
        dns_solution, mesh_dns = read_data(directory=solver_data_path, final_only=True)
        _, mesh_les = read_data(solver_data_path_les_analytical, final_only=True)

        configs = [
            SolutionConfig(
                data_path=solver_data_path,
                label="DNS",
                color="gray",
                linestyle="-",
                marker="",  # no marker for the reference curve
                alpha=0.7,
                mesh=mesh_dns,
                solution=dns_solution,
            ),
            SolutionConfig(
                data_path=solver_data_path_les_analytical,
                label="LES - A",
                color="royalblue",
                marker="x",
            ),
            SolutionConfig(
                data_path=solver_data_path_les_no_model,
                label="LES - no model",
                color="tab:orange",
                marker=".",
            ),
            SolutionConfig(
                data_path=solver_data_path,  # unused when mesh/solution are provided
                label="LES - projection",
                color="lightgreen",
                marker="^",
                mesh=mesh_les,
                solution=projected_solution,
            ),
            SolutionConfig(
                data_path=predictor_data_path,
                label="LES - ANN",
                color="salmon",
                marker="d",
            ),
        ]

        fig, ax = plot_solution_comparison(configs, output_path=solver_data_path)
