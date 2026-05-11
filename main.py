"""Main pipeline for thesis."""

from pathlib import Path

import numpy as np
import torch

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
)
from data_generation.configurations import (
    create_code_test_config,
    create_solver_configs,
)
from functions import run_config, read_data, SolutionConfig, plot_solution_comparison
from problems.problem_creator import problem_robijns_one
from projection_and_stencils.project import run_projection
from projection_and_stencils.split_training_data import (
    save_splits,
    split_data_shuffled,
    verify_splits,
)

# ── Pipeline flags ────────────────────────────────────────────────────────────
test_pipeline: bool = False
generate_data_dns: bool = True
run_les_models: bool = True

compare_solvers: bool = True

create_projection: bool = True
perform_split: bool = False
perform_training: bool = False

# ── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent.resolve()


if __name__ == "__main__":
    problem: dict = problem_robijns_one

    # ── DNS data generation ───────────────────────────────────────────────────
    if generate_data_dns and run_les_models:
        if not test_pipeline:
            config_dns, config_les_analytical, config_les_no_model = (
                create_solver_configs(problem_definition=problem)
            )
        else:
            config_dns = create_code_test_config()

        solver_data_path, run_folder = run_config(config_dns)
        if run_les_models:
            solver_data_path_les_analytical, run_folder_les_analytical = run_config(
                config_les_analytical
            )
            solver_data_path_les_no_model, run_folder_no_model = run_config(
                config_les_no_model
            )
    else:
        run_folder = "run_dns_n512_0507_162249"  # hard-coded directory
        solver_data_path = CURRENT_DIR / RUNS_FOLDER / run_folder / SOLVER_DATA_FOLDER

    # ── Derived paths ─────────────────────────────────────────────────────────
    training_data_path = CURRENT_DIR / RUNS_FOLDER / run_folder / TRAINING_DATA_FOLDER
    pre_split_path = training_data_path / PRE_SPLIT_FOLDER
    post_split_path = training_data_path / POST_SPLIT_FOLDER

    # ── Projection / stencils ─────────────────────────────────────────────────
    if create_projection or test_pipeline:
        if not solver_data_path.exists():
            raise FileNotFoundError(f"DNS data not found at: {solver_data_path}")

        X, y, stats, projected_solution = run_projection(
            solver_data_path, save=True, output_dir=pre_split_path
        )

    # ── Compare models ────────────────────────────────────────────────────────

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
    ]

    fig, ax = plot_solution_comparison(configs, output_path=solver_data_path)

    # ── Train/test split ──────────────────────────────────────────────────────
    if perform_split or test_pipeline:
        if not create_projection:
            X = np.load(pre_split_path / INPUT_STENCIL)
            y = np.load(pre_split_path / OUTPUT_STENCIL)

        splits = split_data_shuffled(x_input=X, y_target=y)
        save_splits(output_dir=post_split_path, splits=splits)
        verify_splits(post_split_path)

    # ── Predictor training ────────────────────────────────────────────────────
    if perform_training or test_pipeline:
        model_save_path = (
            CURRENT_DIR / RUNS_FOLDER / run_folder / AGENTS_FOLDER / PREDICTOR_FOLDER
        )
        model_save_path.mkdir(parents=True, exist_ok=True)

        model, train_stats, test_data = train_and_diagnose(post_split_path)
        test_preds = evaluate_test_performance(
            model, test_data, output_dir=model_save_path
        )

        torch.save(model.state_dict(), model_save_path / "sgs_mlp_model.pth")
        np.save(model_save_path / "training_history.npy", train_stats)
        print(f"Model saved to: {model_save_path / 'sgs_mlp_model.pth'}")

        plot_training_diagnostics(model_save_path, train_stats)
