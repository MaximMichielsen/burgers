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
)
from data_generation.configurations import create_code_test_config, create_dns_config
from functions import run_config
from problems.problem_creator import problem_robijns_one
from projection_and_stencils.project import run_projection
from projection_and_stencils.split_training_data import (
    save_splits,
    split_data_shuffled,
    verify_splits,
)

# ── Pipeline flags ────────────────────────────────────────────────────────────
test_pipeline: bool = True
generate_data_dns: bool = True

create_projection: bool = True
perform_split: bool = False
perform_training: bool = False

# ── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent.resolve()


if __name__ == "__main__":
    problem: dict = problem_robijns_one

    # ── DNS data generation ───────────────────────────────────────────────────
    if generate_data_dns:
        config = (
            create_code_test_config()
            if test_pipeline
            else create_dns_config(problem_definition=problem)
        )
        solver_data_path, run_folder = run_config(config)
    else:
        run_folder = "run_dns_n512_0507_162249"  # hard-coded directory
        solver_data_path = CURRENT_DIR / RUNS_FOLDER / run_folder / SOLVER_DATA_FOLDER

    # ── Derived paths ─────────────────────────────────────────────────────────
    predictor_root = CURRENT_DIR / RUNS_FOLDER / run_folder / PREDICTOR_FOLDER
    pre_split_path = predictor_root / PRE_SPLIT_FOLDER
    post_split_path = predictor_root / POST_SPLIT_FOLDER

    # ── Projection / stencils ─────────────────────────────────────────────────
    if create_projection or test_pipeline:
        if not solver_data_path.exists():
            raise FileNotFoundError(f"DNS data not found at: {solver_data_path}")

        X, y, stats = run_projection(
            solver_data_path, save=True, output_dir=pre_split_path
        )

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
        test_preds = evaluate_test_performance(model, test_data)

        torch.save(model.state_dict(), model_save_path / "sgs_mlp_model.pth")
        np.save(model_save_path / "training_history.npy", train_stats)
        print(f"Model saved to: {model_save_path / 'sgs_mlp_model.pth'}")

        plot_training_diagnostics(train_stats)
