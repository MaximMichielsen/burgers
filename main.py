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
    DNS_SAVE_PATH,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    LES_ANN_SAVE_PATH,
)
from data_generation.configurations import (
    create_code_test_config,
    create_solver_configs,
)
from functions import run_config, read_data, SolutionConfig, plot_solution_comparison
from problems.problems import (
    periodic_steady_forcing_sin_high_visc,
)
from projection_and_stencils.project import run_projection
from projection_and_stencils.split_training_data import (
    save_splits,
    split_data_shuffled,
    verify_splits,
)
from burgers import Burgers
from agents.predictor import SGSPredictor

import datetime

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
    problem: dict = periodic_steady_forcing_sin_high_visc

    # ── Master Path Definition ───────────────────────────────────────────────
    if generate_data_dns:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        master_run_id = f"run_{problem['name']}_{timestamp}"
        master_path = CURRENT_DIR / RUNS_FOLDER / master_run_id
    else:
        # Update this string to the specific folder
        _manual_run_id = "run_psfshv_0512_113222"
        master_path = CURRENT_DIR / RUNS_FOLDER / _manual_run_id

    master_path.mkdir(parents=True, exist_ok=True)

    # ── Derived paths  ───────────────────────────────────────────────────
    training_data_path = master_path / TRAINING_DATA_FOLDER
    pre_split_path = training_data_path / PRE_SPLIT_FOLDER
    post_split_path = training_data_path / POST_SPLIT_FOLDER
    model_save_path = master_path / AGENTS_FOLDER / PREDICTOR_FOLDER
    predictor_data_path = master_path / SOLVER_DATA_FOLDER / LES_ANN_SAVE_PATH

    # ── DNS data generation ──────────────────────────────────────────────
    if generate_data_dns or run_les_models:
        if not test_pipeline:
            config_dns, config_les_analytical, config_les_no_model = (
                create_solver_configs(
                    problem_definition=problem, master_dir=master_path
                )
            )
        else:
            config_dns = create_code_test_config()

    if generate_data_dns:
        solver_data_path = run_config(
            config_dns, save_path=f"{SOLVER_DATA_FOLDER}/{DNS_SAVE_PATH}"
        )
    else:
        solver_data_path = master_path / SOLVER_DATA_FOLDER / DNS_SAVE_PATH
        solver_data_path_les_analytical = (
            master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH
        )
        solver_data_path_les_no_model = (
            master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH
        )

    solver_data_path = Path(solver_data_path)
    print(solver_data_path)

    # ── LES data generation ──────────────────────────────────────────────
    if run_les_models:
        solver_data_path_les_analytical = run_config(
            config_les_analytical,
            save_path=f"{SOLVER_DATA_FOLDER}/{LES_ANALYTICAL_SAVE_PATH}",
        )
        solver_data_path_les_no_model = run_config(
            config_les_no_model,
            save_path=f"{SOLVER_DATA_FOLDER}/{LES_NO_MODEL_SAVE_PATH}",
        )

    # ── Projection / stencils ─────────────────────────────────────────────────
    if create_projection:
        pre_split_path.mkdir(parents=True, exist_ok=True)
        if not solver_data_path.exists():
            raise FileNotFoundError(f"DNS data not found at: {solver_data_path}")

        X, y, stats, projected_solution = run_projection(
            solver_data_path, save=True, output_dir=pre_split_path, verify=False
        )

    # ── Train/test split ──────────────────────────────────────────────────────
    if perform_split:
        post_split_path.mkdir(parents=True, exist_ok=True)
        if not create_projection:
            X = np.load(pre_split_path / INPUT_STENCIL)
            y = np.load(pre_split_path / OUTPUT_STENCIL)

        splits = split_data_shuffled(x_input=X, y_target=y)
        save_splits(output_dir=post_split_path, splits=splits)
        verify_splits(post_split_path)

    # ── Predictor training ────────────────────────────────────────────────────
    if perform_training:
        model_save_path.mkdir(parents=True, exist_ok=True)

        model, train_stats, test_data = train_and_diagnose(post_split_path)
        test_preds = evaluate_test_performance(
            model, test_data, output_dir=model_save_path
        )

        torch.save(model.state_dict(), model_save_path / "sgs_mlp_model.pth")

        import shutil

        stats_src = pre_split_path / (Path(NORM_STATS).stem + ".npz")
        shutil.copy(stats_src, model_save_path / stats_src.name)

        np.save(model_save_path / "training_history.npy", train_stats)

        if ann_diagnostics:
            plot_training_diagnostics(model_save_path, train_stats)

    # ── Predictor Run ──────────────────────────────────────────────────────────
    if run_predictor_model:
        predictor_data_path.mkdir(parents=True, exist_ok=True)
        config_predictor = create_solver_configs(
            problem_definition=problem,
            master_dir=master_path,
            create_predictor_config=True,
        )

        config_predictor["save_path"] = f"{SOLVER_DATA_FOLDER}/{LES_ANN_SAVE_PATH}"

        solver_predictor = Burgers(
            config_predictor,
            ann_model_path=model_save_path,
            ann_model_class=SGSPredictor,
        )

        # ── Run the simulation directly on the coupled solver ──
        solver_predictor.print_configuration()
        solver_predictor.run_simulation()
        solver_predictor.post_logging()

        predictor_data_path = solver_predictor.save_path_dir

    # ── Compare models ────────────────────────────────────────────────────────
    if compare_solvers:
        print(solver_data_path)
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
                mesh=mesh_les,
            ),
            SolutionConfig(
                data_path=solver_data_path_les_no_model,
                label="LES - no model",
                color="tab:orange",
                marker=".",
                mesh=mesh_les,
            ),
            SolutionConfig(
                data_path=predictor_data_path,
                label="LES - ANN",
                color="salmon",
                marker="d",
                mesh=mesh_les,
            ),
        ]

        if create_projection:
            projection_config = SolutionConfig(
                data_path=solver_data_path,  # unused when mesh/solution are provided
                label="LES - projection",
                color="lightgreen",
                marker="^",
                mesh=mesh_les,
                solution=projected_solution,
            )
            configs.append(projection_config)

        fig, ax = plot_solution_comparison(configs, output_path=master_path)
