"""Entry point: runs the full DNS → LES → projection → training → coupled-solver pipeline."""

from pathlib import Path

import numpy as np

from constants import RUNS_FOLDER, BLOWUP_THRESHOLD, BLOWUP_BUFFER_SIZE
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.configurations import (
    create_sgsp_config,
    create_solver_configs,
)
from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problem, Problems
from solvers.burgers_sgsp import BurgersSGSP
from utils.io_utils import read_data
from utils.plot_utils import (
    build_plot_configs,
    plot_solution_comparison,
    is_viable_solution_path,
)
from utils.solver_utils import run_config

CURRENT_DIR = Path(__file__).parent.resolve()

pipeline = PipelineConfig.all_but_dns(manual_path="")  # or a custom one
problem: Problem = Problems.raj_two

# Point at your fully trained run
PRETRAINED_RUN = Path(
    r"C:\Users\poopy\PycharmProjects\burgers\runs\run_raj_two_0603_130247"
)

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()

# Override data sources
paths.dns_data = PRETRAINED_RUN / "solver_data" / "DNS"
paths.projection = PRETRAINED_RUN / "training_data" / "pre_split"
paths.training = PRETRAINED_RUN / "training_data" / "post_split"
paths.model_output = PRETRAINED_RUN / "agents"

# Skip everything except the coupled solver + plotting
pipeline.run_solvers = False
pipeline.run_projection = False
pipeline.run_training_assembly = False
pipeline.run_training_sgsp = False
pipeline.verify_apriori = True
pipeline.run_sgsp = True
pipeline.run_plotting = True

pipeline.clip_pusuluri = True
pipeline.clip_rajampeta = False
pipeline.debug_sgsp = True

disc_cfg = DiscretisationConfig(
    n_elements_les=8,
    temporal_refinement=1,
    courant_les=0.04,
    domain_length=problem.domain_length,
    initial_condition_fn=problem.initial_condition,
)

problem.domain_timespan = disc_cfg.dt_les * 3

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
    sgsp_warmup_steps=0,
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
    from ml.data_assembly.projection import run_projection
    from utils.io_utils import read_data as _read

    _, dns_times, _, _ = _read(paths.dns_data)
    run_projection(
        dns_directory=paths.dns_data,
        bc_mode=problem.boundary_condition_type,
        bc_values=problem.boundary_condition_value,
        output_directory=paths.projection,
        verify=False,
        les_snapshot_indices=np.arange(0, len(dns_times), disc_cfg.temporal_refinement),
        n_nodes_les=disc_cfg.n_nodes_les,
    )

# ---------------------------------------------------------------------------
# Step 3: Training data assembly
# ---------------------------------------------------------------------------

if pipeline.run_training_assembly:
    from ml.data_assembly.training_data_assembly import run_training_data_assembly

    run_training_data_assembly(
        projection_path=paths.projection,
        output_dir=paths.training,
        dt=disc_cfg.dt_les,
        element_size=disc_cfg.element_size_les,
    )

# ---------------------------------------------------------------------------
# Step 6: SGSP coupled solver
# ---------------------------------------------------------------------------

solver_sgsp: BurgersSGSP | None = None
if pipeline.run_sgsp:
    projected_solutions = np.load(paths.projection / "solutions_projection.npy")
    target_step = round(problem.domain_timespan / disc_cfg.dt_les)

    seed_nm2 = projected_solutions[0]  # t=0
    seed_nm1 = projected_solutions[1]  # t=dt

    solver_sgsp = BurgersSGSP(configuration=config_sgsp)
    solver_sgsp.print_configuration()

    # Write seed snapshots for inspection
    seed_dir = paths.master / "seed_snapshots"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for seed_idx, (snap, label) in enumerate(
        [(seed_nm2, "t0"), (seed_nm1, "t1")], start=1
    ):
        np.savetxt(
            seed_dir / f"seed_{label}.csv",
            np.column_stack([disc_cfg.mesh_les, snap]),
            delimiter=",",
            header="x_coordinate,velocity",
            comments="",
        )

    solver_sgsp.seed_history_from_projection(
        projected_solutions=projected_solutions[0:2],
        forcing_fn=problem.external_forcing if not problem.forcing_steady else None,
    )

    correction = solver_sgsp.compute_sgsp_contribution()
    net_convective = correction[:, 0] + correction[:, 1]
    net_temporal = correction[:, 2] + correction[:, 3]
    net_viscous = correction[:, 4]

    print("per-element corrections:")
    print(
        f"{'elem':>4} {'cross':>12} {'reynolds':>12} {'temp_L':>12} {'temp_R':>12} {'visc':>12}"
    )
    for elem_idx, row in enumerate(correction):
        print(
            f"{elem_idx:>4} {row[0]:>12.4e} {row[1]:>12.4e} {row[2]:>12.4e} {row[3]:>12.4e} {row[4]:>12.4e}"
        )

    print(f"\nnet convective sum: {net_convective.sum():.4e}")
    print(f"net temporal sum:   {net_temporal.sum():.4e}")
    print(f"net viscous sum:    {net_viscous.sum():.4e}")

    from ml.data_assembly.training_data_assembly import (
        compute_element_output_terms,
        compute_u_prime_field,
        compute_du_prime_dx,
        compute_du_prime_dt,
    )

    dns_on_les = np.load(paths.projection / "dns_on_les.npy")

    # True SGS terms at t=2 (predicting step using t=0,1,2)
    for snap_idx, u_bar in enumerate(projected_solutions[:3]):
        u_prime = compute_u_prime_field(dns_on_les[snap_idx], u_bar)

    u_bar_n = projected_solutions[2]
    u_bar_nm1 = projected_solutions[1]
    u_prime_n = compute_u_prime_field(dns_on_les[2], u_bar_n)
    u_prime_nm1 = compute_u_prime_field(dns_on_les[1], u_bar_nm1)
    du_prime_dt_n = compute_du_prime_dt(u_prime_n, u_prime_nm1, disc_cfg.dt_les)
    du_prime_dx_n = compute_du_prime_dx(u_prime_n, disc_cfg.element_size_les)
    du_bar_dt_n = (u_bar_n - u_bar_nm1) / disc_cfg.dt_les

    true_it = np.zeros((solver_sgsp.n_elements, 5))
    for elem_idx in range(solver_sgsp.n_elements):
        nl, nr = elem_idx, elem_idx + 1
        true_it[elem_idx] = compute_element_output_terms(
            u_bar_left=float(u_bar_n[nl]),
            u_bar_right=float(u_bar_n[nr]),
            u_prime_left=float(u_prime_n[nl]),
            u_prime_right=float(u_prime_n[nr]),
            du_bar_dt_left=float(du_bar_dt_n[nl]),
            du_bar_dt_right=float(du_bar_dt_n[nr]),
            du_prime_dt_left=float(du_prime_dt_n[nl]),
            du_prime_dt_right=float(du_prime_dt_n[nr]),
            du_prime_dx_left=float(du_prime_dx_n[nl]),
            du_prime_dx_right=float(du_prime_dx_n[nr]),
            element_size=disc_cfg.element_size_les,
        )

    print("\nper-element TRUE vs PREDICTED:")
    print(f"{'elem':>4} {'chan':>10} {'true':>12} {'pred':>12} {'err':>12}")
    chan_names = ["cross", "reynolds", "temp_L", "temp_R", "visc"]
    for elem_idx in range(solver_sgsp.n_elements):
        for chan_idx, chan_name in enumerate(chan_names):
            true_val = true_it[elem_idx, chan_idx]
            pred_val = correction[elem_idx, chan_idx]
            print(
                f"{elem_idx:>4} {chan_name:>10} {true_val:>12.4e} {pred_val:>12.4e} {pred_val - true_val:>12.4e}"
            )

    true_net_conv = (true_it[:, 0] + true_it[:, 1]).sum()
    true_net_temp = (true_it[:, 2] + true_it[:, 3]).sum()
    true_net_visc = true_it[:, 4].sum()

    print(f"\n{'':>20} {'true':>12} {'pred':>12} {'err':>12}")
    print(
        f"{'net convective':>20} {true_net_conv:>12.4e} {net_convective.sum():>12.4e} {net_convective.sum() - true_net_conv:>12.4e}"
    )
    print(
        f"{'net temporal':>20} {true_net_temp:>12.4e} {net_temporal.sum():>12.4e} {net_temporal.sum() - true_net_temp:>12.4e}"
    )
    print(
        f"{'net viscous':>20} {true_net_visc:>12.4e} {net_viscous.sum():>12.4e} {net_viscous.sum() - true_net_visc:>12.4e}"
    )
    print(
        f"\n{'true net total':>20} {(true_net_conv + true_net_temp + true_net_visc):>12.4e}"
    )
    print(
        f"{'pred net total':>20} {(net_convective.sum() + net_temporal.sum() + net_viscous.sum()):>12.4e}"
    )

    solver_sgsp.run_simulation()
    solver_sgsp.post_processing()

les_sgsp_data_path = (
    solver_sgsp.master_path if solver_sgsp is not None else les_sgsp_stable_path
)

if pipeline.run_plotting:
    dns_mesh, dns_times, dns_solutions, _ = read_data(directory=paths.dns_data)
    target_idx = int(np.argmin(np.abs(np.array(dns_times) - problem.domain_timespan)))
    dns_solution = dns_solutions[target_idx]

    projected_solutions = np.load(paths.projection / "solutions_projection.npy")
    target_step = round(problem.domain_timespan / disc_cfg.dt_les)
    projected_solution = projected_solutions[target_step]

    seed_nm2 = projected_solutions[0]
    seed_nm1 = projected_solutions[1]

    print(f"seed t^{{n-2}}: index=0, range={seed_nm2.min():.4f} – {seed_nm2.max():.4f}")
    print(f"seed t^{{n-1}}: index=1, range={seed_nm1.min():.4f} – {seed_nm1.max():.4f}")
    print(f"target_step: {target_step}, projected shape: {projected_solutions.shape}")

    from utils.plot_utils import SolutionConfig

    plot_configs_all = build_plot_configs(
        paths=paths,
        disc_cfg=disc_cfg,
        dns_solution=dns_solution,
        projected_solution=projected_solution,
        les_sgsp_data_path=les_sgsp_data_path,
        extra_configs=[
            SolutionConfig(
                data_path=paths.dns_data,
                label="seed t^{n-2}",
                color="steelblue",
                linestyle=":",
                marker="v",
                mesh=disc_cfg.mesh_les,
                solution=seed_nm2,
            ),
            SolutionConfig(
                data_path=paths.dns_data,
                label="seed t^{n-1}",
                color="darkcyan",
                linestyle=":",
                marker="^",
                mesh=disc_cfg.mesh_les,
                solution=seed_nm1,
            ),
        ],
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
