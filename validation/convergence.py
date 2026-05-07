import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from validation import initial_condition, exact_solution, evaluate_on_mesh
from fem.burgers import Burgers

# ── Domain & physics ─────────────────────────────────────────────────────────────
LENGTH: float = 1.0
SPACE_STEP: float = 0.01
REYNOLDS: int = 180
# VISCOSITY: float = LENGTH / REYNOLDS
VISCOSITY = 1
TIMES: list[float] = [0.01, 0.1, 0.3]
TIME: float = TIMES[-1]

OUTPUT_DIR = Path(__file__).parent.parent / "data"

# ── Reference mesh & exact solutions ─────────────────────────────────────────────
ref_cords: np.ndarray = np.linspace(0, LENGTH, int(LENGTH / SPACE_STEP) + 1)
ref_ic: np.ndarray = evaluate_on_mesh(initial_condition, ref_cords, viscosity=VISCOSITY)
exact_solutions: list[np.ndarray] = [
    evaluate_on_mesh(exact_solution, ref_cords, t=t, viscosity=VISCOSITY) for t in TIMES
]

# ── Run definitions ───────────────────────────────────────────────────────────────
LES_NODE_COUNTS: list[int] = [2**5, 2**6, 2**7]


def compute_time_step(mesh: np.ndarray, max_velocity: float, viscosity: float) -> float:
    """CFL-based time step: minimum of convective and diffusive limits."""
    dx = abs(mesh[1] - mesh[0])
    return min(dx / max_velocity, dx**2 / viscosity)


def build_configs(
    node_counts: list[int],
    simulation_type: str,
    viscosity: float,
    length: float,
    time: float,
    extract_at_times: list[float],
    max_iterations: int,
) -> list[dict]:
    """Build one Burgers config per node count."""
    configs = []
    for n_nodes in node_counts:
        mesh = np.linspace(0, length, n_nodes)
        ic = evaluate_on_mesh(initial_condition, mesh, viscosity=viscosity)
        dt = compute_time_step(mesh, max_velocity=float(np.max(np.abs(ic))), viscosity=viscosity)
        configs.append(
            Burgers.create_config(
                solution_initial=ic,
                simulation_type=simulation_type,
                run_objective="data generation",
                node_amount=n_nodes,
                boundary_conditions="fixed",
                time=time,
                time_step=dt,
                length=length,
                convergence_tol_residual=1e-6,
                convergence_tol_update=1e-6,
                max_iterations=max_iterations,
                relaxation=None,
                viscosity=viscosity,
                extract_at_times=extract_at_times,
            )
        )
    return configs


def run_configs(configs: list[dict]) -> list[Burgers]:
    """Instantiate, run, and log each config. Returns solved instances."""
    solvers = []
    for config in configs:
        solver = Burgers(configuration=config)
        solver.print_configuration()
        solver.run_simulation()
        solver.post_logging()
        solvers.append(solver)
    return solvers


def write_snapshots(
    solver: Burgers,
    simulation_type: str,
    n_nodes: int,
    extract_at_times: list[float],
    output_dir: Path,
) -> list[Path]:
    """
    Write each extracted solution snapshot to a named CSV.
    Filename: {simulation_type}_N{n_nodes}_t{time}.csv  e.g. les_N64_t0.100.csv
    Returns the list of written paths.
    """
    written = []
    for t, snapshot in zip(extract_at_times, solver.extracted_solutions):
        filename = output_dir / f"{simulation_type}_N{n_nodes}_t{t:.3f}.csv"
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_coordinate", "velocity"])
            for x, u in zip(solver.node_cords, snapshot):
                writer.writerow([x, u])
        written.append(filename)
        print(f"Written: {filename}")
    return written


def read_snapshot(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a snapshot CSV, returning (x_coordinates, velocity)."""
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def snapshots_exist(node_counts: list[int], times: list[float], simulation_type: str, output_dir: Path) -> bool:
    """Return True only if every expected CSV is already on disk."""
    return all((output_dir / f"{simulation_type}_N{n}_t{t:.3f}.csv").exists() for n in node_counts for t in times)


# ── Build, run & write (skipped if data already exists) ──────────────────────────
les_configs = build_configs(
    node_counts=LES_NODE_COUNTS,
    simulation_type="les",
    viscosity=VISCOSITY,
    length=LENGTH,
    time=TIME,
    extract_at_times=TIMES,
    max_iterations=50,
)

if not snapshots_exist(LES_NODE_COUNTS, TIMES, "les", OUTPUT_DIR):
    les_solvers = run_configs(les_configs)
    for solver, n_nodes in zip(les_solvers, LES_NODE_COUNTS):
        write_snapshots(solver, "les", n_nodes, TIMES, OUTPUT_DIR)
else:
    print("All LES snapshots found on disk, skipping simulation.")

# ── Plotting from CSV ─────────────────────────────────────────────────────────────
exact_color = "black"

fig, axes = plt.subplots(1, len(TIMES), figsize=(5 * len(TIMES), 5), sharey=True)

print(exact_solutions[0])
print(exact_solutions[1])


for ax, t, t_idx in zip(axes, TIMES, range(len(TIMES))):
    ax.plot(ref_cords, exact_solutions[t_idx], color=exact_color, linestyle="--", label="Exact", zorder=3)

    for n_nodes in LES_NODE_COUNTS:
        filepath = OUTPUT_DIR / f"les_N{n_nodes}_t{t:.3f}.csv"
        x, u = read_snapshot(filepath)
        ax.plot(x, u, linestyle="-", label=f"LES N={n_nodes}")

    ax.set_title(f"t = {t}")
    ax.set_xlabel("x")
    ax.grid(True, alpha=0.4)

axes[0].set_ylabel("Velocity")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle(f"Burgers equation — LES vs Exact  (Re = {REYNOLDS})", fontsize=13)
plt.tight_layout()
plt.show()
