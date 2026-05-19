# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research codebase for solving the 1D Burgers' equation using FEM, comparing DNS (Direct Numerical Simulation) and LES (Large Eddy Simulation) approaches, with an ANN-based SGS (sub-grid scale) model trained on projected DNS data.

## Project Context
MSc thesis on Burgers equation with Galerkin discretization.
Predictive AI agent learns sub-grid scales but is numerically 
anti-dissipative online. Goal: RL-based corrector model that 
adds artificial viscosity to drain spurious energy.

## Code Style
- Python, Black + Ruff formatting
- Type hints on all functions
- Short docstrings
- Verbose variable names (x_var not x)

## Running the pipeline

The main pipeline is `main.py`. It runs DNS + LES simulations, projects DNS onto the LES grid, and plots a comparison:

```
python main.py
```

Output is written to `runs/run_<problem_name>_<timestamp>/`.

For a quick code-behavior test (no full simulation), use `create_code_test_config()` from `problems_and_configurations/configurations.py`.

## Stack

- Python 3.14, PyTorch, NumPy, SciPy, Matplotlib
- No package manager config exists — dependencies must be installed manually
- Linting: `ruff` (`.ruff_cache/` is present; no config file found, uses defaults)

## Architecture

### Simulation modes

`BurgersPure` (in `solvers/burgers_pure.py`) is the single FEM solver that handles all four modes controlled by `simulation_mode` in the config dict:

| Mode | Description |
|------|-------------|
| `"dns"` | Pure Galerkin, fine grid |
| `"no_model"` | Same as dns on a coarse grid (LES without SGS model) |
| `"les"` | Galerkin + analytic VMS/τ-based stabilisation |
| `"ann"` | Galerkin + ANN-predicted SGS corrections (disables VMS) |

### Data flow

1. **Solver configs** are built by `create_solver_configs()` in `problems_and_configurations/configurations.py`, using a problem dict from `problems_and_configurations/problems.py`.
2. `run_config()` in `functions.py` instantiates `BurgersPure` and runs it; snapshots are written as `sol_t<time>.csv` files under the run directory.
3. **Projection** (`data_curation/projection.py`): DNS snapshots are box-filtered and downsampled to the LES grid to produce training data (`solutions_projection.npy`, τ_sgs arrays).
4. **Training** (`agents/predictor.py`): `SGSPredictor` (3-layer MLP) is trained offline on stencil features → SGS corrections; uses `train_and_diagnose()`.
5. **Stencil creation** (`data_curation/stencil_creation.py`): currently a stub (`create_stencils` is empty).

### Key constants (`constants.py`)

- DNS grid: `N_NODES_DNS = 2^9 = 512` nodes over `[0, 2π]`
- LES grid: `N_NODES_LES = N_NODES_DNS / 4 = 128` nodes (`DNS_TO_LES_RATIO = 4`)
- SGS predictor: `INPUT_UNITS=20`, `OUTPUT_UNITS=4`, `HIDDEN_UNITS=128`
- `REYNOLDS=180`, `SIMULATION_DURATION=10`

### Validation (`validation/`)

- `validation/validation.py` — exact-solution comparison (uses `old/burgers.py`, an older solver version)
- `validation/mms/manufactured_validation.py` — Method of Manufactured Solutions convergence study
- `validation/convergence.py` — h-refinement convergence from CSV snapshots

The validation scripts reference `fem.burgers` and `old.burgers` which are older module paths and may not run against the current solver without adjustment.

### Problem definitions

Problems are plain dicts created by `create_problem_definition()` and stored in `problems_and_configurations/problems.py`. The active problem in `main.py` is `robijns_one` (Dirichlet BCs, Re=100, uniform IC with steady forcing).

### Output structure

```
runs/
  run_<name>_<timestamp>/
    solver_data/
      DNS/       ← sol_t*.csv, config.json, *.log, post_plotting_dns.png
      LES_A/     ← same structure, mode=les
      LES_NM/    ← same structure, mode=no_model
    training_data/
      pre_split/ ← solutions_projection.npy
      post_split/← X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
```
