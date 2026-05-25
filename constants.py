"""Constants used over codebase."""

from math import pi

import numpy as np
from numpy.typing import NDArray

# SIMULATION PARAMETERS
SIMULATION_DURATION: float = 10
SIMULATION_LENGTH: float = 2 * pi
REYNOLDS: int = 180
VISCOSITY_UNIT: float = 1 * SIMULATION_LENGTH / REYNOLDS

STANDARD_EXTRACTION_AMOUNT: int = 10
DNS_SNAPSHOT_AMOUNT: int = 2000

TOLERANCE_RESIDUAL: float = 1e-6
TOLERANCE_UPDATE: float = 1e-6
MAXIMUM_ITERATIONS: int = 50

# DNS SPECIFICS
N_NODES_DNS_POWER = 9
N_NODES_DNS: int = 2**N_NODES_DNS_POWER

MESH_DNS: NDArray
DELTA_X_DNS: float
MESH_DNS, DELTA_X_DNS = np.linspace(
    start=0, stop=SIMULATION_LENGTH, num=N_NODES_DNS, retstep=True
)
DNS_SPATIAL_FACTOR: float = 0.5
DNS_POINTS_FACTOR: float = 1.1

# LES SPECIFICS
DNS_TO_LES_RATIO: int = 2**6
N_NODES_LES: int = int(N_NODES_DNS / DNS_TO_LES_RATIO)

MESH_LES: NDArray
DELTA_X_LES: float
MESH_LES, DELTA_X_LES = np.linspace(
    start=0, stop=SIMULATION_LENGTH, num=N_NODES_LES, retstep=True
)
N_NODES_LES_FINE: int = int(N_NODES_LES * 2)
N_NODES_LES_COARSE: int = int(N_NODES_LES / 2)
NODES_LIST: list[int] = [N_NODES_LES, N_NODES_LES_COARSE, N_NODES_LES_FINE]

# PREDICTOR AGENT
HIDDEN_UNITS: int = 64
INPUT_UNITS: int = 20
OUTPUT_UNITS: int = 5
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.001
EPOCHS: int = 150

# PATH NAMING
RUNS_FOLDER: str = "runs"
SOLVER_DATA_FOLDER: str = "solver_data"
TRAINING_DATA_FOLDER: str = "training_data"
PREDICTOR_FOLDER: str = "predictor"
PRE_SPLIT_FOLDER: str = "pre_split"
POST_SPLIT_FOLDER: str = "post_split"
AGENTS_FOLDER: str = "ml_agents"
INPUT_STENCIL: str = "X.npy"
OUTPUT_STENCIL: str = "y.npy"
NORM_STATS: str = "norm_stats.npz"

DNS_SAVE_PATH: str = "DNS"
LES_ANALYTICAL_SAVE_PATH: str = "LES_A"
LES_NO_MODEL_SAVE_PATH: str = "LES_NM"
LES_ANN_SAVE_PATH: str = "LES_ANN"
PREDICTOR_AGENT_FOLDER: str = "predictor"
A_PRIORI_FOLDER: str = "apriori"

LES_ANN_UNCLIPPED_FOLDER: str = "unclipped"
LES_ANN_PUSULURI_FOLDER: str = "pusuluri"
LES_ANN_RAJAMPETA_FOLDER: str = "rajampeta"

LES_ANN_STABLE_FOLDER: str = "stable"
LES_ANN_BLOWN_UP_FOLDER: str = "blown_up"
