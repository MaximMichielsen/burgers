"""Project-wide constants: simulation defaults, ML hyperparameters, and path names."""

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

TOLERANCE_RESIDUAL: float = 1e-6
TOLERANCE_UPDATE: float = 1e-6
MAXIMUM_ITERATIONS_DNS: int = 50
MAXIMUM_ITERATIONS_LES: int = 10

# ---------------------------------------------------------------------------
# DNS / LES resolution
# ---------------------------------------------------------------------------

DNS_SPATIAL_FACTOR: float = 0.5
DNS_POINTS_FACTOR: float = 1.1
DNS_TO_LES_RATIO: int = 2**5

# ---------------------------------------------------------------------------
# Predictor agent hyperparameters
# ---------------------------------------------------------------------------

NUM_HIDDEN_LAYERS = 3
HIDDEN_UNITS: int = 64
INPUT_UNITS: int = 20
OUTPUT_UNITS: int = 5
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.001
EPOCHS: int = 300

BLOWUP_THRESHOLD: float = 1e4
BLOWUP_BUFFER_SIZE: int = 5000


# ---------------------------------------------------------------------------
# Path names
# ---------------------------------------------------------------------------

RUNS_FOLDER: str = "runs"
SOLVER_DATA_FOLDER: str = "solver_data"
TRAINING_DATA_FOLDER: str = "training_data"
AGENT_FOLDER: str = "agents"
AVC_CORRECTOR_FOLDER: str = "corrector"
A_PRIORI_FOLDER: str = "apriori"

DNS_FOLDER: str = "DNS"
LES_ANALYTICAL_SAVE_PATH: str = "LES_A"
LES_NO_MODEL_SAVE_PATH: str = "LES_NM"
LES_SGSP_SAVE_PATH: str = "LES_SGSP"
LES_AVC_SAVE_PATH: str = "LES_AVC"
PROJECTION_SAVE_PATH: str = "projection"

STABLE_FOLDER: str = "stable"
BLOWN_UP_FOLDER: str = "blown_up"

INPUT_STENCIL: str = "X.npy"
OUTPUT_STENCIL: str = "y.npy"
NORM_STATS: str = "normalisation_stats.npz"
