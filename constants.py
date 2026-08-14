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

SGSP_NUM_HIDDEN_LAYERS = 5
SGSP_HIDDEN_UNITS: int = 256
SGSP_INPUT_UNITS: int = 20
SGSP_OUTPUT_UNITS: int = 5
SGSP_BATCH_SIZE: int = 128
SGSP_LEARNING_RATE: float = 0.001
SGSP_EPOCHS: int = 100

BLOWUP_THRESHOLD: float = 1e12
BLOWUP_BUFFER_SIZE: int = 5000

# ---------------------------------------------------------------------------
# Corrector agent hyperparameters
# ---------------------------------------------------------------------------

AVC_HIDDEN_UNITS = 64
AVC_GLOBAL_OUTPUT_UNITS = 1

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
LES_SHAKIB_ONE_SAVE_PATH: str = "LES_SHAKIB_ONE"
LES_SHAKIB_TWO_SAVE_PATH: str = "LES_SHAKIB_TWO"
LES_SHAKIB_THREE_SAVE_PATH: str = "LES_SHAKIB_THREE"
LES_NO_MODEL_SAVE_PATH: str = "LES_NM"
LES_SGSP_SAVE_PATH: str = "LES_SGSP"
LES_AVC_SAVE_PATH: str = "LES_AVC"
PROJECTION_SAVE_PATH: str = "projection"

STABLE_FOLDER: str = "stable"
BLOWN_UP_FOLDER: str = "blown_up"

INPUT_STENCIL: str = "X.npy"
OUTPUT_STENCIL: str = "y.npy"
NORM_STATS: str = "normalisation_stats.npz"
