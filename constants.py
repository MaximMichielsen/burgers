"""Project-wide constants: simulation defaults, ML hyperparameters, and path names."""

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

STANDARD_EXTRACTION_AMOUNT: int = 10
DNS_SNAPSHOT_AMOUNT: int = 2000

TOLERANCE_RESIDUAL: float = 1e-6
TOLERANCE_UPDATE: float = 1e-6
MAXIMUM_ITERATIONS: int = 50

# ---------------------------------------------------------------------------
# DNS / LES resolution
# ---------------------------------------------------------------------------

DNS_SPATIAL_FACTOR: float = 0.5
DNS_POINTS_FACTOR: float = 1.1
DNS_TO_LES_RATIO: int = 2**6

# ---------------------------------------------------------------------------
# Predictor agent hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_UNITS: int = 64
INPUT_UNITS: int = 20
OUTPUT_UNITS: int = 5
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.001
EPOCHS: int = 150

BLOWUP_THRESHOLD: float = 1e4
BLOWUP_BUFFER_SIZE: int = 5000

# ---------------------------------------------------------------------------
# Corrector agent hyperparameters
# ---------------------------------------------------------------------------

AVC_EPOCHS: int = 200

# ---------------------------------------------------------------------------
# Path names
# ---------------------------------------------------------------------------

RUNS_FOLDER: str = "runs"
SOLVER_DATA_FOLDER: str = "solver_data"
TRAINING_DATA_FOLDER: str = "training_data"
PRE_SPLIT_FOLDER: str = "pre_split"
POST_SPLIT_FOLDER: str = "post_split"
AGENT_FOLDER: str = "agents"
AVC_CORRECTOR_FOLDER: str = "corrector"
A_PRIORI_FOLDER: str = "apriori"

DNS_SAVE_PATH: str = "DNS"
LES_ANALYTICAL_SAVE_PATH: str = "LES_A"
LES_NO_MODEL_SAVE_PATH: str = "LES_NM"
LES_ANN_SAVE_PATH: str = "LES_SGSP"
LES_AVCG_SAVE_PATH: str = "LES_AVCG"
LES_AVCL_SAVE_PATH: str = "LES_AVCL"

LES_ANN_UNCLIPPED_FOLDER: str = "unclipped"
LES_ANN_PUSULURI_FOLDER: str = "pusuluri"
LES_ANN_RAJAMPETA_FOLDER: str = "rajampeta"
STABLE_FOLDER: str = "stable"
BLOWN_UP_FOLDER: str = "blown_up"

INPUT_STENCIL: str = "X.npy"
OUTPUT_STENCIL: str = "y.npy"
NORM_STATS: str = "norm_stats.npz"
