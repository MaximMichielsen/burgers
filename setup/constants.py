# TEMPORAL
T_ZERO = 308.0101
DT_DNS = 1e-4
DT_LES = 1e-3

# SPATIAL
N_NODES_DNS = 513
N_NODES_LES = 65
Z_LENGTH_DNS = 2.0

# SOLVER STANDARDS
N_GAUSS = 3
NR_RELAXATION = 1
FIXED_HOMOGENEOUS_BCS: tuple[float, float] = (0, 0)
TOLERANCE_RESIDUAL: float = 1e-3

# PATHING
PATH_TO_LES_CACHE = r"C:\Users\poopy\PycharmProjects\burgers\solver_data\les_caches"
PATH_TO_DNS_REF_PROFILES = (
    r"C:\Users\poopy\PycharmProjects\burgers\runs\DNS\filtered\dns_intervals"
)

# INTERVALS
INTERVALS = (
    (308.0101, 309.01),
    (309.01, 310.01),
    (310.01, 311.01),
    (311.01, 312.01),
    (312.01, 313.01),
    (313.01, 314.01),
    (314.01, 314.99),
)

# PLOTTING
SINGLE_FIG_SIZE = (5, 3.5)
DOUBLE_FIG_SIZE = (10, 3.5)

STYLE_REFERENCE = {
    "color": "grey",
    "linestyle": "-",
    "linewidth": 1,
    "alpha": 0.8,
}

STYLE_VERMILLION = {
    "color": "#D55E00",
    "linestyle": "--",
    "marker": "s",
    "linewidth": 1.3,
    "markersize": 4.5,
    "markerfacecolor": "white",
    "markeredgewidth": 1.1,
}

STYLE_BLUE = {
    "color": "#0072B2",
    "linestyle": "--",
    "marker": "o",
    "linewidth": 1.3,
    "markersize": 4.5,
    "markerfacecolor": "white",
    "markeredgewidth": 1.1,
}

STYLE_PROJECTED = {
    "color": "#56B4E9",
    "linestyle": "-",
    "marker": "^",
    "linewidth": 1.3,
    "markersize": 4.5,
    "markerfacecolor": "white",
    "markeredgewidth": 1.1,
}

PLT_PARAMS = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
}
