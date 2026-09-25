from dataclasses import dataclass

MAX_ACTION = 1.0
MIN_ACTION = 0.5


@dataclass
class TauANNHyperparameters:
    """Hyperparameters tied directly to the ANN, regardless of training mechanism."""

    max_action: float = MAX_ACTION
    min_action: float = MIN_ACTION + 1e-3

    smoothing_factor = 0.002
    burn_in_steps = 100

    weight_improvement = 100
    weight_absolute_error = 1e-3
    weight_spectral = 1e-3
    weight_action = 1.0

    gamma = 5.0 / 3.0


@dataclass
class TD3Hyperparameters(TauANNHyperparameters):
    """Hyperparameters for TD3 Agent training and environment interactions."""

    # Agent / Optimization Params
    lr: float = 1e-4
    discount: float = 0.99
    tau_polyak: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # Training / Environment Setup Params
    batch_size: int = 64
    expl_noise: float = 0.1
    replay_buffer_max_size: int = int(1e5)
    stochastic_timesteps = 100
