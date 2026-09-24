from dataclasses import dataclass


@dataclass
class TauANNHyperparameters:
    """Hyperparameters tied directly to the ANN, regardless of training mechanism."""

    max_action: float = 1.0
    min_action: float = (0.0 + 1e-3) / 2

    smoothing_factor = 0.002
    burn_in_steps = 300
    weight_improvement = 1e3
    weight_absolute_error = 1.0
    weight_spectral = 1.0
    weight_action = 0.1
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
    total_episodes: int = 100
    batch_size: int = 64
    expl_noise: float = 0.1
    replay_buffer_max_size: int = int(1e5)