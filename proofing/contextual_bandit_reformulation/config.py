from dataclasses import dataclass

from ml.original.ann_config import TauANNConfig, TD3Hyperparameters


BANDIT_MIN_ACTION = 0.9
BANDIT_MAX_ACTION = 5.5


@dataclass
class ANNBanditConfig(TauANNConfig):
    link_coefficients: bool = False
    n_random_episodes = 1
    n_skip_steps = 0
    proof_mode = "a"

    max_action: float = BANDIT_MAX_ACTION
    min_action: float = BANDIT_MIN_ACTION

    @property
    def n_agent_steps_per_episode(self) -> int:
        return 1

    def _set_state_dimension_size(self):
        self.state_dimension = 1


@dataclass
class TD3BanditHyperparameters(TD3Hyperparameters):
    max_action: float = BANDIT_MAX_ACTION
    min_action: float = BANDIT_MIN_ACTION
    discount: float = 0.0  # no next decision
    batch_size: int = 32
    updates_per_step: int = 20  # cheap compared with one LES run
    expl_noise: float = 0.15  # fraction of action span
    critic_mse: bool = True
    reward_scale: float = 1.0
