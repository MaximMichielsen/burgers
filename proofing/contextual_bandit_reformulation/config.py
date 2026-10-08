from dataclasses import dataclass

from ml.ann_config import TauANNConfig, TD3Hyperparameters



BANDIT_MIN_ACTION = 0.9
BANDIT_MAX_ACTION = 5.0

@dataclass
class ANNBanditConfig(TauANNConfig):
    n_random_episodes = 8
    n_skip_steps = 0
    proof_mode = "a"

    max_action: float = BANDIT_MAX_ACTION
    min_action: float = BANDIT_MIN_ACTION

    @property
    def n_agent_steps_per_episode(self) -> int:
        return 1

@dataclass
class TD3BanditHyperparameters(TD3Hyperparameters):
    max_action: float = BANDIT_MAX_ACTION
    min_action: float = BANDIT_MIN_ACTION
    discount: float = 0.0        # no next decision
    batch_size: int = 32
    updates_per_step: int = 100  # cheap compared with one LES run
    expl_noise: float = 0.15     # fraction of action span
    critic_mse: bool = True
    reward_scale: float = 1.0    # see note below
