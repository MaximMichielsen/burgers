from dataclasses import dataclass

from proofing.contextual_bandit_reformulation.config import (
    ANNBanditConfig,
    TD3BanditHyperparameters,
)


@dataclass
class ShortMDPConfig(ANNBanditConfig):
    agent_steps: int = 2
    n_random_episodes = 30

    @property
    def n_agent_steps_per_episode(self) -> int:
        return self.agent_steps

    sweet_spot_low = 4.0
    sweet_spot_high = 5.0


@dataclass
class TD3ShortMDPHyperparameters(TD3BanditHyperparameters):
    discount = 0.99
    updates_per_step: int = 10
