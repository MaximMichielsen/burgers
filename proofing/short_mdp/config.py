from dataclasses import dataclass

from proofing.contextual_bandit_reformulation.config import (
    ANNBanditConfig,
    TD3BanditHyperparameters,
)



@dataclass
class ShortMDPConfig(ANNBanditConfig):
    agent_steps: int = 2
    n_random_episodes = 20

    @property
    def n_agent_steps_per_episode(self) -> int:
        return self.agent_steps


@dataclass
class TD3ShortMDPHyperparameters(TD3BanditHyperparameters):
    discount = 0.9
    updates_per_step: int = 5  # cheap compared with one LES run
