from dataclasses import dataclass

from ml.ann_config import TauANNConfig


@dataclass
class ANNSingleActionConfig(TauANNConfig):
    n_random_episodes = 2
    n_skip_steps = 0
    proof_mode = "a"

    max_action = 5.0
    min_action = 0.9

    @property
    def n_agent_steps_per_episode(self) -> int:
        return 1
