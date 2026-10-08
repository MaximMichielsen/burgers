"""ANN configuration and agent hyperparameters classes."""

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


from setup.config_discretization import DiscretizationConfig
from solvers.solver_base import TauModel

MAX_ACTION = 1.4
MIN_ACTION = 0.6

SMOOTHING_FACTOR = 0.002


class Scope(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass
class TauANNHyperparameters:
    """Hyperparameters tied directly to the ANN, regardless of training mechanism."""

    max_action: float = MAX_ACTION
    min_action: float = MIN_ACTION + 1e-6

    init_factor_stochastic: float = 0.2

    init_max_action: float = 1.0 + (MAX_ACTION - 1.0) * init_factor_stochastic
    init_min_action: float = 1.0 - (1.0 - MIN_ACTION) * init_factor_stochastic

    smoothing_factor = SMOOTHING_FACTOR
    burn_in_steps = 0

    weight_improvement = 10
    weight_absolute_error = 1.0
    weight_spectral = 0.0
    weight_action = 0.0
    gamma = 5.0 / 3.0


@dataclass
class TD3Hyperparameters(TauANNHyperparameters):
    """Hyperparameters for TD3 Agent training and environment interactions."""

    # Agent / Optimization Params
    lr: float = 1e-3
    discount: float = 0.99
    tau_polyak: float = 0.005
    policy_noise: float = 0.15
    noise_clip: float = 0.3
    policy_freq: int = 2

    # Training / Environment Setup Params
    batch_size: int = 64
    expl_noise: float = 0.1
    replay_buffer_max_size: int = int(1e5)
    stochastic_timesteps = 1000

    updates_per_step = 0


@dataclass
class TauANNConfig:
    """Config file for ANN simulation parameters"""

    tau_model: TauModel
    disc_config: DiscretizationConfig
    ann_path: Path | None

    n_skip_steps: int
    n_training_episodes: int
    output_scope: Scope
    input_scope: Scope

    training_mode: bool

    max_action: float = MAX_ACTION
    min_action: float = MIN_ACTION

    smoothing_factor: float = SMOOTHING_FACTOR

    n_total_allowed_episodes = 500

    n_local_action_groups: int | None = None
    local_stencil_size: int | None = None

    run_final_evaluation: bool = True

    group_map: NDArray = field(init=False, repr=False)

    @property
    def n_elements(self) -> int:
        return self.disc_config.n_nodes_les - 1

    @property
    def n_agent_steps_per_episode(self) -> int:
        return ceil(self.disc_config.n_timesteps / self.n_skip_steps)

    def __post_init__(self):
        if self.n_training_episodes > self.n_total_allowed_episodes:
            raise ValueError(
                f"Amount of training episodes ({self.n_training_episodes}) is higher than maximum allowable value ({self.n_total_allowed_episodes})!"
                f"\nSet the amount of training episodes lower or increase maximum allowed amount."
            )

        self.n_coefficients = self.tau_model.output_dimensions

        if self.output_scope == Scope.GLOBAL:
            self.n_local_action_groups = 1

        elif self.output_scope in (Scope.LOCAL, Scope.HYBRID):
            if self.n_local_action_groups is None or self.local_stencil_size is None:
                raise TypeError(
                    f"Set n_local_action_groups ({self.n_local_action_groups}) and local_stencil_size ({self.local_stencil_size})"
                )

            else:
                self.n_local_action_groups = max(
                    1, min(self.n_local_action_groups, self.n_elements)
                )
        else:
            raise ValueError(
                f"Invalid output scope received: {self.output_scope} | "
                f"choose from: {', '.join(scope.name for scope in Scope)}"
            )

        assert self.n_local_action_groups is not None
        self.group_map = self._create_group_map(
            n_elements=self.n_elements, n_groups=self.n_local_action_groups
        )

        self._set_action_dimension_size()
        self._set_state_dimension_size()

        self.hidden_dimension = max(64, int(self.state_dimension * 1.5))

    def _set_action_dimension_size(self) -> None:
        """Set the size of the action dimension the ANN will use based on the output scope."""
        if self.output_scope == Scope.GLOBAL:
            self.action_dimension = self.n_coefficients
        else:
            self.action_dimension = (
                self.n_coefficients * (self.n_local_action_groups + 1)
                if self.output_scope == Scope.HYBRID
                else self.n_coefficients * self.n_local_action_groups
            )

    def _set_state_dimension_size(self) -> None:
        """Set the size of the state dimension."""
        local_input_stencil_dimension = 0
        if self.input_scope == Scope.LOCAL:
            # u_local (n_points) + u_x_local (n_points) per node
            features_per_node = 2 * self.local_stencil_size
            local_input_stencil_dimension = (
                self.disc_config.n_nodes_les * features_per_node
            )

        self.state_dimension = (
            self.disc_config.n_wavenumber_bins
            + local_input_stencil_dimension
            + self.action_dimension
            + self.disc_config.n_nodes_les
        )

    @staticmethod
    def _create_group_map(n_elements: int, n_groups: int) -> NDArray:
        """Builds a 1D mapping array mapping each element index to a group ID."""
        base_size = n_elements // n_groups
        remainder = n_elements % n_groups

        group_map = np.zeros(n_elements, dtype=int)
        current_element = 0

        for g in range(n_groups):
            group_size = base_size + (1 if g < remainder else 0)
            group_map[current_element : current_element + group_size] = g
            current_element += group_size

        return group_map
