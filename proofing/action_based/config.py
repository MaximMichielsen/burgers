from dataclasses import dataclass
from enum import Enum

from ml.ann_config import TauANNConfig, Scope


class ProofMode(str, Enum):
    a = "a"
    b = "b"
    c = "c"
    d = "d"
    e = "e"


@dataclass
class ANNConfigProof(TauANNConfig):
    proof_mode: ProofMode | None = None

    # b parameters
    upper_bound: float = 1.3
    lower_bound: float = 1.1

    # c parameters
    amplitude: float = 0.1
    frequency: float = 2
    phase: float = 0.5
    omega: float = 0.8

    # d parameters
    target_mean_1: float = 1.1
    target_mean_2: float = 1.2

    def _set_state_dimension_size(self) -> None:
        """Set the size of the state dimension."""
        action_dimension = self.action_dimension
        if self.proof_mode == "b":
            self.state_dimension = action_dimension

        elif self.proof_mode == "c":
            phase_dimension = 4  # [sin(2wt), cos(2wt), sin(2(t+phi)), cos(2(t+phi))]
            self.state_dimension = action_dimension + phase_dimension

        elif self.proof_mode == "d":
            self.state_dimension = 2 * action_dimension

        else:
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
                + action_dimension
                + self.disc_config.n_nodes_les
            )
