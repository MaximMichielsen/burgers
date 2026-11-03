"""ANN Architecture Class."""

from pathlib import Path

import torch
from torch import Tensor, nn

from ml.original.ann_config import (
    TauANNConfig,
)
from proofing.action_based.config import ANNConfigProof
from proofing.contextual_bandit_reformulation.config import ANNBanditConfig


class TauANN(nn.Module):
    """MLP policy πθ : S → A for the Coefficient Controller.

    Maps state sₙ to physical action vector a ∈ [min_action, max_action]
    using a Tanh activation mapped linearly to physical action bounds.
    """

    def __init__(
        self,
        config: TauANNConfig | ANNConfigProof | ANNBanditConfig,
    ):
        super().__init__()

        self.config = config
        self.max_action = config.max_action
        self.min_action = config.min_action
        self.state_dim = config.state_dimension
        self.action_dim = config.action_dimension
        self.hidden_dim = config.hidden_dimension

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        # Action mapping constants: a = center + half * a_norm
        self.act_center = 0.5 * (self.max_action + self.min_action)
        self.act_half = 0.5 * (self.max_action - self.min_action)

    def forward(self, state_input: Tensor) -> Tensor:
        """
        Compute the TD3 actor's action output.

        Maps raw network output to normalized range [-1, 1] via tanh,
        then scales to physical range [min_action, max_action].
        """
        raw_out = self.network(state_input)

        if isinstance(self.config, ANNBanditConfig):
            if self.config.link_coefficients:
                raw_out = raw_out.mean(dim=-1, keepdim=True).expand(-1, self.action_dim)

        a_norm = torch.tanh(raw_out)
        return self.act_center + self.act_half * a_norm


def save_tau_ann(model: TauANN, save_path: Path) -> None:
    """Save tau-ann to save_path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
        },
        save_path,
    )


def load_tau_ann(model_path: Path) -> TauANN:
    """Load tau-ann from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TauANN(
        config=checkpoint["config"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
