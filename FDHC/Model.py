import torch
import torch.nn as nn


class ValueModel(nn.Module):
    """
    Value network predicting (v_seller, v_buyer) in [-1, 1] from the PUBLIC state vector.
    Privacy-safe by design: reserves are never part of the state.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
