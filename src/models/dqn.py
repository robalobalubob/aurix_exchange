from __future__ import annotations

import torch
import torch.nn as nn

OBS_DIM = 10
N_ACTIONS = 8


class DuelingDQN(nn.Module):
    """Dueling Double DQN network.

    Outputs a raw Q-value vector of shape (batch, N_ACTIONS). No argmax is
    applied — action selection and masking are the caller's responsibility.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden: tuple[int, int] = (128, 128),
        head_hidden: int = 64,
    ) -> None:
        super().__init__()

        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(h2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(h2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: float32 tensor of shape (batch, OBS_DIM) or (OBS_DIM,)

        Returns:
            Q-values: float32 tensor of shape (batch, N_ACTIONS) or (N_ACTIONS,)
        """
        features = self.trunk(obs)
        value = self.value_head(features)           # (..., 1)
        advantage = self.advantage_head(features)   # (..., N_ACTIONS)
        # Mean-subtract advantage to separate V and A (Wang et al. 2016)
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q
