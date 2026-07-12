"""Categorical policy network for the Phase 2 algorithm ladder (roadmap v2).

The first policy-gradient rung (REINFORCE) optimizes a stochastic policy directly;
the network outputs raw logits over the discrete action set. Softmax / sampling /
argmax happen in the caller, mirroring the DQN convention that the network emits a
raw vector and selection is the caller's responsibility (also what keeps a future
ONNX export clean — no argmax baked into the graph).

The trunk deliberately matches ``DuelingDQN`` (128-128 ReLU) so ladder comparisons
measure the algorithm, not the architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """MLP policy: observation -> action logits.

    Outputs raw logits of shape (batch, n_actions). No softmax is applied —
    sampling, argmax, and masking are the caller's responsibility.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: float32 tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns:
            logits: float32 tensor of shape (batch, n_actions) or (n_actions,)
        """
        return self.net(obs)

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Uniform interface shared with ``ActorCriticNet`` so rollout collection
        and greedy evaluation are net-agnostic."""
        return self.net(obs)


class ActorCriticNet(nn.Module):
    """Shared-trunk actor-critic: observation -> (action logits, state value).

    Used by the bootstrapped rungs of the ladder (A2C, PPO). Trunk capacity
    matches ``PolicyNet`` and ``DuelingDQN`` so the ladder compares algorithms,
    not architectures. The value head outputs V(s) with the last dimension
    squeezed away: shape (batch,) for batched input.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(h2, n_actions)
        self.value_head = nn.Linear(h2, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: float32 tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns:
            (logits (..., n_actions), value (...,)) — value has the head's
            singleton dimension squeezed.
        """
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Policy logits only (rollout collection / greedy evaluation)."""
        return self.policy_head(self.trunk(obs))
