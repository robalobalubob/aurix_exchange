"""Focused tests for action selection and Double-DQN target semantics.

The tests use small table-driven networks so every Q-value has an intentional
meaning.  This makes failures diagnostic: they identify whether masking,
online-network selection, target-network evaluation, or bootstrapping changed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.train_dqn import compute_loss, select_action


class _TableQNet(nn.Module):
    """Return Q-values keyed by integer observation feature zero."""

    def __init__(self, table: Mapping[int, Sequence[float]]) -> None:
        super().__init__()
        self._table = {
            key: torch.tensor(values, dtype=torch.float32)
            for key, values in table.items()
        }

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rows = [self._table[int(key)] for key in obs[..., 0].tolist()]
        return torch.stack(rows)


class _FailIfCalledNet(nn.Module):
    """Guard that exploration does not unnecessarily evaluate the network."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise AssertionError(
            "network should not run during forced exploration"
        )


def _batch(
    *,
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    next_masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a complete, uniformly weighted loss batch."""
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "next_obs": next_obs,
        "dones": dones,
        "next_masks": next_masks,
        "weights": np.ones(len(actions), dtype=np.float32),
    }


@pytest.mark.correctness
def test_greedy_action_selection_excludes_invalid_high_q_action() -> None:
    """The highest raw Q-value must lose if its action is invalid."""
    net = _TableQNet({0: [1.0, 100.0, 5.0, 50.0]})
    obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mask = np.array([True, False, True, False], dtype=np.bool_)

    action = select_action(
        net, obs, mask, eps=0.0, rng=np.random.default_rng(0)
    )

    assert action == 2


@pytest.mark.correctness
def test_exploration_samples_only_valid_actions() -> None:
    """Epsilon exploration samples the mask, not the whole action space."""
    obs = np.zeros(3, dtype=np.float32)
    mask = np.array([False, True, False, True], dtype=np.bool_)
    rng = np.random.default_rng(91)

    actions = {
        select_action(_FailIfCalledNet(), obs, mask, eps=1.0, rng=rng)
        for _ in range(100)
    }

    assert actions == {1, 3}


@pytest.mark.correctness
def test_double_dqn_uses_masked_online_choice_and_target_evaluation() -> None:
    """Online selects a valid action; target evaluates that exact action."""
    online = _TableQNet(
        {
            0: [1.0, 9.0, 3.0],
            1: [4.0, 8.0, 2.0],
            10: [10.0, 100.0, 20.0],
            11: [50.0, 40.0, 30.0],
        }
    )
    target = _TableQNet(
        {
            10: [1_000.0, 2_000.0, 7.0],
            11: [100.0, 5.0, 200.0],
        }
    )
    batch = _batch(
        obs=np.array([[0.0], [1.0]], dtype=np.float32),
        actions=np.array([0, 2], dtype=np.int64),
        rewards=np.array([1.0, -2.0], dtype=np.float32),
        next_obs=np.array([[10.0], [11.0]], dtype=np.float32),
        dones=np.zeros(2, dtype=np.float32),
        next_masks=np.array(
            [[True, False, True], [False, True, True]],
            dtype=np.bool_,
        ),
    )

    loss, td_errors = compute_loss(online, target, batch, gamma=0.9)

    # Row zero: online picks action 2 after action 1 is masked; target says 7.
    # Row one: online picks action 1; target says 5 but prefers action 2.
    expected_td = np.array(
        [1.0 + 0.9 * 7.0 - 1.0, -2.0 + 0.9 * 5.0 - 2.0]
    )
    np.testing.assert_allclose(td_errors, expected_td, rtol=1e-6)
    expected_huber = np.where(
        np.abs(expected_td) < 1.0,
        0.5 * expected_td ** 2,
        np.abs(expected_td) - 0.5,
    ).mean()
    assert float(loss.item()) == pytest.approx(expected_huber, rel=1e-6)


@pytest.mark.correctness
def test_only_true_termination_zeroes_bootstrap() -> None:
    """Terminations stop bootstrapping; truncations retain it.

    Replay stores a single ``done`` flag whose contract is ``terminated`` only.
    The second row is a time-limit truncation, so it deliberately carries
    ``done=False`` even though the rollout itself ended.
    """
    online = _TableQNet(
        {
            0: [0.0, -1.0],
            1: [0.0, -1.0],
            10: [1.0, 2.0],
            11: [1.0, 2.0],
        }
    )
    target = _TableQNet({10: [5.0, 10.0], 11: [5.0, 10.0]})
    batch = _batch(
        obs=np.array([[0.0], [1.0]], dtype=np.float32),
        actions=np.array([0, 0], dtype=np.int64),
        rewards=np.array([2.0, 2.0], dtype=np.float32),
        next_obs=np.array([[10.0], [11.0]], dtype=np.float32),
        dones=np.array([1.0, 0.0], dtype=np.float32),
        next_masks=np.ones((2, 2), dtype=np.bool_),
    )

    _, td_errors = compute_loss(online, target, batch, gamma=0.5)

    np.testing.assert_allclose(td_errors, [2.0, 7.0], rtol=1e-6)
