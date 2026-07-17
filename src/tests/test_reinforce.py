"""
REINFORCE trainer test suite (Phase 2, first policy-gradient rung)
==================================================================
Markers:
  correctness — pure pieces (returns-to-go, advantages, loss) and rollout
                consistency with the canonical env must be exact.

Spec reference: docs/aurix_rl_roadmap_v2.md Phase 2
Implementation: src/training/train_reinforce.py, src/models/policy.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.ou_core import OUCoreConfig, OUTradingEnv
from src.models.policy import PolicyNet
from src.training.train_reinforce import (
    advantages,
    collect_batch,
    draw_training_reset_seed,
    greedy_policy,
    held_out_seed_intervals,
    reinforce_loss,
    returns_to_go,
)

_CFG = OUCoreConfig(t_max=20)


class _ScriptedRng:
    """Small deterministic RNG stand-in for exercising rejection sampling."""

    def __init__(self, values):
        self._values = iter(values)
        self.draw_count = 0

    def integers(self, high):
        value = next(self._values)
        assert 0 <= value < high
        self.draw_count += 1
        return value


def _hand_rolled_rtg(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards)
    for t in range(len(rewards)):
        for k in range(t, len(rewards)):
            out[t] += gamma ** (k - t) * rewards[k]
    return out


@pytest.mark.correctness
@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_returns_to_go_matches_definition(gamma):
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=12)
    np.testing.assert_allclose(
        returns_to_go(rewards, gamma), _hand_rolled_rtg(rewards, gamma),
        rtol=1e-12,
    )


@pytest.mark.correctness
def test_returns_to_go_batched_matches_per_episode():
    rng = np.random.default_rng(1)
    rewards = rng.normal(size=(15, 4))
    batched = returns_to_go(rewards, 0.95)
    for i in range(rewards.shape[1]):
        np.testing.assert_allclose(
            batched[:, i], returns_to_go(rewards[:, i], 0.95), rtol=1e-12
        )


@pytest.mark.correctness
def test_advantages_are_baselined_and_normalized():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(30, 8))
    adv = advantages(g)
    # Per-timestep batch mean removed exactly, global unit scale.
    np.testing.assert_allclose(adv.mean(axis=1), 0.0, atol=1e-12)
    assert adv.std() == pytest.approx(1.0, rel=1e-6)


@pytest.mark.correctness
def test_held_out_seed_intervals_are_half_open_and_disjoint():
    assert held_out_seed_intervals(10_000, 500, 100_000, 5_000) == (
        (10_000, 10_500),
        (100_000, 105_000),
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        held_out_seed_intervals(100, 50, 125, 50)


@pytest.mark.correctness
def test_training_reset_seed_rejects_reserved_draws_and_resamples():
    rng = _ScriptedRng([10_000, 104_999, 42])
    reserved = held_out_seed_intervals(10_000, 500, 100_000, 5_000)

    seed = draw_training_reset_seed(rng, reserved)

    assert seed == 42
    assert rng.draw_count == 3


@pytest.mark.correctness
def test_training_reset_seed_rejects_an_exhausted_seed_domain():
    rng = _ScriptedRng([])
    with pytest.raises(ValueError, match="leave no training seeds"):
        draw_training_reset_seed(rng, ((0, 2**31 - 1),))


@pytest.mark.correctness
def test_collect_batch_matches_env_ledger():
    """The batch collector must reproduce the environment ledger exactly.

    Per-episode reward sums are ln(W_T/W_0) under the canonical objective,
    and shapes and dtypes match the float32 observation contract.
    """
    net = PolicyNet(obs_dim=_CFG.obs_dim, n_actions=_CFG.n_actions)
    envs = [OUTradingEnv(config=_CFG) for _ in range(3)]
    rng = np.random.default_rng(3)

    obs, actions, rewards = collect_batch(net, envs, rng)
    assert obs.shape == (_CFG.t_max, 3, _CFG.obs_dim)
    assert obs.dtype == np.float32
    assert actions.shape == rewards.shape == (_CFG.t_max, 3)
    assert set(np.unique(actions)) <= set(range(_CFG.n_actions))
    # t/T feature advances one step per row for every episode.
    np.testing.assert_allclose(
        obs[:, :, 2], np.arange(_CFG.t_max)[:, None] / _CFG.t_max * np.ones(3),
        atol=1e-6,
    )
    # Replaying actions through a same-seeded env reproduces rewards.
    rng_replay = np.random.default_rng(3)
    seeds = [int(rng_replay.integers(2**31 - 1)) for _ in range(3)]
    for i, seed in enumerate(seeds):
        env = OUTradingEnv(config=_CFG)
        env.reset(seed=seed)
        total = 0.0
        for t in range(_CFG.t_max):
            _, r, term, trunc, _ = env.step(int(actions[t, i]))
            total += r
        assert term and not trunc
        assert total == pytest.approx(float(rewards[:, i].sum()), rel=1e-12)


@pytest.mark.correctness
def test_training_update_moves_parameters():
    torch.manual_seed(0)
    net = PolicyNet(obs_dim=_CFG.obs_dim, n_actions=_CFG.n_actions)
    envs = [OUTradingEnv(config=_CFG) for _ in range(4)]
    rng = np.random.default_rng(4)
    before = [p.detach().clone() for p in net.parameters()]

    obs, actions, rewards = collect_batch(net, envs, rng)
    adv = advantages(returns_to_go(rewards, _CFG.gamma))
    loss, entropy = reinforce_loss(
        net,
        torch.from_numpy(obs.reshape(-1, _CFG.obs_dim)),
        torch.from_numpy(actions.reshape(-1)),
        torch.from_numpy(adv.reshape(-1).astype(np.float32)),
        entropy_coef=0.01,
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert np.isfinite(float(loss.item()))
    assert 0.0 < entropy <= np.log(_CFG.n_actions) + 1e-6
    assert any(
        not torch.equal(before_param, param.detach())
        for before_param, param in zip(before, net.parameters())
    )


@pytest.mark.correctness
def test_greedy_policy_is_deterministic_and_valid():
    net = PolicyNet(obs_dim=_CFG.obs_dim, n_actions=_CFG.n_actions)
    pol = greedy_policy(net)
    obs = np.array([0.3, 0.5, 0.1], dtype=np.float32)
    a1, a2 = pol(obs, 0), pol(obs, 0)
    assert a1 == a2
    assert a1 in range(_CFG.n_actions)
