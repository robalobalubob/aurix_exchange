"""
A2C trainer test suite (Phase 2, second policy-gradient rung)
=============================================================
Markers:
  correctness — GAE must match its defining recursion; the lambda=1 limit must
                recover Monte-Carlo advantages; the update must be well-formed.

Spec reference: docs/aurix_rl_roadmap_v2.md Phase 2
Implementation: src/training/train_a2c.py, src/models/policy.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.ou_core import OUCoreConfig, OUTradingEnv
from src.models.policy import ActorCriticNet
from src.training.train_a2c import a2c_loss, gae, normalize
from src.training.train_reinforce import collect_batch, greedy_policy, returns_to_go

_CFG = OUCoreConfig(t_max=20)


def _hand_rolled_gae(rewards, values, gamma, lam):
    """A_t = sum_k (gamma*lam)^k delta_{t+k}, with V(s_T) = 0."""
    t_max = len(rewards)
    v_next = np.append(values[1:], 0.0)
    delta = rewards + gamma * v_next - values
    adv = np.zeros(t_max)
    for t in range(t_max):
        for k in range(t_max - t):
            adv[t] += (gamma * lam) ** k * delta[t + k]
    return adv


@pytest.mark.correctness
@pytest.mark.parametrize("gamma,lam", [(1.0, 0.95), (0.99, 0.9), (1.0, 0.0)])
def test_gae_matches_definition(gamma, lam):
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(12, 3))
    values = rng.normal(size=(12, 3))
    adv, targets = gae(rewards, values, gamma, lam)
    for i in range(3):
        np.testing.assert_allclose(
            adv[:, i],
            _hand_rolled_gae(rewards[:, i], values[:, i], gamma, lam),
            rtol=1e-10, atol=1e-12,
        )
    np.testing.assert_allclose(targets, adv + values, rtol=1e-12)


@pytest.mark.correctness
def test_gae_lambda_one_recovers_monte_carlo_advantage():
    """At lambda=1 the GAE telescopes to G_t - V(s_t) regardless of the critic."""
    rng = np.random.default_rng(1)
    rewards = rng.normal(size=(15, 4))
    values = rng.normal(size=(15, 4))
    adv, _ = gae(rewards, values, 1.0, 1.0)
    np.testing.assert_allclose(
        adv, returns_to_go(rewards, 1.0) - values, rtol=1e-10, atol=1e-12
    )


@pytest.mark.correctness
def test_gae_lambda_zero_is_one_step_td():
    rng = np.random.default_rng(2)
    rewards = rng.normal(size=(10, 2))
    values = rng.normal(size=(10, 2))
    adv, _ = gae(rewards, values, 1.0, 0.0)
    v_next = np.vstack([values[1:], np.zeros((1, 2))])
    np.testing.assert_allclose(
        adv, rewards + v_next - values, rtol=1e-12, atol=1e-12
    )


@pytest.mark.correctness
def test_normalize_is_zero_mean_unit_scale():
    rng = np.random.default_rng(3)
    x = normalize(rng.normal(loc=5.0, scale=3.0, size=(30, 8)))
    assert x.mean() == pytest.approx(0.0, abs=1e-12)
    assert x.std() == pytest.approx(1.0, rel=1e-6)


@pytest.mark.correctness
def test_actor_critic_shapes_and_logits_consistency():
    net = ActorCriticNet(obs_dim=_CFG.obs_dim, n_actions=_CFG.n_actions)
    obs = torch.randn(7, _CFG.obs_dim)
    logits, value = net(obs)
    assert logits.shape == (7, _CFG.n_actions)
    assert value.shape == (7,)
    # action_logits (the collector/eval path) must agree with forward.
    torch.testing.assert_close(net.action_logits(obs), logits)


@pytest.mark.correctness
def test_a2c_update_moves_both_heads():
    torch.manual_seed(0)
    net = ActorCriticNet(obs_dim=_CFG.obs_dim, n_actions=_CFG.n_actions)
    envs = [OUTradingEnv(config=_CFG) for _ in range(4)]
    rng = np.random.default_rng(4)

    obs, actions, rewards = collect_batch(net, envs, rng)
    obs_t = torch.from_numpy(obs.reshape(-1, _CFG.obs_dim))
    act_t = torch.from_numpy(actions.reshape(-1))
    with torch.no_grad():
        _, values_t = net(obs_t)
    values = values_t.numpy().astype(np.float64).reshape(rewards.shape)
    adv, targets = gae(rewards, values, _CFG.gamma, 0.95)

    policy_before = net.policy_head.weight.detach().clone()
    value_before = net.value_head.weight.detach().clone()
    loss, entropy, value_loss = a2c_loss(
        net, obs_t, act_t,
        torch.from_numpy(normalize(adv).reshape(-1).astype(np.float32)),
        torch.from_numpy(targets.reshape(-1).astype(np.float32)),
        value_coef=0.5, entropy_coef=0.01,
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert np.isfinite(float(loss.item()))
    assert value_loss >= 0.0
    assert 0.0 < entropy <= np.log(_CFG.n_actions) + 1e-6
    assert not torch.equal(policy_before, net.policy_head.weight.detach())
    assert not torch.equal(value_before, net.value_head.weight.detach())


@pytest.mark.correctness
def test_greedy_policy_works_on_actor_critic():
    net = ActorCriticNet(obs_dim=_CFG.obs_dim, n_actions=_CFG.n_actions)
    pol = greedy_policy(net)
    obs = np.array([-0.7, 0.2, 0.4], dtype=np.float32)
    assert pol(obs, 0) == pol(obs, 0)
    assert pol(obs, 0) in range(_CFG.n_actions)
