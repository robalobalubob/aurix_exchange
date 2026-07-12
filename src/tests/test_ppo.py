"""
PPO trainer test suite (Phase 2, third policy-gradient rung)
============================================================
Markers:
  correctness — the clipped surrogate must satisfy its defining identities:
                ratio=1 recovers the plain policy gradient with zero clipping,
                and clipping is uniformly pessimistic.

Spec reference: docs/aurix_rl_roadmap_v2.md Phase 2
Implementation: src/training/train_ppo.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from src.env.ou_core import OUCoreConfig, OUTradingEnv
from src.models.policy import ActorCriticNet
from src.training.train_a2c import gae, normalize
from src.training.train_ppo import ppo_loss
from src.training.train_reinforce import collect_batch

_CFG = OUCoreConfig(t_max=20)


def _fresh_batch(seed: int, n_envs: int = 4):
    torch.manual_seed(seed)
    net = ActorCriticNet(obs_dim=_CFG.obs_dim, n_actions=_CFG.n_actions)
    envs = [OUTradingEnv(config=_CFG) for _ in range(n_envs)]
    obs, actions, rewards = collect_batch(net, envs, np.random.default_rng(seed))
    obs_t = torch.from_numpy(obs.reshape(-1, _CFG.obs_dim))
    act_t = torch.from_numpy(actions.reshape(-1))
    with torch.no_grad():
        logits, values_t = net(obs_t)
        old_log_prob = Categorical(logits=logits).log_prob(act_t)
    values = values_t.numpy().astype(np.float64).reshape(rewards.shape)
    adv, targets = gae(rewards, values, _CFG.gamma, 0.95)
    adv_t = torch.from_numpy(normalize(adv).reshape(-1).astype(np.float32))
    target_t = torch.from_numpy(targets.reshape(-1).astype(np.float32))
    return net, obs_t, act_t, old_log_prob, adv_t, target_t


@pytest.mark.correctness
def test_ratio_one_recovers_plain_policy_gradient():
    """Before any update the ratio is exactly 1, so with the value and entropy
    terms zeroed the surrogate loss equals -mean(A) and nothing clips."""
    net, obs, act, old_lp, adv, target = _fresh_batch(0)
    loss, entropy, clip_fraction = ppo_loss(
        net, obs, act, old_lp, adv, target,
        clip_eps=0.2, value_coef=0.0, entropy_coef=0.0,
    )
    assert float(loss.item()) == pytest.approx(float(-adv.mean().item()), abs=1e-6)
    assert clip_fraction == 0.0
    assert 0.0 < entropy <= np.log(_CFG.n_actions) + 1e-6


@pytest.mark.correctness
def test_clipping_is_pessimistic():
    """Against a perturbed behaviour policy the clipped surrogate is bounded
    below by the unclipped one (min(rA, clip(r)A) <= rA elementwise, so the
    negated loss is >=)."""
    net, obs, act, old_lp, adv, target = _fresh_batch(1)
    # Perturb the stored behaviour log-probs so ratios leave [1-eps, 1+eps].
    old_lp = old_lp + 0.5 * torch.randn_like(old_lp)

    clipped_loss, _, clip_fraction = ppo_loss(
        net, obs, act, old_lp, adv, target,
        clip_eps=0.2, value_coef=0.0, entropy_coef=0.0,
    )
    unclipped_loss, _, _ = ppo_loss(
        net, obs, act, old_lp, adv, target,
        clip_eps=1e9, value_coef=0.0, entropy_coef=0.0,
    )
    assert clip_fraction > 0.0
    assert float(clipped_loss.item()) >= float(unclipped_loss.item()) - 1e-8


@pytest.mark.correctness
def test_ppo_update_moves_parameters_and_is_finite():
    net, obs, act, old_lp, adv, target = _fresh_batch(2)
    before = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3):  # several epochs over the same batch (the PPO point)
        loss, _, _ = ppo_loss(
            net, obs, act, old_lp, adv, target,
            clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert np.isfinite(float(loss.item()))
    assert any(
        not torch.equal(b, p.detach()) for b, p in zip(before, net.parameters())
    )
