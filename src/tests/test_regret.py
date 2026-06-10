"""
Regret harness test suite
=========================
Markers:
  correctness  — the harness must satisfy its defining properties (phase1_plan §5).
  slow         — full-resolution DP solve + many rollouts.

Spec reference: docs/phase1_plan.md §5
Implementation: src/solvers/regret.py
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.ou_core import OUCoreConfig
from src.solvers import dp, regret

_SEED0 = 10_000


@pytest.fixture(scope="module")
def res() -> dp.DPResult:
    # Short horizon keeps the harness tests quick while preserving all properties.
    return dp.solve(OUCoreConfig(t_max=40), dp.DPGrid(n_z=121, n_f=81))


@pytest.mark.correctness
def test_crn_rollout_is_deterministic(res):
    cfg = res.cfg
    seeds = regret.make_seed_block(50, _SEED0)
    pol = regret.from_dp(res)
    r1, a1 = regret.rollout(pol, cfg, seeds)
    r2, a2 = regret.rollout(pol, cfg, seeds)
    np.testing.assert_array_equal(r1, r2)
    # No reference policy -> agreement is NaN (and NaN != NaN, so check explicitly).
    assert np.isnan(a1) and np.isnan(a2)


@pytest.mark.correctness
def test_pistar_self_agreement_is_total(res):
    """pi* scored against itself agrees on every decision."""
    seeds = regret.make_seed_block(30, _SEED0)
    pol = regret.from_dp(res)
    _, agreement = regret.rollout(pol, res.cfg, seeds, reference=pol)
    assert agreement == pytest.approx(1.0)


@pytest.mark.correctness
def test_pistar_regret_near_zero(res):
    rep = regret.evaluate_regret(regret.from_dp(res), res, n_episodes=1500, seed0=_SEED0)
    # pi* must achieve V* up to MC + grid/interpolation slack.
    assert abs(rep.regret) < rep.return_ci + 0.05
    # Paired against itself the difference is exactly zero.
    assert rep.paired_regret == pytest.approx(0.0, abs=1e-12)
    assert rep.agreement_rate == pytest.approx(1.0)


@pytest.mark.correctness
def test_random_policy_has_positive_regret(res):
    rep = regret.evaluate_regret(
        regret.random_policy(res.cfg.n_actions, seed=1), res,
        n_episodes=1500, seed0=_SEED0,
    )
    # A random policy must be clearly worse than the optimum.
    assert rep.regret > 0.0
    assert rep.paired_regret > rep.paired_ci  # significantly positive
    assert rep.agreement_rate < 0.9


@pytest.mark.correctness
def test_crn_reduces_variance_for_correlated_policy(res):
    """CRN pairing slashes the variance of the regret *difference* when the candidate
    is correlated with pi* (the case that matters — scoring a near-optimal agent). An
    eps-greedy-pi* policy shares most decisions with pi*, so its paired CI is far
    tighter than its unpaired return CI."""
    rng = np.random.default_rng(3)
    base = regret.from_dp(res)
    n_a = res.cfg.n_actions

    def eps_greedy(obs, t):
        if rng.random() < 0.1:
            return int(rng.integers(n_a))
        return base(obs, t)

    rep = regret.evaluate_regret(eps_greedy, res, n_episodes=1500, seed0=_SEED0)
    assert rep.paired_ci < rep.return_ci
    assert rep.paired_regret > 0.0  # eps-greedy is strictly worse than pi*


@pytest.mark.slow
def test_canonical_regret_report_renders(res):
    rep = regret.evaluate_regret(regret.from_dp(res), res, n_episodes=500, seed0=_SEED0)
    text = rep.render()
    assert "regret (vs V*)" in text
    assert "decision agreement" in text
