"""
MPC second-oracle test suite
============================
Markers:
  correctness  — MPC must be a sane independent controller.
  slow         — CRN cross-check against the DP optimum (builds both solvers).

Spec reference: docs/phase1_plan.md §3.4
Implementation: src/solvers/mpc.py
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.ou_core import CORE_BUY, CORE_SELL, OUCoreConfig, OUTradingEnv
from src.solvers import dp, mpc

_SEED = 42


def _crn_rollout(policy_fn, cfg, n=2500, seed0=_SEED):
    """Common-random-numbers rollout: episode ep always uses seed seed0+ep."""
    env = OUTradingEnv(config=cfg, seed=seed0)
    rets = np.empty(n)
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        total, t = 0.0, 0
        while True:
            a = policy_fn(obs, t)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            t += 1
            if term or trunc:
                break
        rets[ep] = total
    return rets


@pytest.mark.correctness
def test_mpc_policy_is_directional():
    """A sane controller buys when deeply underpriced and sells when overpriced."""
    cfg = OUCoreConfig()
    pol = mpc.build(cfg, mpc.MPCGrid(n_z=121, n_f=81, n_quad=12, horizon=25))
    assert pol.policy_at(-3.0, 0.0) == CORE_BUY
    assert pol.policy_at(3.0, 0.8) == CORE_SELL


@pytest.mark.slow
def test_mpc_matches_dp_value_under_crn():
    """The two oracles must agree within Monte-Carlo CI (phase1_plan §3.4 acceptance).
    MPC uses a fully independent backward induction, so agreement validates dp.py's
    sparse-operator precompute."""
    cfg = OUCoreConfig()
    res = dp.solve(cfg, dp.DPGrid())
    pol = mpc.build(cfg, mpc.MPCGrid())

    dp_ret = _crn_rollout(lambda o, t: res.policy_at(t, float(o[0]), float(o[1])), cfg)
    mpc_ret = _crn_rollout(lambda o, t: pol.policy_at(float(o[0]), float(o[1])), cfg)

    # Paired (CRN) difference has far lower variance than either mean's CI.
    diff = mpc_ret - dp_ret
    ci_diff = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    # Pre-registered tolerance: MPC's finite-lookahead/zero-terminal end-effect is a
    # small, documented suboptimality, not a fitted fudge.
    assert abs(diff.mean()) < ci_diff + 0.02, f"mpc-dp={diff.mean():.4f} ci={ci_diff:.4f}"
