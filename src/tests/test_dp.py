"""
DP ground-truth test suite
==========================
Markers:
  correctness  — the solver must satisfy the validation checklist (phase1_plan §4).
  regression   — locks in the blessed V (updated only deliberately).
  slow         — Monte-Carlo self-consistency; the single strongest check.

Run fast checks:  pytest src/tests/test_dp.py -m "not slow"
Run everything:   pytest src/tests/test_dp.py

Spec reference: docs/phase1_plan.md §3, §4
Implementation: src/solvers/dp.py
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.ou_core import (
    ACTION_SET_TARGET,
    CORE_BUY,
    CORE_SELL,
    OUCoreConfig,
    OUTradingEnv,
)
from src.solvers.dp import DPGrid, DPResult, config_hash, one_step_q, solve

_SEED = 42


# A coarse, short-horizon solve keeps the fast checks quick. Module-scoped so the
# several correctness tests share one solve.
@pytest.fixture(scope="module")
def coarse() -> DPResult:
    cfg = OUCoreConfig(t_max=40)
    grid = DPGrid(n_z=121, n_f=81, n_quad=16)
    return solve(cfg, grid)


# ---------------------------------------------------------------------------
# §4.1 Quadrature convergence
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_quadrature_convergence():
    cfg = OUCoreConfig(t_max=40)
    g16 = solve(cfg, DPGrid(n_z=121, n_f=81, n_quad=16))
    g32 = solve(cfg, DPGrid(n_z=121, n_f=81, n_quad=32))
    # Same grid, doubled nodes: the reported scalar is converged to ~1e-4; the
    # pointwise value barely moves (a few 1e-4/step accumulated over the horizon).
    assert abs(g16.expected_initial_value() - g32.expected_initial_value()) < 5e-4
    assert np.max(np.abs(g16.V[0] - g32.V[0])) < 5e-3


# ---------------------------------------------------------------------------
# §4.2 Grid convergence (value and policy boundaries)
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_grid_convergence():
    cfg = OUCoreConfig(t_max=40)
    coarse = solve(cfg, DPGrid(n_z=121, n_f=81, n_quad=16))
    fine = solve(cfg, DPGrid(n_z=241, n_f=161, n_quad=16))
    # Compare E_z[V0] (a grid-independent scalar) across resolutions.
    assert abs(coarse.expected_initial_value() - fine.expected_initial_value()) < 5e-3


# ---------------------------------------------------------------------------
# §4 (internal) Bellman consistency on-grid: V == max_a Q, pi == argmax_a Q.
# Exact by construction; catches operator/reward wiring bugs.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_bellman_consistency_offgrid(coarse):
    rng = np.random.default_rng(_SEED)
    for _ in range(60):
        t = int(rng.integers(0, coarse.cfg.t_max - 1))
        z = float(rng.uniform(-3.0, 3.0))
        f = float(rng.uniform(0.0, 1.0))
        q = one_step_q(coarse, t, z, f)
        best = coarse.policy_at(t, z, f)
        # §4.4 no profitable deviation: the policy's action is within interpolation
        # tolerance of the best Q at this state.
        assert q[best] >= q.max() - 5e-3


# ---------------------------------------------------------------------------
# §4.3 Monte-Carlo self-consistency — the big one.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_monte_carlo_self_consistency():
    cfg = OUCoreConfig(t_max=200)
    res = solve(cfg, DPGrid())
    v0 = res.expected_initial_value()

    n_ep = 4000
    env = OUTradingEnv(config=cfg, seed=_SEED)
    returns = np.empty(n_ep)
    for ep in range(n_ep):
        obs, _ = env.reset(seed=_SEED + ep)
        total = 0.0
        t = 0
        while True:
            a = res.policy_at(t, float(obs[0]), float(obs[1]))
            obs, r, term, trunc, _ = env.step(a)
            total += r
            t += 1
            if term or trunc:
                break
        returns[ep] = total

    mean = returns.mean()
    ci = 1.96 * returns.std(ddof=1) / np.sqrt(n_ep)
    # Rolling pi* must realise the predicted value within MC CI (+ a small grid/
    # interpolation slack). This single test catches transition/reward mismatch.
    assert abs(mean - v0) < ci + 0.05, f"mean={mean:.4f} v0={v0:.4f} ci={ci:.4f}"


# ---------------------------------------------------------------------------
# §4.5 Limit behaviour: fee -> 0 collapses the no-trade band to the myopic sign
# rule. E[dX | z] = sigma_stat*(rho-1)*z, so z<0 => price expected to rise (be in),
# z>0 => expected to fall (be out).
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_zero_fee_collapses_to_myopic_sign_rule():
    cfg = OUCoreConfig(t_max=40, fee=1e-9)
    res = solve(cfg, DPGrid(n_z=121, n_f=81))
    t = 0  # deep in the horizon
    # Strongly underpriced -> want to be invested (BUY from cash).
    assert res.policy_at(t, -3.0, 0.0) == CORE_BUY
    # Strongly overpriced while holding -> liquidate.
    assert res.policy_at(t, 3.0, 0.8) == CORE_SELL


@pytest.mark.correctness
def test_no_trade_band_widens_with_fee():
    """Higher proportional fees should make the agent trade less often (Davis-Norman
    no-trade band widens)."""
    grid = DPGrid(n_z=121, n_f=81)
    lo = solve(OUCoreConfig(t_max=40, fee=0.005), grid)
    hi = solve(OUCoreConfig(t_max=40, fee=0.05), grid)
    hold_lo = np.mean(lo.pi[0] == 0)
    hold_hi = np.mean(hi.pi[0] == 0)
    assert hold_hi > hold_lo


# ---------------------------------------------------------------------------
# §4.6 Qualitative structure: the asymmetric set's ratchet (climb in f via BUY,
# instant drop to 0 via SELL).
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_ratchet_structure(coarse):
    t = 0
    # Deeply underpriced: accumulate (BUY) across a range of current fractions.
    for f in (0.0, 0.2, 0.4):
        assert coarse.policy_at(t, -3.0, f) == CORE_BUY
    # Deeply overpriced while holding: dump everything at once (SELL), not a trickle.
    for f in (0.3, 0.6, 0.9):
        assert coarse.policy_at(t, 3.0, f) == CORE_SELL


# ---------------------------------------------------------------------------
# Twins / secondary action set solve cleanly (§3.3).
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_clipped_twin_lowers_value():
    """The clipped objective caps favourable tail returns, so its value (scored on
    its own clipped reward) cannot exceed the unclipped optimum's."""
    grid = DPGrid(n_z=121, n_f=81)
    unclipped = solve(OUCoreConfig(t_max=40), grid)
    clipped = solve(OUCoreConfig(t_max=40, reward_clip=True, reward_clip_value=0.1), grid)
    assert clipped.expected_initial_value() < unclipped.expected_initial_value()


@pytest.mark.correctness
def test_target_action_set_solves():
    cfg = OUCoreConfig(t_max=30, action_set=ACTION_SET_TARGET)
    res = solve(cfg, DPGrid(n_z=81, n_f=81))
    assert res.pi.max() <= cfg.n_actions - 1
    assert res.V[0].max() > res.V[0].min()


# ---------------------------------------------------------------------------
# Persistence + config hash.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_save_load_roundtrip(tmp_path, coarse):
    path = str(tmp_path / "dp.npz")
    coarse.save(path)
    loaded = DPResult.load(path)
    np.testing.assert_array_equal(coarse.V, loaded.V)
    np.testing.assert_array_equal(coarse.pi, loaded.pi)
    assert loaded.config_hash == coarse.config_hash


@pytest.mark.correctness
def test_config_hash_distinguishes_configs():
    g = DPGrid()
    assert config_hash(OUCoreConfig(), g) != config_hash(OUCoreConfig(fee=0.02), g)
    assert config_hash(OUCoreConfig(), g) != config_hash(OUCoreConfig(), DPGrid(n_z=99))


# ---------------------------------------------------------------------------
# Regression: the blessed canonical optimum. Update only deliberately.
# ---------------------------------------------------------------------------
@pytest.mark.regression
def test_blessed_canonical_value():
    res = solve(OUCoreConfig(), DPGrid())
    # Blessed 2026-06: canonical Byrinium, n_z=201/n_f=101/n_quad=16.
    assert res.expected_initial_value() == pytest.approx(6.6385, abs=1e-2)
