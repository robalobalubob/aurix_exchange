"""
Clean OU trading-core test suite
================================
Markers:
  correctness  — the core math must match phase1_plan §2 exactly.
  regression   — locks in observed behaviour.

Run:  pytest src/tests/test_ou_core.py

Spec reference: docs/phase1_plan.md §2
Implementation: src/env/ou_core.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.env.ou_core import (
    ACTION_SET_TARGET,
    CORE_BUY,
    CORE_HOLD,
    CORE_SELL,
    OUCoreConfig,
    OUTradingEnv,
    apply_trade,
    delta_logprice,
    next_f,
    reward,
    step_z,
)

_SEED = 42


# ---------------------------------------------------------------------------
# Reference: explicit wealth-carrying ledger (the thing the (z, f) reduction
# claims to be equivalent to). Used to pin scale-invariance.
# ---------------------------------------------------------------------------
def _wealth_ledger_step(W, f, action, dX, cfg):
    """One step of the full absolute-wealth ledger. Returns (log_return, f_next)."""
    cash = (1.0 - f) * W
    asset = f * W
    phi = cfg.fee

    if action == CORE_BUY:
        notional = cfg.buy_fraction * cash
        cash -= notional
        asset += notional * (1.0 - phi)
    elif action == CORE_SELL:
        cash += asset * (1.0 - phi)
        asset = 0.0
    # HOLD: no trade

    asset *= math.exp(dX)
    w_next = cash + asset
    return math.log(w_next / W), asset / w_next


# ---------------------------------------------------------------------------
# AR(1) dynamics
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_z_step_is_exact_ar1():
    cfg = OUCoreConfig()
    rng = np.random.default_rng(_SEED)
    z, eps = 1.3, rng.standard_normal()
    expected = cfg.rho * z + math.sqrt(1.0 - cfg.rho**2) * eps
    assert step_z(z, eps, cfg) == pytest.approx(expected)


@pytest.mark.correctness
def test_ar1_is_unit_variance_stationary():
    """z ~ N(0,1) in, one step out, must remain unit variance (exact AR(1))."""
    cfg = OUCoreConfig()
    rng = np.random.default_rng(_SEED)
    z = rng.standard_normal(200_000)
    eps = rng.standard_normal(200_000)
    z_next = step_z(z, eps, cfg)
    assert np.var(z_next) == pytest.approx(1.0, abs=0.01)
    assert np.mean(z_next) == pytest.approx(0.0, abs=0.01)


@pytest.mark.correctness
def test_delta_logprice_is_sigma_stat_times_dz():
    """dX must equal sigma_stat*(z' - z) — the single-source-of-truth shock."""
    cfg = OUCoreConfig()
    rng = np.random.default_rng(_SEED)
    z = rng.standard_normal(1000)
    eps = rng.standard_normal(1000)
    dX = delta_logprice(z, eps, cfg)
    expected = cfg.sigma_stat * (step_z(z, eps, cfg) - z)
    np.testing.assert_allclose(dX, expected)


@pytest.mark.correctness
def test_core_coefficients_match_game_env():
    """The clean core must reproduce the game env's Byrinium GOU coefficients, or
    the two simulations have silently drifted (single-source rule)."""
    from src.env.aurix_env import AurixExchangeEnv, IDX_B

    env = AurixExchangeEnv()
    cfg = OUCoreConfig()  # canonical = Byrinium
    assert cfg.rho == pytest.approx(env._gou_decay[IDX_B])
    assert cfg.sigma_stat == pytest.approx(env._sigma_stat[IDX_B])
    assert cfg.sigma_stat * cfg.noise_scale == pytest.approx(env._gou_noise_std[IDX_B])
    assert cfg.mu == pytest.approx(env.cfg.gou_mu[IDX_B])


# ---------------------------------------------------------------------------
# Trade map (ledger arithmetic)
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_hold_is_a_no_op():
    cfg = OUCoreConfig()
    f_prime, cost = apply_trade(0.4, CORE_HOLD, cfg)
    assert float(f_prime) == pytest.approx(0.4)
    assert float(cost) == pytest.approx(0.0)


@pytest.mark.correctness
@pytest.mark.parametrize("f", [0.0, 0.25, 0.5, 0.9])
def test_buy_ledger(f):
    cfg = OUCoreConfig()
    n = cfg.buy_fraction * (1.0 - f)
    f_prime, cost = apply_trade(f, CORE_BUY, cfg)
    assert float(cost) == pytest.approx(cfg.fee * n)
    assert float(f_prime) == pytest.approx((f + n * (1.0 - cfg.fee)) / (1.0 - cfg.fee * n))
    assert float(f_prime) > f  # buying raises the asset fraction


@pytest.mark.correctness
@pytest.mark.parametrize("f", [0.1, 0.5, 1.0])
def test_sell_liquidates_fully(f):
    cfg = OUCoreConfig()
    f_prime, cost = apply_trade(f, CORE_SELL, cfg)
    assert float(f_prime) == pytest.approx(0.0)
    assert float(cost) == pytest.approx(cfg.fee * f)


@pytest.mark.correctness
def test_target_set_moves_to_fraction_with_distance_cost():
    cfg = OUCoreConfig(action_set=ACTION_SET_TARGET)
    # target_fractions = (0, .25, .5, .75, 1); action 3 -> 0.75
    f_prime, cost = apply_trade(0.25, 3, cfg)
    assert float(f_prime) == pytest.approx(0.75)
    assert float(cost) == pytest.approx(cfg.fee * abs(0.75 - 0.25))


# ---------------------------------------------------------------------------
# Reward & next-fraction
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_reward_formula():
    cfg = OUCoreConfig()
    f_prime, cost, dX = 0.6, 0.003, 0.05
    expected = math.log(1.0 - cost) + math.log((1.0 - f_prime) + f_prime * math.exp(dX))
    assert float(reward(f_prime, cost, dX, cfg)) == pytest.approx(expected)


@pytest.mark.correctness
def test_reward_clipping_toggle():
    clipped = OUCoreConfig(reward_clip=True, reward_clip_value=0.01)
    # A large favourable move would give r > 0.01; clipping must bind.
    r = float(reward(1.0, 0.0, 2.0, clipped))
    assert r == pytest.approx(0.01)


@pytest.mark.correctness
def test_all_cash_has_zero_holding_return():
    """f'=0 -> holding term is ln(1) = 0 regardless of dX."""
    cfg = OUCoreConfig()
    assert float(reward(0.0, 0.0, 0.5, cfg)) == pytest.approx(0.0)
    assert float(next_f(0.0, 0.5)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scale-invariance: the load-bearing claim of the (z, f) reduction.
# The pure core must reproduce the full wealth-carrying ledger at ANY wealth.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.parametrize("action", [CORE_HOLD, CORE_BUY, CORE_SELL])
@pytest.mark.parametrize("wealth", [1.0, 137.0, 1.0e6, 5.5e-3])
def test_scale_invariance_against_wealth_ledger(action, wealth):
    cfg = OUCoreConfig()
    rng = np.random.default_rng(_SEED)
    for _ in range(50):
        f = float(rng.uniform(0.0, 1.0))
        dX = float(rng.normal(0.0, cfg.sigma_stat))

        f_prime, cost = apply_trade(f, action, cfg)
        core_r = float(reward(f_prime, cost, dX, cfg))
        core_fnext = float(next_f(f_prime, dX))

        ref_r, ref_fnext = _wealth_ledger_step(wealth, f, action, dX, cfg)
        assert core_r == pytest.approx(ref_r, abs=1e-9)
        assert core_fnext == pytest.approx(ref_fnext, abs=1e-9)


# ---------------------------------------------------------------------------
# Gymnasium adapter
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_env_obs_shape_and_dtype():
    env = OUTradingEnv(seed=_SEED)
    obs, info = env.reset()
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert info["action_mask"].all()
    # f starts at 0, time feature starts at 0.
    assert obs[1] == pytest.approx(0.0)
    assert obs[2] == pytest.approx(0.0)


@pytest.mark.correctness
def test_env_no_time_feature_obs_dim():
    env = OUTradingEnv(config=OUCoreConfig(time_feature=False), seed=_SEED)
    obs, _ = env.reset()
    assert obs.shape == (2,)


@pytest.mark.correctness
def test_env_terminates_at_horizon():
    """Reaching t_max is terminated=True (D2), never truncated."""
    cfg = OUCoreConfig(t_max=5)
    env = OUTradingEnv(config=cfg, seed=_SEED)
    env.reset()
    terms = []
    for _ in range(cfg.t_max):
        _, _, terminated, truncated, _ = env.step(CORE_HOLD)
        terms.append((terminated, truncated))
    assert terms[-1] == (True, False)
    assert all(t == (False, False) for t in terms[:-1])


@pytest.mark.correctness
def test_env_reset_seeding_is_reproducible():
    a = OUTradingEnv(seed=7)
    b = OUTradingEnv(seed=7)
    obs_a, _ = a.reset()
    obs_b, _ = b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    for _ in range(20):
        sa = a.step(CORE_BUY)
        sb = b.step(CORE_BUY)
        np.testing.assert_array_equal(sa[0], sb[0])
        assert sa[1] == sb[1]
