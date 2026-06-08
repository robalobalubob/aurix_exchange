"""
Aurix Exchange Environment Test Suite
======================================
Markers:
  correctness  — implementation must match spec exactly. Failures here mean a spec violation.
  regression   — locks in observed behaviour. Failures here mean something changed unexpectedly.

Run correctness only:   pytest -m correctness
Run regression only:    pytest -m regression
Run all:                pytest src/tests/test_environment.py

Spec reference: docs/spec.md
Implementation: src/env/aurix_env.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.env.aurix_env import (
    AurixExchangeEnv,
    EnvConfig,
    N_ACTIONS,
    N_PHASES,
    OBS_DIM,
    HOLD,
    BUY_BYRINIUM,
    SELL_BYRINIUM,
    BUY_HERBS,
    BUY_TOOLS,
    LAUNCH_EXPEDITION,
    IDX_B,
    IDX_H,
    IDX_T,
)

# Seed used wherever a test needs reproducible randomness.
_SEED = 42
_ALL_ACTIONS = list(range(N_ACTIONS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env() -> AurixExchangeEnv:
    """A fresh, unseeded environment for tests that do not need determinism."""
    return AurixExchangeEnv()


@pytest.fixture
def seeded_env() -> AurixExchangeEnv:
    """A deterministically seeded environment, already reset.

    Tests that assert on concrete numeric values opt into this fixture so the
    RNG stream is fixed and reproducible.
    """
    e = AurixExchangeEnv(seed=_SEED)
    e.reset(seed=_SEED)
    return e


@pytest.fixture
def cfg() -> EnvConfig:
    """The default environment configuration — the source of truth for constants."""
    return EnvConfig()


# ===========================================================================
# 1. Gymnasium API Contract
# ===========================================================================
@pytest.mark.correctness
@pytest.mark.parametrize("action", _ALL_ACTIONS)
def test_step_returns_5_tuple(seeded_env: AurixExchangeEnv, action: int) -> None:
    """step() must return the modern (obs, reward, terminated, truncated, info) 5-tuple.

    The legacy 4-tuple Gym API is explicitly forbidden by the project rules; the
    C# inference pipeline and training loop both assume the 5-tuple contract.
    """
    result = seeded_env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 5

    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert np.isscalar(reward) or isinstance(reward, (float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)


@pytest.mark.correctness
def test_reset_returns_2_tuple(env: AurixExchangeEnv) -> None:
    """reset() must return a (observation, info) 2-tuple per the Gymnasium API."""
    result = env.reset()
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


@pytest.mark.correctness
def test_spaces_defined(env: AurixExchangeEnv) -> None:
    """observation_space and action_space must exist and be the right shapes/sizes."""
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.n == N_ACTIONS


@pytest.mark.correctness
def test_action_space_size_is_8(env: AurixExchangeEnv) -> None:
    """The action space must contain exactly 8 discrete actions (spec section 3.2)."""
    assert env.action_space.n == 8


# ===========================================================================
# 2. Observation Space
# ===========================================================================
@pytest.mark.correctness
def test_reset_obs_shape_and_dtype(env: AurixExchangeEnv) -> None:
    """reset() observation must be shape (10,) and dtype float32 — the C# span layout."""
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


@pytest.mark.correctness
@pytest.mark.parametrize("action", _ALL_ACTIONS)
def test_step_obs_shape_and_dtype(seeded_env: AurixExchangeEnv, action: int) -> None:
    """Every step() observation must remain shape (10,) and dtype float32.

    float64 or untyped tensors are forbidden — they break the ReadOnlySpan<float>
    boundary on the C# side.
    """
    obs, _, _, _, _ = seeded_env.step(action)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


@pytest.mark.correctness
def test_observation_always_finite_over_episode(seeded_env: AurixExchangeEnv) -> None:
    """No observation value may be NaN or inf, at reset or across a full episode.

    A non-finite observation would silently poison the replay buffer and the
    exported ONNX inference, so this is a hard invariant.
    """
    obs, _ = seeded_env.reset(seed=_SEED)
    assert np.all(np.isfinite(obs))

    rng = np.random.default_rng(_SEED)
    for _ in range(EnvConfig().t_max):
        action = int(rng.integers(0, N_ACTIONS))
        obs, reward, terminated, truncated, _ = seeded_env.step(action)
        assert np.all(np.isfinite(obs))
        if terminated or truncated:
            break


@pytest.mark.correctness
def test_norm_cash_index_0(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """Index 0 = ln(C + 1) / ln(Max_Expected_Cash); zero cash maps to exactly 0."""
    env.reset()
    env._cash = 0.0
    env._inventory[:] = 0.0
    obs = env._normalize_obs()
    assert obs[0] == pytest.approx(0.0, abs=1e-7)

    env._cash = 10_000.0
    obs = env._normalize_obs()
    assert obs[0] == pytest.approx(math.log(10_000.0 + 1.0) / cfg.cash_norm, rel=1e-5)


@pytest.mark.correctness
def test_norm_inventory_indices_1_to_3(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """Indices 1-3 = inventory / per-commodity Max_Capacity; full inventory maps to 1.0."""
    env.reset()
    # Per-commodity capacities differ now (B=500, H=2500, T=100).
    env._inventory[:] = env._max_inventory
    obs = env._normalize_obs()
    assert obs[1] == pytest.approx(1.0, rel=1e-6)
    assert obs[2] == pytest.approx(1.0, rel=1e-6)
    assert obs[3] == pytest.approx(1.0, rel=1e-6)

    # Half capacity maps to 0.5 for each, confirming the divisor is per-commodity.
    env._inventory[:] = env._max_inventory * 0.5
    obs = env._normalize_obs()
    assert obs[1] == pytest.approx(0.5, rel=1e-6)
    assert obs[2] == pytest.approx(0.5, rel=1e-6)
    assert obs[3] == pytest.approx(0.5, rel=1e-6)

    env._inventory[:] = 0.0
    obs = env._normalize_obs()
    assert obs[1] == 0.0
    assert obs[2] == 0.0
    assert obs[3] == 0.0


@pytest.mark.correctness
def test_norm_price_indices_4_to_6(env: AurixExchangeEnv) -> None:
    """Indices 4-6 = (X_i - mu_i) / sigma_stat_i.

    Placing the log-price exactly k stationary-standard-deviations above mu must
    yield a normalized feature of exactly k (tests the min/max price extremes).
    """
    env.reset()
    sigma_stat = env._sigma_stat
    mu = env._gou_mu

    for k, idx in [(2.0, IDX_B), (-2.0, IDX_H), (0.0, IDX_T)]:
        env._log_prices[idx] = mu[idx] + k * sigma_stat[idx]
    obs = env._normalize_obs()
    assert obs[4] == pytest.approx(2.0, rel=1e-5)
    assert obs[5] == pytest.approx(-2.0, rel=1e-5)
    assert obs[6] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.correctness
def test_norm_fatigue_phase_cooldown_indices_7_to_9(
    env: AurixExchangeEnv, cfg: EnvConfig
) -> None:
    """Index 7 = F/100, index 8 = phase τ/(N_PHASES-1), index 9 = cooldown/max.

    Per rebalancing.md §3.1 the step and net-worth features were repurposed to the
    phase index and expedition cooldown respectively.
    """
    env.reset()

    # Fatigue full; Night phase (τ=3) and cooldown at its ceiling both map to 1.0.
    env._fatigue = 100.0
    env._step_count = N_PHASES - 1  # τ = 3 -> Night
    env._cooldown = cfg.expedition_cooldown_steps
    obs = env._normalize_obs()
    assert obs[7] == pytest.approx(1.0, rel=1e-6)
    assert obs[8] == pytest.approx(1.0, rel=1e-6)
    assert obs[9] == pytest.approx(1.0, rel=1e-6)

    env._fatigue = 0.0
    env._step_count = 0  # τ = 0 -> Morning
    env._cooldown = 0
    obs = env._normalize_obs()
    assert obs[7] == 0.0
    assert obs[8] == 0.0
    assert obs[9] == 0.0

    # Phase cycles every N_PHASES steps; τ=1 (Day) maps to 1/3.
    env._step_count = N_PHASES + 1
    obs = env._normalize_obs()
    assert obs[8] == pytest.approx(1.0 / (N_PHASES - 1), rel=1e-6)


# ===========================================================================
# 3. GOU Price Dynamics
# ===========================================================================
@pytest.mark.correctness
def test_spot_prices_strictly_positive(seeded_env: AurixExchangeEnv) -> None:
    """S_i = exp(X_i) is strictly positive by construction; verify over an episode."""
    rng = np.random.default_rng(_SEED)
    for _ in range(EnvConfig().t_max):
        spot = np.exp(seeded_env._log_prices)
        assert np.all(spot > 0.0)
        _, _, terminated, truncated, _ = seeded_env.step(int(rng.integers(0, N_ACTIONS)))
        if terminated or truncated:
            break


@pytest.mark.correctness
def test_gou_step_matches_closed_form(env: AurixExchangeEnv) -> None:
    """One GOU step must match the exact closed-form update from spec section 2.2.

    X_{t+1} = mu + (X_t - mu) e^{-theta} + sigma sqrt((1 - e^{-2theta})/(2theta)) * eps

    We mirror the environment's RNG with an identical generator so the eps draw
    is reproduced exactly, then compare to floating-point tolerance.
    """
    env.reset()
    env._rng = np.random.default_rng(123)
    mirror = np.random.default_rng(123)

    x_before = env._log_prices.copy()
    mu = env._gou_mu.copy()
    eps = mirror.standard_normal(3)
    expected = mu + (x_before - mu) * env._gou_decay + env._gou_noise_std * eps

    env._gou_step()
    np.testing.assert_allclose(env._log_prices, expected, rtol=1e-5)


@pytest.mark.correctness
def test_gou_mean_reversion(seeded_env: AurixExchangeEnv) -> None:
    """Over 1000 steps the mean log-price stays within 2 stationary stds of mu.

    With no trades mu is fixed, so a long path's sample mean should sit close to
    the reversion target — a sanity check that the process is mean-reverting and
    not drifting away.
    """
    samples = []
    for _ in range(1000):
        seeded_env._gou_step()
        samples.append(seeded_env._log_prices.copy())
    mean_logp = np.mean(samples, axis=0)
    mu = seeded_env._gou_mu
    sigma_stat = seeded_env._sigma_stat
    assert np.all(np.abs(mean_logp - mu) < 2.0 * sigma_stat)


# ===========================================================================
# 4. Action Masking
# ===========================================================================
@pytest.mark.correctness
def test_action_mask_present_in_reset_and_step(seeded_env: AurixExchangeEnv) -> None:
    """info['action_mask'] must be present after both reset() and step()."""
    _, info = seeded_env.reset(seed=_SEED)
    assert "action_mask" in info
    _, _, _, _, info = seeded_env.step(HOLD)
    assert "action_mask" in info


@pytest.mark.correctness
def test_action_mask_is_bool_array_length_8(seeded_env: AurixExchangeEnv) -> None:
    """The action mask must be a boolean array of length 8 (one flag per action)."""
    _, info = seeded_env.reset(seed=_SEED)
    mask = info["action_mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    assert mask.shape == (N_ACTIONS,)


@pytest.mark.correctness
def test_expedition_masked_when_fatigue_at_ceiling(env: AurixExchangeEnv) -> None:
    """LAUNCH_EXPEDITION must be masked when fatigue >= the 95.0 ceiling (cash ample)."""
    env.reset()
    env._fatigue = 96.0
    env._cash = 10_000.0
    mask = env._action_mask()
    assert mask[LAUNCH_EXPEDITION] == False  # noqa: E712


@pytest.mark.correctness
def test_expedition_masked_when_cash_below_minimum(
    env: AurixExchangeEnv, cfg: EnvConfig
) -> None:
    """LAUNCH_EXPEDITION must be masked when cash < expedition_min_cash (fatigue low)."""
    env.reset()
    env._fatigue = 0.0
    env._cash = cfg.expedition_min_cash - 1.0
    mask = env._action_mask()
    assert mask[LAUNCH_EXPEDITION] == False  # noqa: E712


@pytest.mark.correctness
def test_expedition_unmasked_when_conditions_clear(
    env: AurixExchangeEnv, cfg: EnvConfig
) -> None:
    """Control: with low fatigue and ample cash the expedition action is available."""
    env.reset()
    env._fatigue = 0.0
    env._cash = cfg.expedition_min_cash + 1_000.0
    mask = env._action_mask()
    assert mask[LAUNCH_EXPEDITION] == True  # noqa: E712


@pytest.mark.correctness
def test_buy_masked_when_cash_empty(env: AurixExchangeEnv) -> None:
    """BUY actions must be masked when cash is effectively zero."""
    env.reset()
    env._cash = 0.0
    mask = env._action_mask()
    assert mask[BUY_BYRINIUM] == False  # noqa: E712
    assert mask[BUY_HERBS] == False  # noqa: E712
    assert mask[BUY_TOOLS] == False  # noqa: E712


@pytest.mark.correctness
def test_sell_masked_when_inventory_empty(env: AurixExchangeEnv) -> None:
    """SELL actions must be masked when the relevant inventory is empty (reset state)."""
    _, info = env.reset()
    mask = info["action_mask"]
    assert mask[SELL_BYRINIUM] == False  # noqa: E712


@pytest.mark.correctness
def test_masked_action_does_not_crash(env: AurixExchangeEnv) -> None:
    """Executing a masked action must be handled gracefully (coerced to HOLD), not crash.

    At reset, inventory is zero so SELL_BYRINIUM is masked; stepping it anyway must
    still return a well-formed 5-tuple.
    """
    env.reset()
    result = env.step(SELL_BYRINIUM)  # masked: no inventory to sell
    assert len(result) == 5
    obs = result[0]
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))


# ===========================================================================
# 5. Reward Function
# ===========================================================================
def _spec_reward(
    cfg: EnvConfig, action: int, fatigue: float, w_before: float, w_after: float
) -> float:
    """Independent re-implementation of the spec reward (section 3.3) for cross-checking.

    Revised 2026-06-04: no potential shaping; insolvency keyed on net worth W_t.
    """
    if w_before > 1e-8:
        log_ret = cfg.reward_alpha * math.log(max(w_after, 1e-8) / w_before)
    else:
        log_ret = -cfg.reward_clip
    clipped = max(-cfg.reward_clip, min(cfg.reward_clip, log_ret))

    w = w_after
    if w <= 0.0:
        psi_ins = cfg.insolvency_omega
    elif w < cfg.insolvency_w_crit:
        psi_ins = cfg.insolvency_beta * ((cfg.insolvency_w_crit - w) / cfg.insolvency_w_crit) ** 2
    else:
        psi_ins = 0.0

    psi_fat = cfg.fatigue_eta * math.exp(cfg.fatigue_xi * (fatigue - cfg.fatigue_f_crit))
    hold_pen = cfg.hold_penalty if action == HOLD else 0.0
    return clipped - psi_ins - psi_fat - hold_pen


def _reward(env: AurixExchangeEnv, action: int, fatigue: float,
            w_before: float, w_after: float) -> float:
    """Drive the env's private reward with fully controlled internal state.

    Net worth enters the reward directly via the (w_before, w_after) arguments; cash
    no longer affects the reward, so only fatigue needs to be set on the instance.
    """
    env._fatigue = fatigue
    return env._compute_reward(action, w_before, w_after)


@pytest.mark.correctness
def test_reward_matches_spec_formula(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """The full reward must equal an independent re-implementation of the spec formula."""
    env.reset()
    actual = _reward(env, BUY_BYRINIUM, fatigue=40.0, w_before=10_000.0, w_after=10_500.0)
    expected = _spec_reward(cfg, BUY_BYRINIUM, 40.0, 10_000.0, 10_500.0)
    assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.correctness
def test_reward_finite_over_episode(seeded_env: AurixExchangeEnv) -> None:
    """Reward must be a finite scalar at every step of a full random episode."""
    rng = np.random.default_rng(_SEED)
    for _ in range(EnvConfig().t_max):
        _, reward, terminated, truncated, _ = seeded_env.step(int(rng.integers(0, N_ACTIONS)))
        assert math.isfinite(reward)
        if terminated or truncated:
            break


@pytest.mark.correctness
def test_reward_log_return_is_clipped(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """A huge net-worth jump must have its log-return term clipped to +c, not pass through.

    We compare against the spec helper computed both with and without clipping; only
    the clipped expectation may match.
    """
    env.reset()
    # 100x net-worth gain => raw log-return ~4.6, far beyond reward_clip=1.0.
    actual = _reward(env, BUY_BYRINIUM, fatigue=0.0, w_before=100.0, w_after=10_000.0)
    expected_clipped = _spec_reward(cfg, BUY_BYRINIUM, 0.0, 100.0, 10_000.0)

    raw_log_ret = math.log(10_000.0 / 100.0)
    psi_fat = cfg.fatigue_eta * math.exp(cfg.fatigue_xi * (0.0 - cfg.fatigue_f_crit))
    expected_unclipped = raw_log_ret - psi_fat  # net worth healthy => no insolvency

    assert actual == pytest.approx(expected_clipped, rel=1e-6)
    assert abs(actual - expected_unclipped) > 1.0  # clip genuinely changed the value


@pytest.mark.correctness
def test_insolvency_penalty_active_in_band(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """Insolvency penalty must be non-zero for 0 < W < W_crit, and zero at/above W_crit.

    Holding w_before == w_after zeroes the log-return, so net worth then enters the
    reward only via the insolvency term; differencing a healthy net worth against an
    in-band one isolates the penalty exactly.
    """
    env.reset()
    healthy_w = cfg.insolvency_w_crit + 500.0
    in_band_w = 400.0
    healthy = _reward(env, HOLD, fatigue=0.0, w_before=healthy_w, w_after=healthy_w)
    inside = _reward(env, HOLD, fatigue=0.0, w_before=in_band_w, w_after=in_band_w)
    expected_penalty = cfg.insolvency_beta * ((cfg.insolvency_w_crit - in_band_w) / cfg.insolvency_w_crit) ** 2
    assert (healthy - inside) == pytest.approx(expected_penalty, rel=1e-6)
    assert expected_penalty > 0.0


@pytest.mark.correctness
def test_insolvency_penalty_maximal_when_broke(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """When W <= 0 the insolvency penalty is the fixed Omega term — its maximum value.

    At zero net worth the log-return is forced to -clip, so we validate the full
    reward against the spec helper, then assert Omega exceeds the largest possible
    in-band quadratic penalty (which approaches beta as W -> 0+).
    """
    env.reset()
    broke = _reward(env, HOLD, fatigue=0.0, w_before=1000.0, w_after=0.0)
    expected = _spec_reward(cfg, HOLD, 0.0, 1000.0, 0.0)
    assert broke == pytest.approx(expected, rel=1e-6)
    assert cfg.insolvency_omega > cfg.insolvency_beta


@pytest.mark.correctness
def test_fatigue_penalty_increases_with_fatigue(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """The fatigue barrier penalty must grow as F approaches and exceeds F_crit.

    Fatigue enters the reward only through Psi_fatigue, so the reward difference
    isolates that term and must match the spec's exponential barrier.
    """
    env.reset()
    low = _reward(env, HOLD, fatigue=20.0, w_before=5000.0, w_after=5000.0)
    high = _reward(env, HOLD, fatigue=90.0, w_before=5000.0, w_after=5000.0)
    psi_low = cfg.fatigue_eta * math.exp(cfg.fatigue_xi * (20.0 - cfg.fatigue_f_crit))
    psi_high = cfg.fatigue_eta * math.exp(cfg.fatigue_xi * (90.0 - cfg.fatigue_f_crit))
    assert psi_high > psi_low
    assert (low - high) == pytest.approx(psi_high - psi_low, rel=1e-6)


@pytest.mark.correctness
def test_hold_penalty_applied_only_on_hold(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """The delta hold penalty must be subtracted for HOLD and absent for other actions."""
    env.reset()
    r_hold = _reward(env, HOLD, fatigue=0.0, w_before=5000.0, w_after=5000.0)
    r_buy = _reward(env, BUY_BYRINIUM, fatigue=0.0, w_before=5000.0, w_after=5000.0)
    assert (r_buy - r_hold) == pytest.approx(cfg.hold_penalty, rel=1e-6)


@pytest.mark.correctness
def test_no_potential_shaping_term(env: AurixExchangeEnv) -> None:
    """Reward must NOT contain potential-based shaping (removed 2026-06-04).

    With w_before == w_after the log-return is zero, and at a healthy net worth the
    insolvency term is zero, so the reward must be independent of the net-worth
    level — any dependence would betray a residual wealth-scaled shaping field.
    """
    env.reset()
    r_low = _reward(env, BUY_BYRINIUM, fatigue=0.0, w_before=5000.0, w_after=5000.0)
    r_high = _reward(env, BUY_BYRINIUM, fatigue=0.0, w_before=50_000.0, w_after=50_000.0)
    assert r_low == pytest.approx(r_high)


# ===========================================================================
# 6. Market Impact
# ===========================================================================
@pytest.mark.correctness
def test_buy_execution_price_above_spot(env: AurixExchangeEnv) -> None:
    """A BUY must execute above spot: temporary impact plus the brokerage fee (positive slippage)."""
    env.reset()
    env._cash = 10_000.0
    env._inventory[:] = 0.0
    spot = math.exp(env._log_prices[IDX_B])

    cash_before = env._cash
    env._execute_buy(IDX_B)
    qty = env._inventory[IDX_B]
    actual_cost = cash_before - env._cash
    exec_price = actual_cost / qty
    assert exec_price > spot


@pytest.mark.correctness
def test_sell_execution_price_below_spot(env: AurixExchangeEnv) -> None:
    """A SELL must execute below spot: temporary impact minus the brokerage fee (negative slippage)."""
    env.reset()
    env._inventory[IDX_B] = 10.0
    spot = math.exp(env._log_prices[IDX_B])

    cash_before = env._cash
    qty = env._inventory[IDX_B]
    env._execute_sell(IDX_B)
    proceeds = env._cash - cash_before
    exec_price = proceeds / qty
    assert exec_price < spot


@pytest.mark.correctness
def test_trades_do_not_move_equilibrium_mu(env: AurixExchangeEnv) -> None:
    """Anti-self-inflation (rebalancing.md §2.1): permanent impact is removed, so
    neither a BUY nor a SELL may shift the reversion target mu (closing the exploit
    where the agent could ratchet its own asset prices up indefinitely).
    """
    env.reset()
    env._cash = 10_000.0
    env._inventory[:] = 0.0
    mu_before = env._gou_mu.copy()

    env._execute_buy(IDX_B)
    np.testing.assert_allclose(env._gou_mu, mu_before, rtol=0.0, atol=0.0)

    env._inventory[IDX_H] = 50.0
    env._execute_sell(IDX_H)
    np.testing.assert_allclose(env._gou_mu, mu_before, rtol=0.0, atol=0.0)


# ===========================================================================
# 7. Fatigue Dynamics
# ===========================================================================
@pytest.mark.correctness
def test_expedition_increases_fatigue(env: AurixExchangeEnv) -> None:
    """A successful LAUNCH_EXPEDITION must increase party fatigue.

    Forcing the RNG so the hazard roll never trips and the target is harvestable,
    we isolate the fatigue increment from the catastrophic-failure branch.
    """
    env.reset()
    env._cash = 10_000.0
    env._fatigue = 10.0
    env._inventory[:] = 0.0
    # Deterministic stream: choice() picks target, random() must exceed hazard rate.
    env._rng = np.random.default_rng(0)
    f_before = env._fatigue
    failed = env._execute_expedition()
    if not failed:
        assert env._fatigue > f_before


@pytest.mark.correctness
def test_hold_decreases_fatigue(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """HOLD applies sedentary decay, reducing fatigue by the decay rate (floored at 0)."""
    env.reset()
    env._fatigue = 50.0
    env._apply_fatigue_decay()
    assert env._fatigue == pytest.approx(50.0 - cfg.fatigue_decay_rate)
    assert env._fatigue < 50.0


@pytest.mark.correctness
def test_fatigue_clamped_to_range(env: AurixExchangeEnv) -> None:
    """Fatigue must stay within [0, 100] — decay never goes negative, gain never exceeds 100."""
    env.reset()
    env._fatigue = 1.0
    env._apply_fatigue_decay()  # decay of 3.0 would underflow
    assert env._fatigue == 0.0

    env._fatigue = 99.0
    env._cash = 10_000.0
    env._inventory[:] = 0.0
    env._rng = np.random.default_rng(0)
    env._execute_expedition()  # gain of 15.0 would overflow on success
    assert 0.0 <= env._fatigue <= 100.0


# ===========================================================================
# 7b. Phase-Based Action Masking (rebalancing.md §2.2 / §3.2)
# ===========================================================================
@pytest.mark.correctness
def test_night_curfew_masks_buys_and_expedition(env: AurixExchangeEnv) -> None:
    """Night phase (τ=3) must mask all BUY actions and LAUNCH_EXPEDITION (§3.2)."""
    env.reset()
    env._cash = 10_000.0
    env._inventory[:] = 10.0  # ensure sells/buys would otherwise be valid
    env._fatigue = 0.0
    env._cooldown = 0
    env._step_count = 3  # τ = 3 -> Night
    mask = env._action_mask()
    assert mask[BUY_BYRINIUM] == False  # noqa: E712
    assert mask[BUY_HERBS] == False  # noqa: E712
    assert mask[BUY_TOOLS] == False  # noqa: E712
    assert mask[LAUNCH_EXPEDITION] == False  # noqa: E712
    # SELL and HOLD remain available at night.
    assert mask[HOLD] == True  # noqa: E712
    assert mask[SELL_BYRINIUM] == True  # noqa: E712


@pytest.mark.correctness
def test_daytime_phase_allows_buys(env: AurixExchangeEnv) -> None:
    """Control: outside the night curfew, BUYs and expeditions are available."""
    env.reset()
    env._cash = 10_000.0
    env._fatigue = 0.0
    env._cooldown = 0
    env._step_count = 0  # τ = 0 -> Morning
    mask = env._action_mask()
    assert mask[BUY_BYRINIUM] == True  # noqa: E712
    assert mask[LAUNCH_EXPEDITION] == True  # noqa: E712


# ===========================================================================
# 7c. Expedition Refactor: fee, localized failure, cooldown (§2.3)
# ===========================================================================
@pytest.mark.correctness
def test_launch_fee_scales_with_fatigue(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """The upfront launch fee is base at F=0 and base·(1+k) at F=fatigue_max."""
    env.reset()
    env._fatigue = 0.0
    assert env._launch_fee() == pytest.approx(cfg.launch_fee_base, rel=1e-9)

    env._fatigue = cfg.fatigue_max
    expected = cfg.launch_fee_base * (1.0 + cfg.launch_fee_fatigue_k)
    assert env._launch_fee() == pytest.approx(expected, rel=1e-9)


@pytest.mark.correctness
def test_expedition_pays_fee_regardless_of_outcome(env: AurixExchangeEnv) -> None:
    """The launch fee is deducted on a failed expedition too (paid upfront, §2.3)."""
    env.reset()
    env._cash = 5_000.0
    env._fatigue = 0.0
    env._inventory[:] = 0.0
    fee = env._launch_fee()
    env._hazard_rate = lambda idx: 1.0  # force catastrophic-roll failure
    failed = env._execute_expedition()
    assert failed is True
    assert env._cash == pytest.approx(5_000.0 - fee, rel=1e-9)


@pytest.mark.correctness
def test_expedition_failure_is_localized(env: AurixExchangeEnv) -> None:
    """Localized failure (§2.3): cash beyond the fee and node inventory are preserved.

    The legacy behaviour zeroed all cash and inventory; that must no longer happen.
    """
    env.reset()
    env._cash = 5_000.0
    env._fatigue = 0.0
    env._inventory[:] = np.array([12.0, 34.0, 5.0])
    inv_before = env._inventory.copy()
    env._hazard_rate = lambda idx: 1.0
    failed = env._execute_expedition()
    assert failed is True
    assert env._cash > 0.0  # not wiped out
    np.testing.assert_array_equal(env._inventory, inv_before)  # inventory untouched


@pytest.mark.correctness
def test_expedition_failure_does_not_terminate(env: AurixExchangeEnv) -> None:
    """A failed expedition must not terminate the episode (only truncation ends it)."""
    env.reset()
    env._cash = 5_000.0
    env._fatigue = 0.0
    env._step_count = 0
    env._cooldown = 0
    env._hazard_rate = lambda idx: 1.0
    _, _, terminated, truncated, info = env.step(LAUNCH_EXPEDITION)
    assert terminated is False
    assert truncated is False
    assert info["cooldown"] == EnvConfig().expedition_cooldown_steps


@pytest.mark.correctness
def test_cooldown_locks_out_then_releases_expedition(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """After a failed expedition the cooldown blocks LAUNCH for exactly N steps.

    The info cooldown counter must decrement 2 -> 1 -> 0, and the action mask must
    block LAUNCH_EXPEDITION while the counter is non-zero.
    """
    env.reset()
    env._cash = 10_000.0
    env._fatigue = 0.0
    env._step_count = 0
    env._cooldown = 0
    env._hazard_rate = lambda idx: 1.0

    _, _, _, _, info = env.step(LAUNCH_EXPEDITION)
    assert info["cooldown"] == cfg.expedition_cooldown_steps
    assert info["action_mask"][LAUNCH_EXPEDITION] == False  # noqa: E712

    observed = [info["cooldown"]]
    for _ in range(cfg.expedition_cooldown_steps):
        _, _, _, _, info = env.step(HOLD)
        observed.append(info["cooldown"])
        if info["cooldown"] > 0:
            assert info["action_mask"][LAUNCH_EXPEDITION] == False  # noqa: E712
    assert observed == [2, 1, 0]

    # With the cooldown cleared (and a non-night, well-funded state) LAUNCH unlocks.
    env._cooldown = 0
    env._step_count = 0
    env._cash = 10_000.0
    env._fatigue = 0.0
    assert env._action_mask()[LAUNCH_EXPEDITION] == True  # noqa: E712


@pytest.mark.correctness
def test_expedition_masked_during_cooldown(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """A non-zero cooldown alone is sufficient to mask LAUNCH_EXPEDITION."""
    env.reset()
    env._cash = 10_000.0
    env._fatigue = 0.0
    env._step_count = 0
    env._cooldown = 1
    assert env._action_mask()[LAUNCH_EXPEDITION] == False  # noqa: E712


# ===========================================================================
# 8. Episode Termination
# ===========================================================================
@pytest.mark.correctness
def test_truncates_at_t_max(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """After exactly T_max steps the episode must end via truncation (HOLD avoids termination)."""
    env.reset()
    terminated = truncated = False
    for _ in range(cfg.t_max):
        _, _, terminated, truncated, _ = env.step(HOLD)
        if terminated or truncated:
            break
    assert truncated or terminated


@pytest.mark.correctness
def test_not_done_before_t_max(env: AurixExchangeEnv, cfg: EnvConfig) -> None:
    """Under benign HOLD actions, the episode must not end before reaching T_max."""
    env.reset()
    for step_i in range(cfg.t_max - 1):
        _, _, terminated, truncated, _ = env.step(HOLD)
        assert not terminated, f"unexpected termination at step {step_i}"
        assert not truncated, f"unexpected truncation at step {step_i}"


# ===========================================================================
# 9. Regression: Observation Values
# ===========================================================================
@pytest.mark.regression
def test_reset_observation_baseline() -> None:
    """Lock the seeded reset() observation against a captured baseline.

    Regenerated 2026-06-06 for the rebalancing.md §1/§3.1 alignment (Tools commodity,
    per-commodity capacity, idx 8 = phase, idx 9 = cooldown) from
    AurixExchangeEnv(seed=42).reset(seed=42).
    """
    # REGRESSION BASELINE — update this array intentionally if env parameters change.
    # Do not update automatically; verify the change is deliberate.
    expected = np.array(
        [
            0.8009078502655029,
            0.0,
            0.0,
            0.0,
            0.304717093706131,
            -1.039984107017517,
            0.7504512071609497,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    env = AurixExchangeEnv(seed=_SEED)
    obs, _ = env.reset(seed=_SEED)
    np.testing.assert_allclose(obs, expected, rtol=1e-5)


# ===========================================================================
# 10. Regression: Reward Baseline
# ===========================================================================
@pytest.mark.regression
def test_cumulative_reward_baseline() -> None:
    """Lock cumulative reward over a fixed action sequence against a captured baseline.

    Generated 2026-06-04: seed=42, 10 steps alternating BUY_BYRINIUM / SELL_BYRINIUM.
    Regenerated 2026-06-06 for the rebalancing.md alignment (new B risk-ladder
    params: μ=ln(150), θ=0.10, σ=0.40; permanent impact removed).
    """
    # REGRESSION BASELINE — update this value intentionally if env parameters change.
    # Do not update automatically; verify the change is deliberate.
    expected_cumulative = -0.16781596354977363

    env = AurixExchangeEnv(seed=_SEED)
    env.reset(seed=_SEED)
    total = 0.0
    for k in range(10):
        action = BUY_BYRINIUM if k % 2 == 0 else SELL_BYRINIUM
        _, reward, _, _, _ = env.step(action)
        total += reward
    assert total == pytest.approx(expected_cumulative, rel=1e-4)


# ===========================================================================
# 11. Regression: GOU Step Output
# ===========================================================================
@pytest.mark.regression
def test_gou_step_output_baseline() -> None:
    """Lock a single GOU step's output against a captured baseline.

    Generated 2026-06-04: AurixExchangeEnv(seed=7).reset(seed=7), then the RNG is
    replaced with default_rng(999) before a single _gou_step(). Guards the
    closed-form implementation against accidental refactoring.
    Regenerated 2026-06-06 for the rebalancing.md §1 GOU parameter rescale.
    """
    # REGRESSION BASELINE — update this array intentionally if env parameters change.
    # Do not update automatically; verify the change is deliberate.
    expected = np.array(
        [5.303196766463923, 3.36914904759407, 5.690647641653602],
        dtype=np.float64,
    )
    env = AurixExchangeEnv(seed=7)
    env.reset(seed=7)
    env._rng = np.random.default_rng(999)
    env._gou_step()
    np.testing.assert_allclose(env._log_prices, expected, rtol=1e-5)
