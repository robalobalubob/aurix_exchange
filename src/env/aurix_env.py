from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Action indices
# ---------------------------------------------------------------------------
HOLD = 0
BUY_BYRINIUM = 1
SELL_BYRINIUM = 2
BUY_HERBS = 3
SELL_HERBS = 4
BUY_TOOLS = 5
SELL_TOOLS = 6
LAUNCH_EXPEDITION = 7

N_ACTIONS = 8
OBS_DIM = 10
N_PHASES = 4

# Commodity indices (rebalancing.md §1: Echo Crystals removed, Refined Tools added)
IDX_B = 0
IDX_H = 1
IDX_T = 2

_BUY_TO_IDX: dict[int, int] = {BUY_BYRINIUM: IDX_B, BUY_HERBS: IDX_H, BUY_TOOLS: IDX_T}
_SELL_TO_IDX: dict[int, int] = {SELL_BYRINIUM: IDX_B, SELL_HERBS: IDX_H, SELL_TOOLS: IDX_T}
# BUY/SELL action indices, used by the night-curfew action mask.
_BUY_ACTIONS: tuple[int, ...] = (BUY_BYRINIUM, BUY_HERBS, BUY_TOOLS)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    # Episode
    t_max: int = 200
    initial_cash: float = 10_000.0

    # GOU parameters per commodity [B, H, T] (rebalancing.md §1 risk-ladder).
    # μ = ln(base price): B=ln(150), H=ln(30), T=ln(600).
    # σ_stat = σ / sqrt(2θ) is the C# price-normalizer divisor: B≈0.894, H=0.15, T≈0.791.
    gou_mu: tuple[float, ...] = (5.0106352940962555, 3.4011973816621555, 6.396929655216146)
    gou_theta: tuple[float, ...] = (0.10, 0.50, 0.05)
    gou_sigma: tuple[float, ...] = (0.40, 0.15, 0.25)

    # Per-commodity carrying capacity I_max [B, H, T] (rebalancing.md §1).
    max_inventory: tuple[float, ...] = (500.0, 2500.0, 100.0)

    # Trade sizing (spec §3.2: BUY allocates exactly 25% of liquid cash)
    buy_fraction: float = 0.25

    # ln(max_expected_cash) — matches C# divisor of 11.5
    cash_norm: float = 11.5

    # Market impact. Permanent impact (μ ← μ + y·q) removed per rebalancing.md §2.1
    # to close the self-inflation exploit; μ is now static at baseline.
    brokerage_fee: float = 0.01
    temp_impact: float = 0.002

    # Expedition
    expedition_fatigue_gain: float = 15.0
    fatigue_decay_rate: float = 3.0
    fatigue_max: float = 100.0
    expedition_fatigue_ceiling: float = 95.0
    expedition_harvest: float = 10.0
    expedition_min_cash: float = 200.0
    # Upfront launch fee, paid on every launch regardless of outcome (rebalancing.md
    # §2.3). Scales with party fatigue: fee = base · (1 + k · F / fatigue_max).
    launch_fee_base: float = 100.0
    launch_fee_fatigue_k: float = 1.0
    # Localized-failure cooldown: steps during which LAUNCH_EXPEDITION is masked.
    expedition_cooldown_steps: int = 2
    # Tools weight=0: expeditions source raw goods only; Tools are manufactured (Phase B).
    expedition_weights: tuple[float, ...] = (0.5, 0.5, 0.0)

    # Hazard function P_fail(F, i) = p_base + (1-p_base) / (1 + exp(-κ(F - midpoint)))
    hazard_p_base: tuple[float, ...] = (0.01, 0.005, 0.0)
    hazard_kappa: tuple[float, ...] = (0.10, 0.10, 0.10)
    hazard_midpoint: tuple[float, ...] = (60.0, 65.0, 50.0)

    # Reward hyperparameters
    reward_alpha: float = 1.0
    reward_clip: float = 1.0
    # Insolvency defense keys off NET WORTH (true ruin), not raw cash — a cash-poor
    # but asset-rich agent is solvent and must not be penalised. See spec §3.3.
    insolvency_beta: float = 2.0
    insolvency_w_crit: float = 1000.0
    insolvency_omega: float = 10.0
    fatigue_eta: float = 0.01
    fatigue_xi: float = 0.10
    fatigue_f_crit: float = 75.0
    hold_penalty: float = 0.001

    # Action-mask thresholds (must be mirrored by the C# client — see sidecar)
    buy_cash_epsilon: float = 1e-6
    sell_inventory_epsilon: float = 1e-9


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class AurixExchangeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Pre-compute stable GOU coefficients (θ and σ are immutable)
        thetas = np.array(self.cfg.gou_theta, dtype=np.float64)
        sigmas = np.array(self.cfg.gou_sigma, dtype=np.float64)

        self._gou_decay: np.ndarray = np.exp(-thetas)
        # Exact closed-form per-step noise std: σ * sqrt((1 - e^{-2θ}) / (2θ))
        self._gou_noise_std: np.ndarray = sigmas * np.sqrt(
            (1.0 - np.exp(-2.0 * thetas)) / (2.0 * thetas)
        )
        # Stationary std for obs normalisation: σ / sqrt(2θ)
        self._sigma_stat: np.ndarray = sigmas / np.sqrt(2.0 * thetas)

        self._expedition_weights = np.array(self.cfg.expedition_weights, dtype=np.float64)
        self._expedition_weights /= self._expedition_weights.sum()

        # Per-commodity carrying capacity, as an array for vectorised normalisation.
        self._max_inventory: np.ndarray = np.array(self.cfg.max_inventory, dtype=np.float64)

        self._rng = np.random.default_rng(seed)

        # Mutable state — initialised in reset(). μ is immutable now (no permanent
        # impact), so it is set once here and never mutated.
        self._log_prices: np.ndarray = np.zeros(3, dtype=np.float64)
        self._gou_mu: np.ndarray = np.array(self.cfg.gou_mu, dtype=np.float64)
        self._cash: float = 0.0
        self._inventory: np.ndarray = np.zeros(3, dtype=np.float64)
        self._fatigue: float = 0.0
        self._step_count: int = 0
        self._cooldown: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._gou_mu = np.array(self.cfg.gou_mu, dtype=np.float64)
        # Sample initial log-prices from stationary distribution N(μ, σ_stat²)
        self._log_prices = self._rng.normal(
            loc=self._gou_mu, scale=self._sigma_stat
        )
        self._cash = self.cfg.initial_cash
        self._inventory = np.zeros(3, dtype=np.float64)
        self._fatigue = 0.0
        self._step_count = 0
        self._cooldown = 0

        obs = self._normalize_obs()
        info = {"action_mask": self._action_mask(), "net_worth": self._net_worth()}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        mask = self._action_mask()
        if not mask[action]:
            action = HOLD

        w_before = self._net_worth()
        expedition_failed = False

        if action == HOLD:
            self._apply_fatigue_decay()
        elif action in _BUY_TO_IDX:
            self._execute_buy(_BUY_TO_IDX[action])
            self._apply_fatigue_decay()
        elif action in _SELL_TO_IDX:
            self._execute_sell(_SELL_TO_IDX[action])
            self._apply_fatigue_decay()
        elif action == LAUNCH_EXPEDITION:
            expedition_failed = self._execute_expedition()

        self._gou_step()
        self._step_count += 1

        # Cooldown bookkeeping (rebalancing.md §2.3): a failed expedition this step
        # arms the cooldown; otherwise an active cooldown ages by one step. The mask
        # at the top of step() already read the pre-decrement value, so blocking
        # spans exactly `expedition_cooldown_steps` subsequent steps.
        if expedition_failed:
            self._cooldown = self.cfg.expedition_cooldown_steps
        elif self._cooldown > 0:
            self._cooldown -= 1

        w_after = self._net_worth()
        reward = self._compute_reward(action, w_before, w_after)

        # Localized failure is not catastrophic — episodes end only by truncation.
        terminated = False
        truncated = self._step_count >= self.cfg.t_max
        obs = self._normalize_obs()
        info = {
            "action_mask": self._action_mask(),
            "net_worth": w_after,
            "cash": self._cash,
            "fatigue": self._fatigue,
            "cooldown": self._cooldown,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Price dynamics
    # ------------------------------------------------------------------

    def _gou_step(self) -> None:
        # X_{t+1} = μ + (X_t - μ)e^{-θ} + σ√((1 - e^{-2θ})/(2θ)) · ε
        eps = self._rng.standard_normal(3)
        self._log_prices = (
            self._gou_mu
            + (self._log_prices - self._gou_mu) * self._gou_decay
            + self._gou_noise_std * eps
        )

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def _phase(self) -> int:
        """Time-of-day phase τ ∈ {0=Morning, 1=Day, 2=Evening, 3=Night} (§2.2)."""
        return self._step_count % N_PHASES

    def _action_mask(self) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)

        # BUY: need enough cash to execute a meaningful trade
        if self._cash < self.cfg.buy_cash_epsilon:
            for action in _BUY_ACTIONS:
                mask[action] = False

        # SELL: need non-zero inventory
        for action, idx in _SELL_TO_IDX.items():
            if self._inventory[idx] < self.cfg.sell_inventory_epsilon:
                mask[action] = False

        # LAUNCH_EXPEDITION: cooldown lockout, fatigue ceiling, or insufficient
        # cash reserve (the reserve covers the worst-case fatigue-scaled launch fee).
        if (
            self._cooldown > 0
            or self._fatigue >= self.cfg.expedition_fatigue_ceiling
            or self._cash < self.cfg.expedition_min_cash
        ):
            mask[LAUNCH_EXPEDITION] = False

        # Night curfew (§3.2): no open-market purchases and no expeditions.
        if self._phase() == 3:
            for action in _BUY_ACTIONS:
                mask[action] = False
            mask[LAUNCH_EXPEDITION] = False

        return mask

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute_buy(self, idx: int) -> None:
        spend = self.cfg.buy_fraction * self._cash
        S = math.exp(self._log_prices[idx])
        # Approximate qty ignoring impact, then refine exec price
        qty_approx = spend / (S * (1.0 + self.cfg.brokerage_fee))
        exec_price = S * (1.0 + self.cfg.temp_impact * qty_approx) * (1.0 + self.cfg.brokerage_fee)
        qty = spend / exec_price
        # Clamp to remaining per-commodity capacity
        qty = min(qty, self._max_inventory[idx] - self._inventory[idx])
        actual_cost = qty * exec_price

        self._cash -= actual_cost
        self._inventory[idx] += qty

    def _execute_sell(self, idx: int) -> None:
        qty = self._inventory[idx]
        S = math.exp(self._log_prices[idx])
        exec_price = S * (1.0 - self.cfg.temp_impact * qty) * (1.0 - self.cfg.brokerage_fee)
        exec_price = max(exec_price, 1e-8)
        proceeds = qty * exec_price

        self._cash += proceeds
        self._inventory[idx] = 0.0

    # ------------------------------------------------------------------
    # Expedition
    # ------------------------------------------------------------------

    def _launch_fee(self) -> float:
        """Upfront launch cost, scaling with party fatigue (rebalancing.md §2.3)."""
        return self.cfg.launch_fee_base * (
            1.0 + self.cfg.launch_fee_fatigue_k * self._fatigue / self.cfg.fatigue_max
        )

    def _execute_expedition(self) -> bool:
        """Run a sourcing expedition. Returns True on localized (non-catastrophic)
        failure, which arms the cooldown but never forfeits cash or node inventory.
        """
        # Upfront fee is paid regardless of outcome.
        self._cash = max(0.0, self._cash - self._launch_fee())
        # The trip is taken either way, so fatigue accrues either way.
        self._fatigue = min(
            self.cfg.fatigue_max, self._fatigue + self.cfg.expedition_fatigue_gain
        )

        target = int(self._rng.choice(3, p=self._expedition_weights))

        if self._rng.random() < self._hazard_rate(target):
            # Localized failure: pending harvest is lost (nothing added); cooldown
            # is armed by the caller. Cash and existing inventory are untouched.
            return True

        harvest = min(
            self.cfg.expedition_harvest,
            self._max_inventory[target] - self._inventory[target],
        )
        self._inventory[target] += harvest
        return False

    def _hazard_rate(self, idx: int) -> float:
        p_base = self.cfg.hazard_p_base[idx]
        kappa = self.cfg.hazard_kappa[idx]
        midpoint = self.cfg.hazard_midpoint[idx]
        return p_base + (1.0 - p_base) / (1.0 + math.exp(-kappa * (self._fatigue - midpoint)))

    def _apply_fatigue_decay(self) -> None:
        self._fatigue = max(0.0, self._fatigue - self.cfg.fatigue_decay_rate)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: int, w_before: float, w_after: float) -> float:
        cfg = self.cfg

        # Clipped log-return — the dense wealth-growth signal.
        if w_before > 1e-8:
            log_ret = cfg.reward_alpha * math.log(max(w_after, 1e-8) / w_before)
        else:
            log_ret = -cfg.reward_clip
        clipped = max(-cfg.reward_clip, min(cfg.reward_clip, log_ret))

        # Insolvency penalty Ψ_insolvency(W_t) — keyed on net worth, not cash, so an
        # asset-rich/cash-poor agent is not punished for being solvent.
        w = w_after
        if w <= 0.0:
            psi_insolvency = cfg.insolvency_omega
        elif w < cfg.insolvency_w_crit:
            psi_insolvency = cfg.insolvency_beta * (
                (cfg.insolvency_w_crit - w) / cfg.insolvency_w_crit
            ) ** 2
        else:
            psi_insolvency = 0.0

        # Fatigue penalty Ψ_fatigue(F_t)
        psi_fatigue = cfg.fatigue_eta * math.exp(
            cfg.fatigue_xi * (self._fatigue - cfg.fatigue_f_crit)
        )

        hold_pen = cfg.hold_penalty if action == HOLD else 0.0

        return clipped - psi_insolvency - psi_fatigue - hold_pen

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _normalize_obs(self) -> np.ndarray:
        cfg = self.cfg
        mu = self._gou_mu
        cap = self._max_inventory

        # obs layout (rebalancing.md §3.1): idx 8 repurposed to phase τ/3, idx 9 to
        # expedition cooldown / max. Net worth is no longer observed (tracked only
        # inside the reward).
        obs = np.array(
            [
                math.log(max(self._cash, 0.0) + 1.0) / cfg.cash_norm,
                self._inventory[IDX_B] / cap[IDX_B],
                self._inventory[IDX_H] / cap[IDX_H],
                self._inventory[IDX_T] / cap[IDX_T],
                (self._log_prices[IDX_B] - mu[IDX_B]) / self._sigma_stat[IDX_B],
                (self._log_prices[IDX_H] - mu[IDX_H]) / self._sigma_stat[IDX_H],
                (self._log_prices[IDX_T] - mu[IDX_T]) / self._sigma_stat[IDX_T],
                self._fatigue / cfg.fatigue_max,
                self._phase() / (N_PHASES - 1),
                self._cooldown / cfg.expedition_cooldown_steps,
            ],
            dtype=np.float32,
        )

        assert obs.shape == (OBS_DIM,)
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _net_worth(self) -> float:
        return float(self._cash + np.dot(self._inventory, np.exp(self._log_prices)))


# ---------------------------------------------------------------------------
# Shared-constants export (single source of truth — see spec Addendum A)
# ---------------------------------------------------------------------------
def export_config(cfg: EnvConfig, path: str) -> None:
    """Serialise the MDP constants to a JSON sidecar.

    This sidecar is the single source of truth shared by the Python training
    environment and the C# deployment client. The C# observation normaliser and
    action-mask thresholds read these values rather than hardcoding literals, so
    the two implementations can never silently drift. Derived quantities
    (``sigma_stat``, GOU decay/noise) are included so the consumer never recomputes
    them — ``sigma_stat`` in particular is what the C# price normaliser divides by.
    """
    thetas = np.array(cfg.gou_theta, dtype=np.float64)
    sigmas = np.array(cfg.gou_sigma, dtype=np.float64)
    sigma_stat = sigmas / np.sqrt(2.0 * thetas)
    gou_decay = np.exp(-thetas)
    gou_noise_std = sigmas * np.sqrt((1.0 - np.exp(-2.0 * thetas)) / (2.0 * thetas))

    payload = {
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "config": asdict(cfg),
        "derived": {
            "sigma_stat": sigma_stat.tolist(),
            "gou_decay": gou_decay.tolist(),
            "gou_noise_std": gou_noise_std.tolist(),
        },
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
