from __future__ import annotations

import math
from dataclasses import dataclass
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
BUY_CRYSTALS = 5
SELL_CRYSTALS = 6
LAUNCH_EXPEDITION = 7

N_ACTIONS = 8
OBS_DIM = 10

# Commodity indices
IDX_B = 0
IDX_H = 1
IDX_C = 2

_BUY_TO_IDX: dict[int, int] = {BUY_BYRINIUM: IDX_B, BUY_HERBS: IDX_H, BUY_CRYSTALS: IDX_C}
_SELL_TO_IDX: dict[int, int] = {SELL_BYRINIUM: IDX_B, SELL_HERBS: IDX_H, SELL_CRYSTALS: IDX_C}


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    # Episode
    t_max: int = 200
    initial_cash: float = 10_000.0

    # GOU parameters per commodity [B, H, C]
    # σ_stat = σ / sqrt(2θ) must match C# normalizer: B=0.50, H=0.15, C=0.90
    gou_mu: tuple[float, ...] = (6.9, 3.4, 9.2)
    gou_theta: tuple[float, ...] = (0.10, 0.50, 0.05)
    gou_sigma: tuple[float, ...] = (0.22361, 0.15000, 0.28460)

    # Inventory
    max_inventory: float = 100.0

    # ln(max_expected_cash) — matches C# divisor of 11.5
    cash_norm: float = 11.5

    # Market impact
    brokerage_fee: float = 0.01
    temp_impact: float = 0.002
    perm_impact: tuple[float, ...] = (0.0001, 0.00005, 0.0005)

    # Expedition
    expedition_fatigue_gain: float = 15.0
    fatigue_decay_rate: float = 3.0
    expedition_harvest: float = 10.0
    expedition_min_cash: float = 200.0
    # Echo Crystals weight=0: spec forbids harvesting C via expedition
    expedition_weights: tuple[float, ...] = (0.5, 0.5, 0.0)

    # Hazard function P_fail(F, i) = p_base + (1-p_base) / (1 + exp(-κ(F - midpoint)))
    hazard_p_base: tuple[float, ...] = (0.01, 0.005, 0.0)
    hazard_kappa: tuple[float, ...] = (0.10, 0.10, 0.10)
    hazard_midpoint: tuple[float, ...] = (60.0, 65.0, 50.0)

    # Reward hyperparameters
    reward_alpha: float = 1.0
    reward_clip: float = 1.0
    reward_gamma: float = 0.99
    reward_pot_w: float = 0.1
    insolvency_beta: float = 2.0
    insolvency_c_crit: float = 500.0
    insolvency_omega: float = 10.0
    fatigue_eta: float = 0.01
    fatigue_xi: float = 0.10
    fatigue_f_crit: float = 75.0
    hold_penalty: float = 0.001


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

        self._rng = np.random.default_rng(seed)

        # Mutable state — initialised in reset()
        self._log_prices: np.ndarray = np.zeros(3, dtype=np.float64)
        self._gou_mu: np.ndarray = np.array(self.cfg.gou_mu, dtype=np.float64)
        self._cash: float = 0.0
        self._inventory: np.ndarray = np.zeros(3, dtype=np.float64)
        self._fatigue: float = 0.0
        self._step_count: int = 0

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

        obs = self._normalize_obs()
        info = {"action_mask": self._action_mask(), "net_worth": self._net_worth()}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        mask = self._action_mask()
        if not mask[action]:
            action = HOLD

        w_before = self._net_worth()
        terminated = False

        if action == HOLD:
            self._apply_fatigue_decay()
        elif action in _BUY_TO_IDX:
            self._execute_buy(_BUY_TO_IDX[action])
            self._apply_fatigue_decay()
        elif action in _SELL_TO_IDX:
            self._execute_sell(_SELL_TO_IDX[action])
            self._apply_fatigue_decay()
        elif action == LAUNCH_EXPEDITION:
            terminated = self._execute_expedition()

        self._gou_step()
        self._step_count += 1

        w_after = self._net_worth()
        reward = self._compute_reward(action, w_before, w_after)

        truncated = self._step_count >= self.cfg.t_max
        obs = self._normalize_obs()
        info = {
            "action_mask": self._action_mask(),
            "net_worth": w_after,
            "cash": self._cash,
            "fatigue": self._fatigue,
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

    def _action_mask(self) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)

        # BUY: need enough cash to execute a meaningful trade
        if self._cash < 1e-6:
            mask[BUY_BYRINIUM] = False
            mask[BUY_HERBS] = False
            mask[BUY_CRYSTALS] = False

        # SELL: need non-zero inventory
        for action, idx in _SELL_TO_IDX.items():
            if self._inventory[idx] < 1e-9:
                mask[action] = False

        # LAUNCH_EXPEDITION: fatigue ceiling or insufficient cash reserve
        if self._fatigue >= 95.0 or self._cash < self.cfg.expedition_min_cash:
            mask[LAUNCH_EXPEDITION] = False

        return mask

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute_buy(self, idx: int) -> None:
        spend = 0.25 * self._cash
        S = math.exp(self._log_prices[idx])
        # Approximate qty ignoring impact, then refine exec price
        qty_approx = spend / (S * (1.0 + self.cfg.brokerage_fee))
        exec_price = S * (1.0 + self.cfg.temp_impact * qty_approx) * (1.0 + self.cfg.brokerage_fee)
        qty = spend / exec_price
        # Clamp to remaining capacity
        qty = min(qty, self.cfg.max_inventory - self._inventory[idx])
        actual_cost = qty * exec_price

        self._cash -= actual_cost
        self._inventory[idx] += qty
        self._gou_mu[idx] += self.cfg.perm_impact[idx] * qty

    def _execute_sell(self, idx: int) -> None:
        qty = self._inventory[idx]
        S = math.exp(self._log_prices[idx])
        exec_price = S * (1.0 - self.cfg.temp_impact * qty) * (1.0 - self.cfg.brokerage_fee)
        exec_price = max(exec_price, 1e-8)
        proceeds = qty * exec_price

        self._cash += proceeds
        self._inventory[idx] = 0.0
        self._gou_mu[idx] -= self.cfg.perm_impact[idx] * qty

    # ------------------------------------------------------------------
    # Expedition
    # ------------------------------------------------------------------

    def _execute_expedition(self) -> bool:
        target = int(self._rng.choice(3, p=self._expedition_weights))

        if self._rng.random() < self._hazard_rate(target):
            # Catastrophic failure — complete asset forfeiture
            self._cash = 0.0
            self._inventory[:] = 0.0
            return True

        harvest = min(
            self.cfg.expedition_harvest,
            self.cfg.max_inventory - self._inventory[target],
        )
        self._inventory[target] += harvest
        self._fatigue = min(100.0, self._fatigue + self.cfg.expedition_fatigue_gain)
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

        # Clipped log-return
        if w_before > 1e-8:
            log_ret = cfg.reward_alpha * math.log(max(w_after, 1e-8) / w_before)
        else:
            log_ret = -cfg.reward_clip
        clipped = max(-cfg.reward_clip, min(cfg.reward_clip, log_ret))

        # Potential-based shaping: γΦ(s') - Φ(s)
        phi_after = cfg.reward_pot_w * math.log(max(w_after, 1e-8))
        phi_before = cfg.reward_pot_w * math.log(max(w_before, 1e-8))
        shaping = cfg.reward_gamma * phi_after - phi_before

        # Insolvency penalty Ψ_insolvency(C_t)
        c = self._cash
        if c <= 0.0:
            psi_insolvency = cfg.insolvency_omega
        elif c < cfg.insolvency_c_crit:
            psi_insolvency = cfg.insolvency_beta * (
                (cfg.insolvency_c_crit - c) / cfg.insolvency_c_crit
            ) ** 2
        else:
            psi_insolvency = 0.0

        # Fatigue penalty Ψ_fatigue(F_t)
        psi_fatigue = cfg.fatigue_eta * math.exp(
            cfg.fatigue_xi * (self._fatigue - cfg.fatigue_f_crit)
        )

        hold_pen = cfg.hold_penalty if action == HOLD else 0.0

        return clipped + shaping - psi_insolvency - psi_fatigue - hold_pen

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _normalize_obs(self) -> np.ndarray:
        cfg = self.cfg
        mu = self._gou_mu
        net_worth = self._net_worth()

        obs = np.array(
            [
                math.log(max(self._cash, 0.0) + 1.0) / cfg.cash_norm,
                self._inventory[IDX_B] / cfg.max_inventory,
                self._inventory[IDX_H] / cfg.max_inventory,
                self._inventory[IDX_C] / cfg.max_inventory,
                (self._log_prices[IDX_B] - mu[IDX_B]) / self._sigma_stat[IDX_B],
                (self._log_prices[IDX_H] - mu[IDX_H]) / self._sigma_stat[IDX_H],
                (self._log_prices[IDX_C] - mu[IDX_C]) / self._sigma_stat[IDX_C],
                self._fatigue / 100.0,
                self._step_count / cfg.t_max,
                math.log(max(net_worth, 0.0) + 1.0) / cfg.cash_norm,
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
