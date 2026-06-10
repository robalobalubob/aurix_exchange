"""Clean OU trading core — the single source of truth for transition, cost, and
reward used by the DP solver, the MPC oracle, and every learner (phase1_plan §2).

The market is a discretized Ornstein-Uhlenbeck process on the log price. Working
in standardized coordinates makes the normalized deviation ``z = (X - mu) / sigma_stat``
an *exact* unit-variance AR(1):

    z' = rho * z + sqrt(1 - rho^2) * eps,    eps ~ N(0, 1)

with ``rho = exp(-theta)`` and ``sigma_stat = sigma / sqrt(2*theta)``. Because the
clean-core cost model is proportional only (phase1_plan D3), log-utility is
scale-invariant and the trading MDP reduces to the state ``(z, f, t)`` where
``f in [0, 1]`` is the fraction of net worth held in the asset — wealth never
enters. Any duplication of the transition/cost/reward logic outside this module
is a bug class that silently invalidates regret, so the DP solver, MPC, and the
learners all import these same pure functions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Action indices
# ---------------------------------------------------------------------------
# Primary asymmetric set (phase1_plan D4): the only set that can regret-score the
# existing DQN. BUY allocates a fixed fraction of cash; SELL liquidates fully.
CORE_HOLD = 0
CORE_BUY = 1
CORE_SELL = 2

ACTION_SET_ASYMMETRIC = "asymmetric"
ACTION_SET_TARGET = "target"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OUCoreConfig:
    """Canonical clean-core instance: single-asset Byrinium, slow-reversion regime
    (phase1_plan D5). Herbs/Tools become parameter sweeps through the same solver.
    """

    # OU price dynamics
    theta: float = 0.10
    sigma: float = 0.40
    mu: float = math.log(150.0)

    # Cost model (D3): proportional brokerage fee only. The optional per-unit-impact
    # axis is deferred to Phase 3 because it breaks scale-invariance.
    fee: float = 0.01

    # Action set
    action_set: str = ACTION_SET_ASYMMETRIC
    buy_fraction: float = 0.25  # primary set: BUY trades this fraction of cash
    # Secondary target-fraction set; ignored unless action_set == "target".
    target_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    # Objective (D1/D2/D6/D7): finite-horizon, undiscounted, unclipped log-growth.
    t_max: int = 200
    gamma: float = 1.0
    reward_clip: bool = False
    reward_clip_value: float = 1.0
    time_feature: bool = True  # whether t/T enters the observation

    # Derived AR(1) coefficients; populated in __post_init__ (frozen-safe).
    rho: float = field(init=False, default=0.0)
    sigma_stat: float = field(init=False, default=0.0)
    noise_scale: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        rho = math.exp(-self.theta)
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "sigma_stat", self.sigma / math.sqrt(2.0 * self.theta))
        object.__setattr__(self, "noise_scale", math.sqrt(1.0 - rho * rho))

    @property
    def n_actions(self) -> int:
        if self.action_set == ACTION_SET_ASYMMETRIC:
            return 3
        return len(self.target_fractions)

    @property
    def obs_dim(self) -> int:
        return 3 if self.time_feature else 2


# ---------------------------------------------------------------------------
# Pure dynamics (NumPy-vectorizable over z / f; action is a Python scalar)
# ---------------------------------------------------------------------------
def step_z(z: np.ndarray | float, eps: np.ndarray | float, cfg: OUCoreConfig):
    """Advance the normalized price deviation one step: z' = rho*z + sqrt(1-rho^2)*eps."""
    return cfg.rho * np.asarray(z, dtype=np.float64) + cfg.noise_scale * np.asarray(
        eps, dtype=np.float64
    )


def delta_logprice(z: np.ndarray | float, eps: np.ndarray | float, cfg: OUCoreConfig):
    """Log-price increment dX = sigma_stat * (z' - z).

    Derived from ``step_z`` rather than re-expanded so there is a single source of
    truth for the shock: dX and z' can never silently disagree.
    """
    z = np.asarray(z, dtype=np.float64)
    return cfg.sigma_stat * (step_z(z, eps, cfg) - z)


def apply_trade(f: np.ndarray | float, action: int, cfg: OUCoreConfig):
    """Map (pre-trade fraction f, action) -> (post-trade fraction f', proportional
    cost c), derived from the post-fee ledger (the single source of truth — not the
    closed-form shortcut in the spec, which omits a term).

    Ledger with wealth W = 1, cash = 1 - f, asset value = f:
      BUY  spends notional n = b*(1-f) of cash; fee phi*n; asset gains n*(1-phi).
           post-trade wealth W' = 1 - phi*n, so c = phi*n, f' = (f + n*(1-phi)) / W'.
      SELL liquidates f at fee phi; c = phi*f, f' = 0.
      TARGET moves to a fixed fraction; c = phi*|target - f|, f' = target.
    """
    f = np.asarray(f, dtype=np.float64)
    phi = cfg.fee

    if cfg.action_set == ACTION_SET_ASYMMETRIC:
        if action == CORE_HOLD:
            return f, np.zeros_like(f)
        if action == CORE_BUY:
            n = cfg.buy_fraction * (1.0 - f)
            cost = phi * n
            f_prime = (f + n * (1.0 - phi)) / (1.0 - cost)
            return f_prime, cost
        if action == CORE_SELL:
            cost = phi * f
            return np.zeros_like(f), cost
        raise ValueError(f"invalid asymmetric action {action}")

    if cfg.action_set == ACTION_SET_TARGET:
        target = cfg.target_fractions[action]
        cost = phi * np.abs(target - f)
        return np.full_like(f, target), cost

    raise ValueError(f"unknown action_set {cfg.action_set!r}")


def reward(
    f_prime: np.ndarray | float,
    cost: np.ndarray | float,
    dX: np.ndarray | float,
    cfg: OUCoreConfig,
):
    """Per-step log-return reward (the canonical, unclipped objective):

        r = ln(1 - c) + ln((1 - f') + f' * e^dX)

    The two terms decompose the per-step log-wealth change into the fee drag and the
    holding return. Clipping is applied only when explicitly enabled (the twin study).
    """
    f_prime = np.asarray(f_prime, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    dX = np.asarray(dX, dtype=np.float64)

    r = np.log1p(-cost) + np.log((1.0 - f_prime) + f_prime * np.exp(dX))
    if cfg.reward_clip:
        r = np.clip(r, -cfg.reward_clip_value, cfg.reward_clip_value)
    return r


def next_f(f_prime: np.ndarray | float, dX: np.ndarray | float):
    """Post-step asset fraction: f_next = f' * e^dX / ((1 - f') + f' * e^dX).

    The post-trade wealth factor (1 - c) cancels, so cost does not enter here.
    """
    f_prime = np.asarray(f_prime, dtype=np.float64)
    growth = f_prime * np.exp(np.asarray(dX, dtype=np.float64))
    return growth / ((1.0 - f_prime) + growth)


# ---------------------------------------------------------------------------
# Gymnasium adapter
# ---------------------------------------------------------------------------
class OUTradingEnv(gym.Env):
    """Thin Gymnasium wrapper over the pure functions above.

    Observation ``[z, f, t/T]`` (float32; t/T dropped when time_feature is off).
    Reaching ``t == t_max`` is ``terminated=True`` (phase1_plan D2): with time in the
    state the horizon is a true MDP boundary, so the bootstrap is correctly zeroed.
    Reset samples ``z ~ N(0, 1)``, ``f = 0`` (D8 — the env's true start law).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[OUCoreConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or OUCoreConfig()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.cfg.n_actions)

        self._rng = np.random.default_rng(seed)
        self._z: float = 0.0
        self._f: float = 0.0
        self._t: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._z = float(self._rng.standard_normal())
        self._f = 0.0
        self._t = 0

        obs = self._obs()
        info = {"action_mask": self._action_mask()}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.cfg

        f_prime, cost = apply_trade(self._f, action, cfg)
        eps = float(self._rng.standard_normal())
        dX = delta_logprice(self._z, eps, cfg)

        r = float(reward(f_prime, cost, dX, cfg))
        self._f = float(next_f(f_prime, dX))
        self._z = float(step_z(self._z, eps, cfg))
        self._t += 1

        # Time is in the state, so the horizon is a genuine MDP boundary (D2).
        terminated = self._t >= cfg.t_max
        truncated = False
        obs = self._obs()
        info = {"action_mask": self._action_mask(), "z": self._z, "f": self._f}
        return obs, r, terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.time_feature:
            obs = np.array(
                [self._z, self._f, self._t / cfg.t_max], dtype=np.float32
            )
        else:
            obs = np.array([self._z, self._f], dtype=np.float32)

        assert obs.shape == (cfg.obs_dim,)
        return obs

    def _action_mask(self) -> np.ndarray:
        """All actions are always valid in the clean core; a trivial all-true mask
        keeps the learner's masked code path identical to the game env's (§6)."""
        return np.ones(self.cfg.n_actions, dtype=bool)
