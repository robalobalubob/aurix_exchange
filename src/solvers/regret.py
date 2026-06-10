"""Regret harness for the clean OU trading core (phase1_plan §5).

Regret is the optimality gap against the DP ground truth:

    Regret(pi) = E_z[V*_0(z, 0)] - E_hat[ ln(W_T / W_0) under pi ]

estimated over N seeded rollouts from the env's true start distribution (z ~ N(0,1),
f = 0, t = 0 — D8). Every policy in a comparison consumes the *same* per-episode seed
block (common random numbers), which slashes the variance of regret *differences*.

A policy is anything mapping ``(obs, t) -> action`` (DP table, MPC, a DQN checkpoint,
a heuristic, random). One interface, one harness. This module stays framework-agnostic
(no torch import); learners wrap their net in a closure of this shape.

The canonical objective is unclipped, so the undiscounted sum of per-step rewards
equals ln(W_T / W_0) exactly. The harness therefore always rolls with reward clipping
*off*, regardless of how a policy was trained — a clipped-trained agent is still scored
on the canonical objective.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from src.env.ou_core import OUCoreConfig, OUTradingEnv

# A policy maps (observation, timestep) -> discrete action.
Policy = Callable[[np.ndarray, int], int]


# ---------------------------------------------------------------------------
# Policy adapters
# ---------------------------------------------------------------------------
def from_dp(res) -> Policy:
    """Wrap a DPResult's time-dependent optimal policy."""
    return lambda obs, t: res.policy_at(t, float(obs[0]), float(obs[1]))


def from_mpc(pol) -> Policy:
    """Wrap a stationary MPC controller."""
    return lambda obs, t: pol.policy_at(float(obs[0]), float(obs[1]))


def random_policy(n_actions: int, seed: int = 0) -> Policy:
    """Uniform-random valid action (clean core has no mask), seeded for repeatability."""
    rng = np.random.default_rng(seed)
    return lambda obs, t: int(rng.integers(n_actions))


def constant_policy(action: int) -> Policy:
    """Always take a fixed action (e.g. buy-and-hold via repeated BUY, or all-HOLD)."""
    return lambda obs, t: action


# ---------------------------------------------------------------------------
# Rollouts (common random numbers)
# ---------------------------------------------------------------------------
def make_seed_block(n_episodes: int, seed0: int = 10_000) -> np.ndarray:
    """The shared per-episode seed block (formalizes the accidental shared seed0 in
    evaluate.py / run_baselines.py)."""
    return seed0 + np.arange(n_episodes, dtype=np.int64)


def rollout(
    policy: Policy,
    cfg: OUCoreConfig,
    seeds: np.ndarray,
    reference: Optional[Policy] = None,
) -> tuple[np.ndarray, float]:
    """Roll ``policy`` once per seed; return (per-episode ln(W_T/W_0), agreement-rate).

    Agreement is the fraction of visited states where ``policy`` and ``reference``
    pick the same action (a diagnostic only — policies can disagree on states that
    carry no value). When ``reference`` is None the agreement rate is NaN.
    """
    eval_cfg = replace(cfg, reward_clip=False)  # canonical objective, always
    env = OUTradingEnv(config=eval_cfg, seed=int(seeds[0]))

    returns = np.empty(len(seeds), dtype=np.float64)
    agree_num = 0
    agree_den = 0
    for i, s in enumerate(seeds):
        obs, _ = env.reset(seed=int(s))
        total, t = 0.0, 0
        while True:
            a = policy(obs, t)
            if reference is not None:
                agree_num += int(a == reference(obs, t))
                agree_den += 1
            obs, r, term, trunc, _ = env.step(a)
            total += r
            t += 1
            if term or trunc:
                break
        returns[i] = total

    agreement = (agree_num / agree_den) if agree_den else float("nan")
    return returns, agreement


# ---------------------------------------------------------------------------
# Regret report
# ---------------------------------------------------------------------------
@dataclass
class RegretReport:
    n_episodes: int
    v_star: float            # E_z[V*_0(z, 0)]
    return_mean: float       # E_hat[ln(W_T/W_0)]
    return_ci: float         # 95% half-width
    regret: float            # v_star - return_mean
    paired_regret: float     # mean(return_pistar - return_policy), CRN
    paired_ci: float         # 95% half-width of the paired difference
    agreement_rate: float    # fraction of decisions matching pi*
    median_growth_x: float   # median W_T/W_0
    p10_growth_x: float
    ruin_rate: float         # fraction with W_T/W_0 < ruin_threshold
    loss_rate: float         # fraction with W_T/W_0 < 1

    def render(self) -> str:
        return (
            f"episodes            {self.n_episodes}\n"
            f"E_z[V*_0]           {self.v_star:>10.4f}\n"
            f"mean ln(W_T/W_0)    {self.return_mean:>10.4f}  +/- {self.return_ci:.4f}\n"
            f"regret (vs V*)      {self.regret:>10.4f}\n"
            f"paired regret (CRN) {self.paired_regret:>10.4f}  +/- {self.paired_ci:.4f}\n"
            f"decision agreement  {self.agreement_rate:>10.1%}\n"
            f"median growth       {self.median_growth_x:>10.2f}x\n"
            f"P10 growth          {self.p10_growth_x:>10.2f}x\n"
            f"ruin rate           {self.ruin_rate:>10.1%}\n"
            f"loss rate (<1x)     {self.loss_rate:>10.1%}"
        )


def _ci95(x: np.ndarray) -> float:
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def evaluate_regret(
    policy: Policy,
    dp_result,
    n_episodes: int = 2000,
    seed0: int = 10_000,
    pistar_returns: Optional[np.ndarray] = None,
    ruin_threshold: float = 0.1,
) -> RegretReport:
    """Score ``policy`` against the DP optimum under common random numbers.

    ``pistar_returns`` (the optimum's per-episode returns on the same seed block) can
    be precomputed once and reused across many candidate policies for a cheap, paired
    comparison.
    """
    cfg = dp_result.cfg
    seeds = make_seed_block(n_episodes, seed0)
    ref = from_dp(dp_result)

    returns, agreement = rollout(policy, cfg, seeds, reference=ref)
    if pistar_returns is None:
        pistar_returns, _ = rollout(ref, cfg, seeds)

    v_star = dp_result.expected_initial_value()
    growth = np.exp(returns)
    paired = pistar_returns - returns

    return RegretReport(
        n_episodes=n_episodes,
        v_star=v_star,
        return_mean=float(returns.mean()),
        return_ci=_ci95(returns),
        regret=float(v_star - returns.mean()),
        paired_regret=float(paired.mean()),
        paired_ci=_ci95(paired),
        agreement_rate=agreement,
        median_growth_x=float(np.median(growth)),
        p10_growth_x=float(np.percentile(growth, 10)),
        ruin_rate=float(np.mean(growth < ruin_threshold)),
        loss_rate=float(np.mean(growth < 1.0)),
    )
