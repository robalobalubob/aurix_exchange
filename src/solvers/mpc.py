"""MPC second oracle for the clean OU trading core (phase1_plan §3.4).

A receding-horizon controller: at every state it faces the same H-step problem with
zero terminal value, so the controller is the *first-decision* policy of an H-horizon
backward induction and can be precomputed once as a stationary table. As H grows past
a few reversion half-lives the end-effects vanish and the controller approaches the
stationary optimum.

Independence is the point. This module shares only the pure-math core (``ou_core``)
with the DP solver; the backward induction here is implemented from scratch with a
*direct* per-sweep quadrature and its own bilinear interpolation — it does NOT reuse
``dp.py``'s precomputed sparse operators. Two solvers built on different machinery
agreeing within Monte-Carlo error is what makes either trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.env.ou_core import (
    OUCoreConfig,
    apply_trade,
    delta_logprice,
    next_f,
    reward,
    step_z,
)


@dataclass(frozen=True)
class MPCGrid:
    n_z: int = 161
    z_max: float = 4.5
    n_f: int = 121
    n_quad: int = 16
    horizon: int = 35  # ~5 reversion half-lives for Byrinium (ln2/theta ≈ 7)


def _gh_normal(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def _interp(V: np.ndarray, zg: np.ndarray, fg: np.ndarray, zt: np.ndarray, ft: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of grid values V (n_z, n_f) at targets (zt, ft).

    Deliberately a fresh, self-contained implementation (not dp._bilinear_weights)
    so a shared bug cannot make the two oracles agree by construction.
    """
    n_z, n_f = V.shape
    zt = np.clip(zt, zg[0], zg[-1])
    ft = np.clip(ft, fg[0], fg[-1])
    zi = np.clip(np.searchsorted(zg, zt) - 1, 0, n_z - 2)
    fi = np.clip(np.searchsorted(fg, ft) - 1, 0, n_f - 2)
    tz = (zt - zg[zi]) / (zg[zi + 1] - zg[zi])
    tf = (ft - fg[fi]) / (fg[fi + 1] - fg[fi])
    return (
        V[zi, fi] * (1 - tz) * (1 - tf)
        + V[zi, fi + 1] * (1 - tz) * tf
        + V[zi + 1, fi] * tz * (1 - tf)
        + V[zi + 1, fi + 1] * tz * tf
    )


@dataclass
class MPCPolicy:
    """Stationary receding-horizon controller as a precomputed (z, f) action table."""

    table: np.ndarray  # (n_z, n_f) int
    zg: np.ndarray
    fg: np.ndarray
    cfg: OUCoreConfig
    grid: MPCGrid

    def policy_at(self, z: float, f: float) -> int:
        i = int(np.clip(np.searchsorted(self.zg, z), 0, len(self.zg) - 1))
        if i > 0 and abs(self.zg[i - 1] - z) < abs(self.zg[i] - z):
            i -= 1
        j = int(np.clip(round(f * (len(self.fg) - 1)), 0, len(self.fg) - 1))
        return int(self.table[i, j])


def build(cfg: OUCoreConfig | None = None, grid: MPCGrid | None = None) -> MPCPolicy:
    """Run H-step backward induction (terminal value 0) and return the first-decision
    policy table — the stationary MPC controller."""
    cfg = cfg or OUCoreConfig()
    grid = grid or MPCGrid()

    zg = np.linspace(-grid.z_max, grid.z_max, grid.n_z)
    fg = np.linspace(0.0, 1.0, grid.n_f)
    n_z, n_f = grid.n_z, grid.n_f
    eps_k, w_k = _gh_normal(grid.n_quad)

    ZG, FG = np.meshgrid(zg, fg, indexing="ij")  # (n_z, n_f)

    # Per-action post-trade fraction / cost (depend on f only).
    trades = [apply_trade(FG, a, cfg) for a in range(cfg.n_actions)]

    V = np.zeros((n_z, n_f), dtype=np.float64)
    table = np.zeros((n_z, n_f), dtype=np.int16)

    for sweep in range(grid.horizon):
        q = np.empty((cfg.n_actions, n_z, n_f), dtype=np.float64)
        for a in range(cfg.n_actions):
            f_prime, cost = trades[a]
            acc = np.zeros((n_z, n_f), dtype=np.float64)
            for k in range(grid.n_quad):
                dX = delta_logprice(ZG, eps_k[k], cfg)          # (n_z, n_f)
                r = reward(f_prime, cost, dX, cfg)
                z_nx = step_z(ZG, eps_k[k], cfg)
                f_nx = next_f(f_prime, dX)
                acc += w_k[k] * (r + cfg.gamma * _interp(V, zg, fg, z_nx, f_nx))
            q[a] = acc
        V = q.max(axis=0)
        table = q.argmax(axis=0).astype(np.int16)  # first-decision policy each sweep

    return MPCPolicy(table=table, zg=zg, fg=fg, cfg=cfg, grid=grid)
