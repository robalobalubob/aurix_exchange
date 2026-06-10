"""Dynamic-programming ground truth for the clean OU trading core (phase1_plan §3).

Backward induction on a (z, f) grid with Gauss-Hermite quadrature over the price
shock. The key implementation move (§3.2): every action's one-step expectation is
*time-invariant*, so we precompute it once as a sparse linear operator ``P_a``
(bilinear-interpolation weights of the K quadrature targets into the grid, weighted
by quadrature weights) plus a reward vector ``R_a``. Backward induction is then T
iterations of |A| sparse matvecs and a pointwise max:

    Q_t(s, a) = R_a(s) + gamma * (P_a V_{t+1})(s),   V_t = max_a Q_t,   pi_t = argmax_a Q_t

Boundary V_T == 0 (the objective is already accumulated through the per-step
log-returns). All transition/cost/reward logic is imported from ``ou_core`` — never
re-implemented here — so the ground truth cannot silently drift from the env.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from src.env.ou_core import (
    OUCoreConfig,
    apply_trade,
    delta_logprice,
    next_f,
    reward,
    step_z,
)


# ---------------------------------------------------------------------------
# Grid / quadrature configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DPGrid:
    """Discretization knobs. Defaults per phase1_plan §3.1."""

    n_z: int = 201          # uniform on [-z_max, z_max]
    z_max: float = 4.5      # stationary mass beyond +/-4.5 is ~7e-6
    n_f: int = 101          # uniform on [0, 1]
    n_quad: int = 16        # Gauss-Hermite nodes (validate vs 32)


# ---------------------------------------------------------------------------
# Internal sparse operator: row-grouped weighted gather (a pure-NumPy matvec).
# ---------------------------------------------------------------------------
class _SparseOp:
    """P_a as COO triplets; ``matvec(V)[s] = sum_k weight_k * V[col_k]`` grouped by
    source state. Avoids a SciPy dependency — the budget (≈1e9 mults) is comfortable.
    """

    __slots__ = ("rows", "cols", "vals", "n")

    def __init__(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n: int) -> None:
        self.rows = rows
        self.cols = cols
        self.vals = vals
        self.n = n

    def matvec(self, V: np.ndarray) -> np.ndarray:
        return np.bincount(self.rows, weights=self.vals * V[self.cols], minlength=self.n)


def _gauss_hermite_normal(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for E[g(eps)], eps ~ N(0,1), via the probabilists' rescaling of
    the physicists' Gauss-Hermite rule: eps = sqrt(2)*x, w = w_phys / sqrt(pi)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def _bilinear_weights(
    zt: np.ndarray, ft: np.ndarray, zg: np.ndarray, fg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear interpolation of target coords (zt, ft) into the (zg, fg) grid.

    Returns (cols, weights), each shape (4, M): the four bracketing grid cells
    (flattened idx = i*Nf + j) and their interpolation weights. Targets are clamped
    to the grid edges (V is near-linear there).
    """
    n_z, n_f = len(zg), len(fg)
    zt = np.clip(zt, zg[0], zg[-1])
    ft = np.clip(ft, fg[0], fg[-1])

    zi = np.clip(np.searchsorted(zg, zt) - 1, 0, n_z - 2)
    fi = np.clip(np.searchsorted(fg, ft) - 1, 0, n_f - 2)

    tz = (zt - zg[zi]) / (zg[zi + 1] - zg[zi])
    tf = (ft - fg[fi]) / (fg[fi + 1] - fg[fi])

    cols = np.stack(
        [
            zi * n_f + fi,
            zi * n_f + (fi + 1),
            (zi + 1) * n_f + fi,
            (zi + 1) * n_f + (fi + 1),
        ]
    )
    weights = np.stack(
        [(1 - tz) * (1 - tf), (1 - tz) * tf, tz * (1 - tf), tz * tf]
    )
    return cols, weights


# ---------------------------------------------------------------------------
# Solver result
# ---------------------------------------------------------------------------
@dataclass
class DPResult:
    """Value/policy tables plus the grids they live on.

    ``V`` has shape (T+1, M) with V[T] == 0; ``pi`` has shape (T, M). States are
    flattened as ``i*n_f + j`` over (z_i, f_j).
    """

    V: np.ndarray
    pi: np.ndarray
    zg: np.ndarray
    fg: np.ndarray
    cfg: OUCoreConfig
    grid: DPGrid

    @property
    def config_hash(self) -> str:
        return config_hash(self.cfg, self.grid)

    def _interp_V(self, t: int, z: float, f: float) -> float:
        cols, w = _bilinear_weights(
            np.array([z], dtype=np.float64), np.array([f], dtype=np.float64), self.zg, self.fg
        )
        return float(np.sum(w[:, 0] * self.V[t][cols[:, 0]]))

    def value_at(self, t: int, z: float, f: float) -> float:
        """Bilinearly interpolated V_t(z, f)."""
        return self._interp_V(t, z, f)

    def policy_at(self, t: int, z: float, f: float) -> int:
        """Greedy action at (z, f, t) via nearest grid cell (policy is discrete)."""
        i = int(np.clip(np.searchsorted(self.zg, z), 0, len(self.zg) - 1))
        # snap to nearest of the two bracketing z nodes
        if i > 0 and abs(self.zg[i - 1] - z) < abs(self.zg[i] - z):
            i -= 1
        j = int(np.clip(round(f * (len(self.fg) - 1)), 0, len(self.fg) - 1))
        return int(self.pi[t][i * len(self.fg) + j])

    def expected_initial_value(self, n: int = 64) -> float:
        """E_{z~N(0,1)}[V_0(z, f=0)] — the regret baseline (start dist per D8)."""
        zq, wq = _gauss_hermite_normal(n)
        return float(sum(wq[k] * self._interp_V(0, zq[k], 0.0) for k in range(n)))

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            V=self.V,
            pi=self.pi,
            zg=self.zg,
            fg=self.fg,
            cfg_json=json.dumps(asdict(self.cfg)),
            grid_json=json.dumps(asdict(self.grid)),
        )

    @staticmethod
    def load(path: str) -> "DPResult":
        data = np.load(path, allow_pickle=False)
        cfg_fields = json.loads(str(data["cfg_json"]))
        grid_fields = json.loads(str(data["grid_json"]))
        # rho/sigma_stat/noise_scale are init=False derived fields; drop before rebuild.
        cfg = OUCoreConfig(
            **{k: v for k, v in cfg_fields.items()
               if k not in ("rho", "sigma_stat", "noise_scale")}
        )
        return DPResult(
            V=data["V"],
            pi=data["pi"],
            zg=data["zg"],
            fg=data["fg"],
            cfg=cfg,
            grid=DPGrid(**grid_fields),
        )


def config_hash(cfg: OUCoreConfig, grid: DPGrid) -> str:
    """Stable hash of the (core config, grid) pair, so a policy is never scored
    against a mismatched optimum (§3.3)."""
    payload = json.dumps(
        {"cfg": asdict(cfg), "grid": asdict(grid)}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Operator precompute + backward induction
# ---------------------------------------------------------------------------
def _build_operators(
    cfg: OUCoreConfig, grid: DPGrid
) -> tuple[np.ndarray, list[_SparseOp], np.ndarray, np.ndarray]:
    """Precompute the reward matrix R (A, M) and the per-action operators P_a."""
    zg = np.linspace(-grid.z_max, grid.z_max, grid.n_z)
    fg = np.linspace(0.0, 1.0, grid.n_f)
    n_z, n_f = grid.n_z, grid.n_f
    m = n_z * n_f
    n_a = cfg.n_actions

    eps_k, w_k = _gauss_hermite_normal(grid.n_quad)

    # dX and z' depend only on (z_i, k): dX_kz[k] shape (Nz,), zp_kz[k] shape (Nz,).
    dX_kz = np.stack([delta_logprice(zg, e, cfg) for e in eps_k])  # (K, Nz)
    zp_kz = np.stack([step_z(zg, e, cfg) for e in eps_k])          # (K, Nz)

    R = np.zeros((n_a, m), dtype=np.float64)
    ops: list[_SparseOp] = []
    src_rows = np.repeat(np.arange(m), 4)  # 4 bilinear corners per source state

    for a in range(n_a):
        f_prime, cost = apply_trade(fg, a, cfg)  # each shape (Nf,)
        f_prime = np.asarray(f_prime, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)

        r_acc = np.zeros((n_z, n_f), dtype=np.float64)
        rows_list, cols_list, vals_list = [], [], []

        for k in range(grid.n_quad):
            dX = dX_kz[k]                       # (Nz,)
            # Holding return and reward (cost handled once, below).
            exp_dX = np.exp(dX)[:, None]        # (Nz, 1)
            holding = np.log((1.0 - f_prime)[None, :] + f_prime[None, :] * exp_dX)
            if cfg.reward_clip:
                # Clip the *full* per-step reward; fold cost in for the clip, then
                # remove it (cost added back uniformly after the k-loop).
                full = np.log1p(-cost)[None, :] + holding
                full = np.clip(full, -cfg.reward_clip_value, cfg.reward_clip_value)
                r_acc += w_k[k] * (full - np.log1p(-cost)[None, :])
            else:
                r_acc += w_k[k] * holding

            # Transition targets: z' depends on (i) only; f_next on (i, j).
            f_next = next_f(f_prime[None, :], dX[:, None])     # (Nz, Nf)
            z_next = np.broadcast_to(zp_kz[k][:, None], (n_z, n_f))

            cols, weights = _bilinear_weights(
                z_next.reshape(-1), f_next.reshape(-1), zg, fg
            )  # (4, M)
            rows_list.append(src_rows)
            cols_list.append(cols.T.reshape(-1))
            vals_list.append((weights * w_k[k]).T.reshape(-1))

        R[a] = (r_acc + np.log1p(-cost)[None, :]).reshape(-1)
        ops.append(
            _SparseOp(
                np.concatenate(rows_list),
                np.concatenate(cols_list),
                np.concatenate(vals_list),
                m,
            )
        )

    return R, ops, zg, fg


def one_step_q(res: DPResult, t: int, z: float, f: float) -> np.ndarray:
    """Q_t(z, f, ·) at an arbitrary (off-grid) state via one-step lookahead:
    quadrature over the shock with the bootstrap V_{t+1} bilinearly interpolated.
    Reuses the same pure-core reward/transition functions as the solver, so it is a
    faithful (interpolation-limited) probe for the no-profitable-deviation check.
    """
    cfg = res.cfg
    eps_k, w_k = _gauss_hermite_normal(res.grid.n_quad)
    q = np.zeros(cfg.n_actions, dtype=np.float64)
    for a in range(cfg.n_actions):
        f_prime, cost = apply_trade(f, a, cfg)
        f_prime = float(f_prime)
        cost = float(cost)
        acc = 0.0
        for k in range(res.grid.n_quad):
            dX = float(delta_logprice(z, eps_k[k], cfg))
            r = float(reward(f_prime, cost, dX, cfg))
            f_nx = float(next_f(f_prime, dX))
            z_nx = float(step_z(z, eps_k[k], cfg))
            acc += w_k[k] * (r + cfg.gamma * res.value_at(t + 1, z_nx, f_nx))
        q[a] = acc
    return q


def solve(cfg: OUCoreConfig | None = None, grid: DPGrid | None = None) -> DPResult:
    """Run backward induction and return the value/policy tables."""
    cfg = cfg or OUCoreConfig()
    grid = grid or DPGrid()

    R, ops, zg, fg = _build_operators(cfg, grid)
    m = grid.n_z * grid.n_f
    n_a = cfg.n_actions
    t_max = cfg.t_max
    gamma = cfg.gamma

    V = np.zeros((t_max + 1, m), dtype=np.float64)
    pi = np.zeros((t_max, m), dtype=np.int16)

    for t in range(t_max - 1, -1, -1):
        v_next = V[t + 1]
        q = np.empty((n_a, m), dtype=np.float64)
        for a in range(n_a):
            q[a] = R[a] + gamma * ops[a].matvec(v_next)
        V[t] = q.max(axis=0)
        pi[t] = q.argmax(axis=0).astype(np.int16)

    return DPResult(V=V, pi=pi, zg=zg, fg=fg, cfg=cfg, grid=grid)
