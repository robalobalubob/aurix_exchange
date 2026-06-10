# Phase 1 Results — π\* Structure, Price-of-X, First Regret Number

**Status:** The first real artifact of the project (phase1_plan §8.9). Produced by the
DP solver (`src/solvers/dp.py`), the regret harness (`src/solvers/regret.py`), the
ablation driver (`src/solvers/ablations.py`), and the clean-core trainer
(`src/training/train_core.py`). Canonical instance: single-asset Byrinium
(θ=0.10, σ=0.40, σ_stat≈0.894, μ=ln 150), T=200, γ=1, unclipped, proportional fee φ=0.01.
Grid: n_z=201 on [−4.5, 4.5], n_f=101, 16-node Gauss–Hermite. Config hash `c7e2adda5358b757`.

---

## 1. The ground truth is trustworthy

Two independent oracles agree, and the optimum reproduces itself in simulation:

- **Monte-Carlo self-consistency:** rolling π\* in `OUTradingEnv` yields mean
  ln(W_T/W_0) within Monte-Carlo CI of E_z[V\*_0(z,0)] = **6.6385**. This single test
  pins the solver's transition/cost/reward against the env (single-source-of-truth).
- **MPC cross-check:** a fully independent receding-horizon backward induction (its own
  bilinear interpolation, direct per-sweep quadrature — no shared sparse operator)
  matches the DP value to **0.006** under common random numbers.
- **Quadrature/grid convergence:** 16→32 nodes moves E_z[V\*_0] by 7×10⁻⁵; grid
  refinement moves it < 5×10⁻³.

So E_z[V\*_0] ≈ 6.64 nats ≈ **765× expected wealth growth** over 200 steps is real, not a
discretization artifact — the consequence of a σ_stat≈0.89 mean-reverting asset (price
ranging ~[0.4×, 2.4×] of median) with only 1% round-trip friction and an oracle that
knows the dynamics.

---

## 2. Structure of π\* — the no-trade band and the ratchet

Optimal action by (z, f) at t=0 (z = normalized price deviation, f = fraction of wealth
in the asset). Ranges are the z-intervals over which each action is optimal:

| f | SELL region (z) | HOLD band (z) | BUY region (z) |
|---|---|---|---|
| 0.0 | — | [+0.58, +4.50] | [−4.50, +0.54] |
| 0.2 | [+0.81, +4.50] | [+0.31, +0.76] | [−4.50, +0.27] |
| 0.4 | [+0.67, +4.50] | [+0.04, +0.63] | [−4.50, +0.00] |
| 0.6 | [+0.50, +4.50] | [−0.23, +0.45] | [−4.50, −0.27] |
| 0.8 | [+0.31, +4.50] | [−0.45, +0.27] | [−4.50, −0.50] |
| 1.0 | [+0.13, +4.50] | [−4.50, +0.09] | — |

Three things to read off it:

1. **A Davis–Norman no-trade band.** There is a HOLD region between a BUY region (asset
   underpriced, z low) and a SELL region (overpriced, z high). Inside the band the 1%
   fee makes the best move no move.
2. **The band tilts with inventory.** The BUY boundary slides from z≈+0.54 at f=0 down
   to z≈−0.50 at f=0.8: the more you already hold, the more underpriced the asset must be
   before adding. The HOLD band tracks the optimal allocation, which rises as z falls.
3. **The asymmetric ratchet.** BUY is optimal at f as high as **0.99** — the agent climbs
   in 25%-of-cash increments — while SELL always liquidates fully to f=0. So f ratchets
   up gradually and collapses to zero instantly, exactly the structure predicted for the
   asymmetric action set (phase1_plan D4).

**End-effects are negligible.** Action shares are essentially identical at t=0, t=190,
t=199 (HOLD 7.9% → 9.3%; BUY ~48%; SELL ~44%). The horizon only perturbs the last few
reversion half-lives (ln2/θ ≈ 7 steps), so the time-aware optimum is, in practice,
stationary — which is why the price of stationarity (below) is ~0.

---

## 3. Price-of-X — three former debates, now measured

Each number is the regret (optimality gap vs the canonical objective) of a policy that is
optimal under a *distorted* objective. Paired = CRN difference against π\* (the clean,
low-variance number); 5000 episodes.

| Distortion | Paired regret (nats) | Decision agreement with π\* |
|---|---|---|
| Reward clipping (±1) | −0.001 ± 0.004 | 98.9% |
| Discounting (γ=0.99) | −0.000 ± 0.002 | 99.8% |
| Stationarity (freeze t=0 slice) | +0.001 ± 0.002 | 99.9% |

All three are **statistically zero** for this regime. Interpretation:

- **Clipping is free** because |Δln W| > 1 in a single step is a ~1% tail event, so the
  clip almost never binds and the optimal policy barely changes. This validates D6
  (the learner drops clipping; Huber + grad-norm clipping supply the stability).
- **Discounting is nearly free** because the decisions are dominated by near-term mean
  reversion (half-life ≈ 7 steps), far inside the γ=0.99 discount window.
- **Stationarity is nearly free** because end-effects reach only a few half-lives — the
  §2 observation, quantified.

These are regime-specific (Byrinium, slow reversion). Herbs/Tools sweeps and the
per-unit-impact axis (Phase 3) are where they may diverge.

---

## 4. First regret number — DQN on the clean core

The existing Double-DQN/PER/dueling stack, retrained on `OUTradingEnv` (obs [z, f, t/T],
3 actions, γ=1, terminated-at-T, unclipped, seeded PER), scored on the canonical
objective over 5000 CRN episodes:

| Metric | Value |
|---|---|
| E_z[V\*_0] (optimum) | 6.6385 |
| DQN mean ln(W_T/W_0) | 6.3375 ± 0.0434 |
| **Regret vs V\*** | **0.301** |
| Paired regret (CRN) | 0.288 ± 0.028 |
| Decision agreement | 70.1% |
| Median growth | 558.8× |
| Ruin rate / loss rate | 0.0% / 0.0% |

The DQN captures **~95.5%** of optimal log-growth after 150k steps. Two observations:

- **Save-best earned its keep.** The eval trajectory is noisy (regret bounced
  0.28 → 1.51 between steps 130k and 150k). `last.pt` would have reported ~1.5; `best.pt`
  holds the step-130k policy at 0.28. This is the §1 fix working as intended.
- **70% agreement, 0.30 regret** is the expected signature of a value-based agent on a
  problem with fine value distinctions between nearby allocations — it disagrees with π\*
  mostly on low-stakes states (cf. the wide HOLD band). Policy-gradient / continuous
  control (Phase 2) is the natural next rung.

---

## 5. Phase 1 exit criteria

- [x] D1–D8 documented and reflected in `OUCoreConfig` defaults.
- [x] Single source of truth: DP, MPC, learner all import `ou_core` pure functions.
- [x] DP solver passes the §4 checklist; blessed-V regression test in place.
- [x] MPC oracle within CI of the DP value.
- [x] CRN regret harness operational for arbitrary policies.
- [x] PER seeded; save-best on greedy seeded eval; deterministic evaluation.
- [x] DQN retrained on the core with a reported regret number + CI (**0.301**, paired 0.288 ± 0.028).
- [x] Price-of-clipping / -discounting / -stationarity reported (all ≈ 0).
- [x] π\* structure writeup committed (this document).

**Reproduce:**
```
.venv/Scripts/python.exe -m pytest src/tests/          # full validation suite
.venv/Scripts/python.exe -m src.solvers.ablations      # price-of-X numbers
.venv/Scripts/python.exe -m src.training.train_core    # first regret number
```
