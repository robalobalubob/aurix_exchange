# Phase 1 Execution Plan — Clean OU Core, DP Ground Truth, Regret Harness

**Status:** Normative for Phase 1. Companion to `aurix_rl_roadmap_v2.md`.
**Purpose:** Finish/update Phase 1: lock the open decisions, specify the clean trading
core, give the full DP-solver blueprint with its validation checklist, define the
regret harness, list the trainer validity fixes, and set the build order, module
layout, test plan, and exit criteria.

---

## 0. Locked decisions (defaults adopted; revisit only with cause)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Canonical objective | Finite-horizon **E[ln(W_T/W_0)]**, T=200, γ=1, **unclipped** | Matches what `evaluate.py` measures; Kelly/growth objective the reward already implies; backward induction gives π\*_t for free |
| D2 | Horizon semantics (clean core) | `t/T` in the observation; reaching T is **`terminated=True`** | With time in the state the horizon is a true MDP boundary; bootstrap correctly zeroed. Deliberate sign-flip vs the legacy env's (also correct) truncation convention |
| D3 | Cost model (clean core) | **Proportional brokerage fee only** (φ=0.01); optional proportional-impact coefficient, default 0; **no inventory caps** | Required for log-utility scale-invariance → valid 2-D (z,f) state reduction. Per-unit impact and caps return as a Phase 3 axis |
| D4 | Action set | Primary: existing asymmetric {HOLD, BUY 25% of cash, SELL 100% of inventory}. Secondary config: symmetric target-fraction {0, ¼, ½, ¾, 1} | Primary is the only way to regret-score the *existing* DQN. Redesign stays out of Phase 1 |
| D5 | Canonical parameters | Single asset, Byrinium: θ=0.10, σ=0.40 (σ_stat≈0.894), μ=ln(150) | Slow-reversion regime is the interesting one; Herbs/Tools become sweeps |
| D6 | Reward clipping in learner | Removed | Huber loss + grad-norm clipping already provide the stability; clipped objective solved as a twin DP study instead |
| D7 | Discounting in learner | γ=1 with termination at T (time-aware). γ=0.99-no-time kept as an ablation | Aligns learner objective with DP and evaluator |
| D8 | Regret start distribution | z ~ N(0,1), f=0, t=0 | Matches the env's actual reset law (stationary price draw, zero inventory) |

---

## 1. Repository fixes carried into Phase 1

These are correctness/validity issues in the current repo, verified against code:

1. **Seed the PER buffer.** `PrioritizedReplayBuffer.sample` uses global
   `np.random.uniform`. Add an `rng: np.random.Generator` constructed from a seed
   parameter; thread the seed from `TrainConfig`. Without this, "reproducible" runs
   with identical `cfg.seed` diverge.
2. **Save-best, not save-last.** `best_avg_return` in `train_dqn.py` is tracked but
   never used. Replace the noisy last-20-training-episodes signal with a **periodic
   greedy, seeded evaluation** (reuse the `evaluate.py` machinery on a fixed seed
   block); checkpoint on improvement; keep a `last.pt` alongside `best.pt`.
3. **Deterministic evaluation.** Greedy policy, fixed seed block, fixed episode count;
   no global-RNG dependence anywhere in the eval path.
4. *(Deferred, tracked)* `spec.md` §4's hardcoded C# normalizer is stale post-rebalance
   (inventories ÷100; wrong μ/σ for B and T). `future.md` §3 is the correct template.
   Phase 5 item; recorded here so it isn't lost.

---

## 2. The clean trading core (`src/env/ou_core.py`)

### 2.1 Mathematical specification

Work in standardized coordinates. Let ρ = e^(−θ) and σ_stat = σ/√(2θ). The normalized
price deviation z = (X − μ)/σ_stat evolves as an **exact unit-variance AR(1)**:

    z' = ρ·z + √(1 − ρ²)·ε,    ε ~ N(0,1)

(the existing normalization makes this exact — keep it). The log-price increment is

    ΔX = σ_stat·[(ρ − 1)·z + √(1 − ρ²)·ε]

State: (z, f, t) where f ∈ [0,1] is the fraction of net worth in the asset.

Trade map (action → post-trade fraction f′ and proportional cost c):

- HOLD: f′ = f, c = 0
- BUY (primary set): trade amount b·(1−f) of wealth with b = 0.25
  → f′ = f + b·(1−f)·(1−φ)/(1 − b·(1−f)·φ) *(implement as: cash → asset at fee φ on
  the traded notional; derive f′ from the post-fee ledger — single source of truth is
  the ledger arithmetic, not a closed-form shortcut)*
- SELL (primary set): liquidate all, f′ = 0, fee φ on the notional f·W
- Target-fraction set (secondary): f′ = target; fee φ·|f′ − f| of wealth

Per-step reward (the canonical, unclipped objective):

    r = ln(1 − c) + ln((1 − f′) + f′·e^ΔX)

Wealth-fraction update:

    f_next = f′·e^ΔX / ((1 − f′) + f′·e^ΔX)

Everything depends on (z, f, action) only — the scale-invariant reduction, valid
because of D3.

### 2.2 Module contract

- `OUCoreConfig` (frozen dataclass): θ, σ, μ, fee φ, buy_fraction b, action-set
  variant, T, reward-clip toggle (default off), time-feature toggle (default on),
  discount γ (default 1.0). Defaults = the canonical instance (D5).
- Pure functions (no hidden state, NumPy-vectorizable): `step_z(z, eps)`,
  `delta_logprice(z, eps)`, `apply_trade(f, action) -> (f_prime, cost)`,
  `reward(f_prime, cost, dX)`, `next_f(f_prime, dX)`.
- Thin Gymnasium adapter `OUTradingEnv` over the pure functions: obs `[z, f, t/T]`
  (float32), `terminated=True` at t=T (D2), seeded reset sampling z ~ N(0,1), f=0.
- **Design rule:** the DP solver, the MPC oracle, and the learners all import these
  same pure functions. Any duplication of transition/cost/reward logic is a bug class
  that silently invalidates regret.

---

## 3. DP solver blueprint (`src/solvers/dp.py`)

### 3.1 Grid and quadrature

- z-grid: uniform on [−4.5, 4.5], ~200 points. Stationary mass beyond ±4.5 is
  ~7×10⁻⁶; clamp quadrature targets at the edges (V is near-linear there).
- f-grid: uniform on [0, 1], ~100–200 points.
- Expectation over ε by **Gauss–Hermite quadrature**, 16 nodes (validate vs 32) —
  near machine precision for these smooth integrands.
- **Linear interpolation only** (bilinear on the (z,f) grid). V is a max of smooth
  functions and has kinks along action-region boundaries; cubic overshoots there.

### 3.2 Operator precompute (the key implementation move)

For each action a and grid point (z_i, f_j): the post-trade f′, the cost, the K
quadrature targets (z′_k, f_next,k), and the expected one-step reward
R_a(i,j) = ln(1−c) + Σ_k w_k·ln((1−f′) + f′·e^{ΔX_k}) are all **time-invariant**.
Precompute each action's expectation as a **sparse linear operator** P_a (bilinear
interpolation weights of the K targets into the grid, weighted by quadrature weights)
plus the reward vector R_a.

Backward induction is then T iterations of |A| sparse matvecs and a pointwise max:

    Q_t(s, a) = R_a(s) + (P_a V_{t+1})(s),    V_t = max_a Q_t,    π_t = argmax_a Q_t

Boundary: V_T ≡ 0 (the objective's terminal value is already accumulated through the
per-step log-returns).

**Budget check:** 200×100 states × 3 actions × 16 nodes × 200 steps ≈ 10⁸
multiply-adds — seconds to a minute in vectorized NumPy/SciPy sparse. No cleverness
required; spend the cleverness budget on validation instead.

### 3.3 Variants produced by the same solver (cheap twins)

1. **Canonical:** unclipped, γ=1, T=200 → V\*, π\*_t. The ground truth.
2. **Clipped twin:** identical except per-step reward clipped to ±1 → "price of
   clipping" when both policies are evaluated under the canonical objective.
3. **Discounted stationary twin:** γ=0.99, long horizon (e.g. T=2000; the policy
   converges away from the boundary since end-effects reach only a few half-lives,
   ln2/θ ≈ 7 steps) → "price of discounting" and "price of stationarity."
4. **Secondary action set:** target-fraction variant, one config change.

Persist for each run: V[t,·,·], π[t,·,·], the full config, and a config hash; load by
hash in the regret harness so a policy is never scored against a mismatched optimum.

### 3.4 MPC second oracle (`src/solvers/mpc.py`)

Receding-horizon MPC using the same pure-math core: at each state, enumerate action
sequences (or do sparse lookahead with the known transition + quadrature) over a
horizon H ≈ 3–5 half-lives, pick the first action of the best sequence. It is a
near-optimal independent policy. **Acceptance:** MPC's CRN-evaluated mean ln(W_T/W_0)
within Monte-Carlo CI of E_z[V\*_0(z,0)] minus a small, documented gap. Two oracles
agreeing is what makes either trustworthy.

---

## 4. Validation checklist (what makes the ground truth *trustworthy*)

All items become `correctness`-marked tests where feasible; the blessed V becomes a
`regression` baseline.

1. **Quadrature convergence:** doubling nodes (16→32) changes V below tolerance.
2. **Grid convergence:** refining z and f grids converges V and, separately, the
   *policy boundaries* (band edges move less than one cell).
3. **Monte-Carlo self-consistency (the big one):** roll π\* in `OUTradingEnv`; mean
   ln(W_T/W_0) must match E_z[V\*_0(z,0)] within CI. This single test catches most
   transition/reward mismatches between solver and env.
4. **No profitable one-step deviation:** at randomly sampled (z, f, t), every non-π\*
   action's Q is ≤ π\*'s Q (up to interpolation tolerance).
5. **Limit behaviors:** φ→0 collapses the no-trade band to the myopic sign rule;
   θ→0 with costs makes never-trading optimal.
6. **Qualitative structure:** no-trade band around the optimal fraction widening with
   φ (Davis–Norman shape); for the asymmetric primary action set, the expected
   **ratchet** structure (incremental climbs in f, instant drops to 0).
7. **MPC cross-check** (§3.4).

---

## 5. Regret harness (`src/solvers/regret.py`)

- **Definition:** Regret(π) = E_z[V\*_0(z, 0)] − Ê[ln(W_T/W_0) under π], start
  distribution per D8, estimated by N seeded rollouts.
- **Common random numbers, explicit:** all policies in a comparison consume the same
  per-episode seed block (formalizing the accidental shared `seed0=10_000` in
  `evaluate.py`/`run_baselines.py`). Large variance reduction on regret *differences*.
- **Policy protocol:** anything mapping obs → action (DP table lookup with
  interpolation, MPC, DQN checkpoint, heuristic, random). One interface, one harness.
- **Report:** regret point estimate + CI; per-episode paired differences vs π\*;
  decision-agreement rate with π\* (diagnostic only — policies can disagree on states
  carrying no value); tail stats (P10, ruin rate) to stay compatible with the existing
  eval report style.

---

## 6. Learner alignment (retraining the DQN on the core)

- Train the existing DQN/PER/dueling stack on `OUTradingEnv`: obs `[z, f, t/T]`
  (OBS_DIM=3 for the core — keep the 10-d game env untouched in its own module), 3
  actions, no action mask needed in the clean core (or a trivial all-true mask to keep
  the code path identical).
- γ=1, terminated-at-T (D2/D7); reward unclipped (D6); PER buffer seeded; save-best
  via periodic greedy seeded eval (§1).
- Ablations (each one flag): clipped reward; γ=0.99 without the time feature.
- **Output:** the first regret number, plus the three price-of-X numbers from §3.3.

---

## 7. Module layout & test plan

```
src/env/ou_core.py          # pure math core + OUCoreConfig + OUTradingEnv adapter
src/solvers/dp.py           # operator precompute + backward induction + persistence
src/solvers/mpc.py          # receding-horizon second oracle
src/solvers/regret.py       # policy protocol + CRN rollouts + regret report
src/training/train_core.py  # DQN training on the clean core (γ=1, no clip)
src/tests/test_ou_core.py   # correctness: trade map, reward, AR(1) exactness,
                            #   scale-invariance (wealth never enters), terminated@T
src/tests/test_dp.py        # correctness: checklist §4 items 1,2,4,5;
                            #   self-consistency §4.3 as a slow/marked test;
                            #   regression: blessed V baseline (locked like the
                            #   existing regression markers, updated only deliberately)
src/tests/test_regret.py    # CRN determinism; π* regret ≈ 0 within CI; random-policy
                            #   regret > 0 sanity
```

The existing game env (`aurix_env.py`) and its test suite remain untouched; the clean
core lives beside it, not inside it.

---

## 8. Build order

1. **Lock decisions** (D1–D8) — done by this document.
2. `ou_core.py` + `test_ou_core.py` (pure functions first, adapter second).
3. `dp.py` + validation tests §4.1–.5; bless V → regression baseline.
4. `mpc.py` + cross-check §4.7.
5. `regret.py` + CRN harness + its tests.
6. Trainer validity fixes (§1.1–.3) in the shared training code.
7. `train_core.py`: retrain DQN on the core; first regret number.
8. Twin solves + ablations: price of clipping / discounting / stationarity.
9. **Writeup:** π\* structure (no-trade band geometry, ratchet effect, parameter
   sweeps across Herbs/Tools regimes) — the first real artifact of the project.

## 9. Exit criteria (restated, testable)

- [ ] D1–D8 documented (this file) and reflected in `OUCoreConfig` defaults.
- [ ] Single source of truth: DP, MPC, and learner all import the same core functions.
- [ ] DP solver passes the full §4 checklist; blessed-V regression test in place.
- [ ] MPC oracle within CI of the DP value.
- [ ] CRN regret harness operational for arbitrary policies.
- [ ] PER seeded; save-best on greedy seeded eval; deterministic evaluation.
- [ ] DQN retrained on the core with a reported regret number + CI.
- [ ] Price-of-clipping, price-of-discounting, price-of-stationarity reported.
- [ ] π\* structure writeup committed.
