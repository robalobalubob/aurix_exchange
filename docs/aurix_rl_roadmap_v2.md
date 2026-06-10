# Aurix Exchange — RL & Math-Simulation Roadmap (v2)

**Status:** Supersedes `aurix_rl_roadmap.md` (v1). Phase 1 is the only fixed commitment;
everything beyond it remains an explicitly fluid menu. v2 incorporates the repository
audit, closes two under-specified points in the v1 Phase 1 plan (objective definition
and scale-invariance of costs), and pulls MPC validation forward into Phase 1.

**Companion document:** `phase1_plan.md` — the concrete execution plan for Phase 1.

---

## Framing (unchanged in spirit, sharpened in letter)

The primary objective is **exploring reinforcement learning and mathematical
simulation**, with the game as an important but downstream goal.

The strategic insight: the market is a discretized Ornstein–Uhlenbeck (OU) process —
linear, Gaussian, mean-reverting, with a known exact transition density. That puts the
trading problem close to the analytically-solvable regime (Davis–Norman /
Shreve–Soner no-trade regions under proportional costs; Gârleanu–Pedersen in the
quadratic-cost LQG setting), which means a **ground-truth optimal policy is cheaply
computable**, and every learned agent can be scored by its **regret against that
optimum** rather than against an arbitrary heuristic bar.

What v2 adds to the framing: a ground truth is only a ground truth if the objective is
fully specified and the state reduction is actually valid. v1 left both partially open.
v2 locks them (see Phase 1.0 and the companion plan).

---

## Repository audit summary (verified against code, 2026-06)

Claims from v1 that were checked and confirmed:

- **PER buffer is unseeded.** `PrioritizedReplayBuffer.sample` draws via the global
  legacy `np.random.uniform`; identical `cfg.seed` runs diverge once learning starts.
- **`best_avg_return` is dead code.** Tracked and displayed in `train_dqn.py`, never
  used; the saved checkpoint is whatever the final step produced.
- **`spec.md` §4's hardcoded C# normalizer is stale post-rebalance** (all inventories
  ÷100; wrong μ/σ for B and T). `future.md` §3's sidecar-reading version is correct.
  Phase 5 concern.
- **The RL core is correctly implemented.** The Double DQN target masks invalid
  next-state actions before the online-net argmax; the OU step is the exact transition
  density (and tested against it); `terminated` vs `truncated` bootstrap semantics are
  correct for the current (infinite-horizon-style) formulation.

Correction to a v1 claim:

- v1 said the expedition/fatigue/cooldown subsystem "has no clean closed form." No
  *closed form*, true — but in isolation it is a small finite MDP
  (fatigue × cooldown × phase × a coarse cash axis) and is **also exactly solvable by
  tabular DP**. Not Phase 1 work, but it should not be written off as unbenchmarkable;
  a ground truth for it is available whenever it becomes interesting.

---

## Phase 1 — The Anchored Foundation (NOT subject to change)

Goal unchanged: a clean, characterized core with a computable optimum and trustworthy
experiments. v2 restructures Phase 1 into a decision layer (1.0) plus execution layers,
because two decisions must precede the solver or the "ground truth" is not well-posed.

### 1.0 Lock the canonical objective and the cost model (NEW — prerequisite)

**Gap A — the objective was under-specified along the horizon/discount axis, not just
the clip axis.** Three objectives currently coexist: the learner optimizes
infinite-horizon discounted return (γ=0.99, with no horizon feature in the obs and
bootstrapping through truncation); the evaluator scores finite-horizon terminal wealth
at T=200; the DP objective was unchosen. Regret is only meaningful once all three agree.

**Canonical objective (locked):** finite-horizon, undiscounted expected log-growth

> maximize E[ln(W_T / W_0)], T = 200, γ = 1, **unclipped** log-return reward.

Consequences:

- Backward induction yields the time-dependent optimum π\*_t directly; the stationary
  optimum falls out of the same solver run at long horizon (end-effects reach only a
  few reversion half-lives — ln2/θ ≈ 7 steps for Byrinium).
- In the clean core env, `t/T` enters the observation and hitting T is
  **`terminated=True`** (not truncated): once time is in the state, the horizon is a
  true MDP boundary and the bootstrap is correctly zeroed there. This is the deliberate
  sign-flip of the existing env's (also correct) convention.
- Three former "decisions" become **measured quantities** — the first three regret
  numbers: the price of clipping (twin clipped-objective DP solve), the price of
  discounting (γ=0.99 stationary solve evaluated under the canonical objective), and
  the price of stationarity (no-time-feature policy vs time-aware optimum).

Clip note: per-step log-price std for Byrinium is ≈0.38, so |Δln W| > 1 even at full
investment is a ~2.6σ (~1%) event — rare, but concentrated exactly in the tail moves
that drive growth. Hence: measure it, don't argue about it. The learner drops reward
clipping (Huber loss + grad-norm clipping already supply the stability it was buying).

**Gap B — the 2-D state reduction silently requires proportional transaction costs;
the current env violates that.** The (z, f) collapse rests on the scale-invariance of
log utility, which requires every per-step return term to be wealth-scale-free. The
action set qualifies (BUY trades b·(1−f) of wealth, SELL trades f — functions of f
alone). The cost model does not: `temp_impact · qty` is per-*unit* (a 10× richer agent
pays 10× the proportional slippage) and absolute inventory caps break invariance the
same way. **Clean-core cost model (locked): proportional brokerage fee only (1%);
optional proportional-impact coefficient defaulted to 0; no inventory caps.** Per-unit
impact is a genuinely interesting later axis (it is what makes wealth matter) but it is
fatal to Phase 1.

Scoping locks:

- **Arena:** the single-asset stripped env. The existing DQN must be **retrained on
  that core** to produce the first regret number; the 3-asset ground truth does not
  come free later (3 z's + 2 f's ≈ 5-D grid → near-oracle territory, see Phase 2/4).
- **Action set:** the existing asymmetric 3-action set (HOLD / BUY-25%-of-cash /
  SELL-100%) is **primary** — it is the only way to regret-score the existing agent;
  expect π\* to show a ratchet structure (slow climb in f, instant drop to 0). A
  symmetric target-fraction variant ({0, ¼, ½, ¾, 1}) ships as a secondary config with
  identical machinery. Action-space *redesign* stays in Phase 2/3 — no scope creep.
- **Canonical parameter instance:** Byrinium (θ=0.1, σ=0.4 — the interesting
  slow-reversion regime). Herbs/Tools are parameter sweeps through the same solver.

### 1.1 Strip to the clean trading core

Separate the pure OU-trading MDP from the expedition/fatigue/cooldown subsystem into
distinct modules. The trading core gets the ground truth; the expedition system is
parked (with the note above that it is tabular-DP-solvable if ever needed).

Design rule above all others: **the DP solver and the learners must consume the same
transition/cost/reward functions** (a single pure-math core module), or regret numbers
become fiction through silent drift.

### 1.2 Build the dynamic-programming ground truth

Backward induction on a (z, f) grid with Gauss–Hermite quadrature over the price shock,
exploiting the fact that the normalized deviation z is an exact unit-variance AR(1)
under the existing normalization. Full blueprint — coordinates, grid, operator
precompute, complexity budget — in `phase1_plan.md` §3.

### 1.3 Establish the optimality gap as the core metric

Regret(π) = E_z[V\*_0(z, 0)] − Ê[ln(W_T/W_0) under π], evaluated at the environment's
true start distribution (z ~ N(0,1), f = 0, t = 0) via rollouts using **common random
numbers** across policies (formalizing what `evaluate.py`/`run_baselines.py` already do
by accident via shared `seed0`). Decision-agreement with π\* is reported as a
diagnostic only. This replaces the "1.8× heuristic" acceptance bar everywhere.

### 1.4 Resolve the reward-clip subtlety

Resolved by 1.0: canonical objective is unclipped; the clipped objective is solved as a
twin study so the distortion is a *measured* regret number, not a debate.

### 1.5 Experimental-validity fixes

- Seed the PER buffer (own `np.random.default_rng(seed)`).
- Fully deterministic, greedy, seeded evaluation.
- **Save-best, not save-last** — keyed off periodic greedy seeded evaluation (the
  `evaluate.py` machinery), not the noisy last-20-episodes training window.

### 1.6 MPC as a second oracle (NEW — pulled forward from Phase 2)

Receding-horizon MPC with the known model is nearly free given closed-form dynamics and
provides an independent route to the optimum. **Two oracles agreeing within Monte-Carlo
error is what makes either trustworthy.** The remaining model-based ladder (MCTS, Dyna)
stays in Phase 2.

### Phase 1 exit criteria (v2)

- Canonical objective and cost model locked and documented (1.0). ✔ by this document.
- A clean single-asset OU-trading core module, consumed by both solver and learners.
- A validated DP solver producing V\* and π\* (validation checklist in
  `phase1_plan.md` §4), cross-checked against MPC.
- A CRN regret harness reporting regret for any policy.
- Reproducible, deterministic experiment runs (PER seeding, save-best, greedy eval).
- The DQN retrained on the clean core, with its **first regret number** reported,
  alongside the price-of-clipping / price-of-discounting / price-of-stationarity
  ablations.
- A short writeup of π\*'s structure (no-trade band geometry, ratchet effect of the
  asymmetric action set) — the first real artifact.

---

## Everything Beyond Phase 1 (explicitly subject to change)

Unchanged contract: the phases below are a *menu and a suggested ordering*. Simulation
base, model family, objective, and state representation all remain open. The OU base is
recommended for its known structure, not sacred.

---

## Phase 2 — The Algorithm Ladder (subject to change)

Implement a ladder of algorithm families on the *identical* core and benchmark each
against the DP optimum:

- **Tabular value iteration** — the ground truth itself (done in Phase 1).
- **Value-based deep** — the existing DQN/PER/dueling stack (first regret number lands
  in Phase 1).
- **Policy gradient** — REINFORCE → A2C → PPO; often suits trading better, since value
  distinctions between nearby allocations are fine.
- **Continuous control** — SAC / DDPG; trade *sizing* is naturally continuous, the
  25%/100% discretization is an artifact.
- **Model-based** — MPC graduates to Phase 1 validation (above); MCTS over the
  low-dimensional state and Dyna remain here. Model-based vs model-free
  sample-efficiency curves against the DP optimum is a self-contained study.

The architecture question remains deliberately un-pre-decided; this phase answers it
empirically.

---

## Phase 3 — Math Enrichment Axes (subject to change)

Each axis a controlled study with regret (or a re-derived optimum) as the readout:

- **Continuous action sizing** — remove the fixed-fraction discretization.
- **Per-unit impact / inventory caps** — reintroduce the scale-invariance breakers
  *deliberately*, now that Phase 1 has shown why they break the 2-D reduction; wealth
  re-enters the state and the problem becomes genuinely richer. (NEW as an explicit
  axis — it was an accident in v1's environment, it becomes an experiment in v2.)
- **Parameter uncertainty → dual control.** Hide θ, μ, σ; Gaussian-OU conjugacy keeps
  the posterior tractable. **Still flagged as the crown jewel** — and strengthened: for
  the single-asset case with unknown μ (known θ, σ), the Bayes-adaptive problem is
  itself a ~4-D grid DP, so an **exact Bayes-optimal benchmark** is available. Thompson
  sampling and Bayes-adaptive MDPs then get scored against a true optimum too.
- **Correlated multi-asset → cointegration → pairs trading.** Multivariate OU. Note:
  the ground truth here is a **near-oracle (MPC/MCTS), not exact DP** — the state is
  ~5-D and the grid dies.
- **Risk-sensitive objectives.** The canonical log-growth objective *is* Kelly;
  tail-aware objectives (CVaR) and constrained RL (Lagrangian CMDP with P(ruin) ≤ ε)
  replace the penalty-tuning treadmill.

---

## Phase 4 — The Endogenous / Equilibrium Frontier (subject to change)

Unchanged. `_gou_mu` is static (a correctness test enforces it); making μ respond to
inventory makes the environment non-stationary from the learner's view — an equilibrium
of a coupled policy↔price system, not a fixed MDP. Stepping stones unchanged:
exogenous-demand endogeneity first (learner still sees a roughly stationary world),
then self-play / mean-field equilibrium ("clone the .onnx across rival factions" is
self-play in product language). Entered only after the stationary core is fully
characterized.

---

## Phase 5 — The Game Layer (subject to change; last)

Unchanged: logistics lags → POMDP + delayed credit assignment (in-transit capital must
enter net worth; n-step/λ-returns; recurrent or manifest-structured state); opaque
politics → value of information (belief states, sensing actions, CMDP framing of
seizure risk); deployment scaffolding (ONNX export, Python↔C# parity test, sidecar —
and the known `spec.md` §4 normalizer bug fixed at that point, with `future.md` §3 as
the correct template).

---

## Suggested Critical Path (v2)

1. **Phase 1** in full, per `phase1_plan.md` — decisions are locked, build order is:
   core env + tests → DP solver + validation (incl. MPC cross-check) → CRN regret
   harness → trainer validity fixes → retrain DQN on the core → first regret number +
   the three price-of-X ablations → π\* structure writeup.
2. **Phase 2** algorithm ladder against the DP optimum.
3. **Phase 3** enrichment axes (parameter-uncertainty and risk highest-value;
   per-unit-impact axis now explicit).
4. **Phase 4** endogenous/equilibrium frontier.
5. **Phase 5** game layer, onto characterized foundations.

The first concrete artifact remains the DP solver for the single-asset known-OU core —
now with its objective fully specified, its state reduction made valid, and a second
oracle to keep it honest.
