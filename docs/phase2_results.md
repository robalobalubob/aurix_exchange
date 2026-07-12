# Phase 2 Results — The Algorithm Ladder

**Status:** Policy-gradient rungs complete (REINFORCE, A2C, PPO — 2026-07-11).
Remaining Phase 2 menu: continuous control (SAC/DDPG), model-based (MCTS, Dyna)
with sample-efficiency curves.
Every rung runs on the *identical* clean core (canonical single-asset Byrinium
instance, config hash `c7e2adda5358b757`, E_z[V\*_0] = **6.6385**) and is scored by
the same CRN regret harness against the same DP ground truth as Phase 1
(`docs/phase1_results.md`). Implementations: `src/models/policy.py`,
`src/training/train_reinforce.py`, `src/training/train_a2c.py`,
`src/training/train_ppo.py`.

Shared design, held fixed across rungs so the ladder isolates the algorithm:

- **Network capacity:** 128-128 ReLU trunk, matching `DuelingDQN`.
- **Collection:** full fixed-horizon episodes, 32 envs in lockstep, one batched
  forward per timestep (`collect_batch` — a correctness test replays its recorded
  actions through `OUTradingEnv` and requires exact reward agreement).
- **Objective:** canonical (gamma=1, unclipped, terminated-at-T). Terminal
  bootstrap is exactly zero, so GAE needs no off-the-end value estimate.
- **Evaluation:** deterministic mode-action (argmax of logits) through the regret
  harness; save-best keyed on 500-episode CRN regret every 50 updates; final
  numbers from the best checkpoint on 5000 CRN episodes (seed block 10_000+).

---

## 1. The ladder so far

| Rung | Regret vs V\* | Paired (CRN) | Agreement | Median growth | Env steps |
|---|---|---|---|---|---|
| π\* (DP oracle) | 0 | 0 | 100% | 765× | — |
| DQN/PER/dueling (Phase 1) | 0.301 | 0.288 ± 0.028 | 70.1% | 559× | 150k |
| REINFORCE | **0.0787** | 0.0658 ± 0.0173 | 67.6% | 729× | 7.68M |
| A2C | 0.1047 | 0.0918 ± 0.0181 | 44.7% | 694× | 7.68M |
| PPO | 0.1026 | 0.0897 ± 0.0180 | 45.3% | 703× | **1.92M** |

Log-growth captured: DQN **95.5%**; every policy-gradient rung ≥ **98.4%**
(REINFORCE 98.8%). The three PG methods are statistically close to one another
(overlapping paired CIs) and all far below the DQN's 0.301 — on this problem the
*family* mattered (policy-based vs value-based), not the refinement within it.
Where the refinements show up instead: A2C's critic doubled convergence *speed*,
and PPO matched the others' quality on **a quarter** of the env interactions.

---

## 2. REINFORCE — first policy-gradient rung

Batch-episode Monte-Carlo policy gradient; critic-free variance reduction only
(per-timestep batch-mean baseline + global advantage normalization), entropy
bonus 0.01, Adam lr 3e-4, 1200 updates × 32 episodes.

Final report (best checkpoint, 5000 CRN episodes):

| Metric | Value |
|---|---|
| E_z[V\*_0] (optimum) | 6.6385 |
| mean ln(W_T/W_0) | 6.5598 ± 0.0428 |
| **Regret vs V\*** | **0.0787** |
| Paired regret (CRN) | 0.0658 ± 0.0173 |
| Decision agreement | 67.6% |
| Median growth | 728.61× |
| P10 growth | 98.04× |
| Ruin rate / loss rate | 0.0% / 0.0% |

Observations:

1. **The Phase 1 hypothesis held.** phase1_results §4 predicted policy-gradient
   methods as "the natural next rung" because the DQN's 0.30-regret/70%-agreement
   signature pointed at fine value distinctions between nearby allocations.
   REINFORCE — the *crudest* policy-gradient method — beats the DQN's regret
   by ~4× (0.301 → 0.0787). Optimizing the policy directly sidesteps exactly
   the failure mode the value-based rung exhibited.
2. **Agreement decouples from regret, in both directions.** Mid-training the
   greedy policy hit regret 0.109 at only **43%** agreement — lower regret than
   the DQN achieved at 70% agreement. By update 1200 agreement drifted up to
   ~68% while regret kept falling. Confirms agreement is a diagnostic, not an
   objective (phase1_plan §5): most disagreements live in the wide HOLD band
   where actions are near-value-equivalent.
3. **Save-best still earns its keep.** The eval trajectory is noisy (0.25 →
   0.48 → 0.25 → 0.12 → 0.21 → 0.10 over the run); the greedy mode of a
   stochastic policy jumps around even while the underlying distribution
   improves smoothly. Policy entropy stayed ≈ 0.7–0.77 nats (max ln 3 ≈ 1.10) —
   the sampled policy keeps exploring; scoring extracts the mode.
4. **Determinism check passed.** A 600-update run and the first 600 updates of
   the 1200-update run (same seed) produced bit-identical eval trajectories —
   the seeded collection/update/eval pipeline is fully reproducible.
5. **The sample-efficiency ordering inverts the regret ordering.** REINFORCE
   consumed ~51× the DQN's env steps (7.68M vs 150k). On a cheap simulator
   that trade is free; the model-based rungs (Phase 2 menu) are where the
   sample-efficiency axis gets its own study.

---

## 3. A2C — bootstrapped critic

Same collector and budget as REINFORCE; the change is GAE(λ=0.95) advantages
from a learned value head (`ActorCriticNet`, shared 128-128 trunk) instead of
Monte-Carlo returns against a batch-mean baseline. Value coefficient 0.5.

Final report (best checkpoint, 5000 CRN episodes): regret **0.1047**, paired
0.0918 ± 0.0181, agreement 44.7%, median growth 694×, ruin/loss 0%.

Observations:

1. **The critic bought speed, not a lower floor.** A2C reached regret ≈ 0.10 by
   update 600 — REINFORCE needed ~1050 to get there — but its final number is
   statistically indistinguishable from REINFORCE's (paired CIs overlap), and
   the point estimate is slightly worse. On a problem this small, Monte-Carlo
   variance over 32 full episodes is apparently cheap enough that the critic's
   bias buys little.
2. **Critic warm-up is a real instability.** At update 100 the greedy policy
   briefly collapsed to regret 2.27 (agreement 16%) while the value head was
   still mis-calibrated and steering the advantages. It recovered by update
   150. Save-best rode through the excursion untouched.
3. **Entropy fell faster** (≈ 0.53 vs REINFORCE's ≈ 0.75 at comparable regret):
   critic-shaped gradients sharpen the policy sooner. The greedy/stochastic gap
   is correspondingly smaller.

## 4. PPO — clipped surrogate, sample reuse

Same net and GAE machinery as A2C; each batch is consumed for 4 epochs of
minibatched clipped-surrogate updates (clip ε=0.2, minibatch 1600 of 6400).
300 collection updates = 1.92M env steps, one quarter of the A2C/REINFORCE
budget at the same gradient-step count.

Final report (best checkpoint, 5000 CRN episodes): regret **0.1026**, paired
0.0897 ± 0.0180, agreement 45.3%, median growth 703×, ruin/loss 0%.

Observations:

1. **Sample reuse worked as advertised.** PPO reached the same quality band as
   A2C (paired CIs overlap almost exactly) on ¼ of the env interactions, and
   with none of A2C's warm-up excursion — the eval trajectory descends nearly
   monotonically from 0.24 to ≈ 0.10.
2. **The trust region barely binds.** Clip fraction stayed ≈ 1–3% throughout:
   at lr 3e-4 on a smooth low-dimensional problem, four epochs of reuse do not
   push the policy near the clip boundary. The clip is insurance, and here the
   premium was nearly free.
3. **The 0.08–0.10 plateau looks like a family-level floor** at this budget and
   architecture. All three PG rungs converge into it while disagreeing with π\*
   on ~55% of decisions — more evidence that the wide HOLD band hosts a large
   set of near-equivalent policies, and closing the last ~0.1 nat likely needs
   either far longer training, a different action parameterization (continuous
   sizing, Phase 3), or model-based search (Phase 2 menu).

---

## Reproduce

All ladder entries are the module defaults (seed 0), so each command below
reproduces its reported number exactly:

```
.venv/Scripts/python.exe -m pytest src/tests/test_reinforce.py src/tests/test_a2c.py src/tests/test_ppo.py
.venv/Scripts/python.exe -m src.training.train_reinforce   # regret 0.0787
.venv/Scripts/python.exe -m src.training.train_a2c         # regret 0.1047
.venv/Scripts/python.exe -m src.training.train_ppo         # regret 0.1026
```
