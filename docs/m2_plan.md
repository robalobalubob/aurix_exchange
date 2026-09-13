# M2 Plan — Stationary Prototype Economic Integrity

**Status:** Active. The candidate stationary contract and its development baseline
are complete and awaiting full-suite validation (2026-07-17).

M2 prepares `AurixExchangeEnv` to be a trustworthy game-facing learning problem
before any advanced logistics simulation is added. The agent optimizes the mechanics
that exist in code, including loopholes. Economic transitions therefore need the same
kind of explicit contract and regression protection that M0/M1 gave experiments.

## Scope and non-goals

M2 keeps the stationary Phase A interface fixed:

- Observation: ten `float32` values in the existing order.
- Actions: HOLD, three BUY/SELL pairs, and LAUNCH_EXPEDITION.
- Prices: exogenous stationary geometric OU processes.
- Boundary: a continuing task exposed through 200-step truncation windows.

M2 does not modify the exact `OUTradingEnv` benchmark, reuse M1 test feedback for a
new exact-OU claim, create the M3 node/logistics simulator, add recurrence, export
ONNX, or start the Godot client.

## Seed and evidence boundary

- Development/baseline paths: seeds `[10000, 10500)`.
- Final game-policy test paths: seeds `[100000, 100500)`.
- The final block remains sealed while M2 mechanics and baselines are being repaired.
- A future trained game study will use a new identity such as `m2_game_v1` and
  predeclared independent training seeds. The exact-OU multiseed runner is not valid
  for this different environment contract.

## Baseline before the first repair

On the 500 development paths, before changing trade execution:

| Policy | Survival | Median terminal net worth |
|---|---:|---:|
| Uniform random over valid actions | 100% | 11,719 |
| Mean-reversion heuristic | 100% | 24,559 |

The old result was not a sound economic baseline. BUY remained valid at full
capacity but filled zero units, and raw-unit linear SELL impact made a full Byrinium
or Herbs liquidation worth almost nothing. Because the three commodities use very
different unit scales, the same coefficient did not represent the same market depth.

## M2 stationary contract v1

`EnvConfig.stationary_contract_version` is `m2_stationary_v1`; exported sidecars use
schema version 2.

### Valid actions must preserve their meaning

- BUY allocates exactly 25% of current cash.
- A BUY is valid only when that entire quoted fill is meaningful and fits in the
  selected commodity's remaining capacity. It is never silently clamped to a partial
  fill.
- SELL liquidates exactly 100% of the selected inventory.
- Trades below `min_trade_notional` are masked so dust actions cannot substitute for
  HOLD merely to avoid its penalty.
- Capacity checks are commodity-specific.

### Capacity-normalized integrated impact

Let `K_i` be commodity capacity, `q` the order quantity, `x = q/K_i`, `S_i` spot
price, `kappa = impact_log_at_capacity`, and `phi` the brokerage fee. Marginal depth
prices are symmetric in log space:

```text
p_buy(u)  = S_i exp(+kappa u)
p_sell(u) = S_i exp(-kappa u),  0 <= u <= x
```

Integrating across the complete block order gives:

```text
buy_cost(q) = (1 + phi) S_i K_i expm1(kappa x) / kappa
sell_cash(q) = (1 - phi) S_i K_i [-expm1(-kappa x)] / kappa
```

At `kappa = 0`, these reduce continuously to spot price plus/minus brokerage. The
default `kappa = 0.10` means the marginal log-price displacement reaches 0.10 only at
an entire warehouse of depth; the default fee remains 1%.

This design is positive, finite, and monotonic over every legal inventory. Equal
fractions of different commodity capacities receive equal slippage, and an immediate
round trip is strictly loss-making.

## What the trade repair revealed

Using the identical 500 development paths:

| Policy | Survival | Median terminal net worth |
|---|---:|---:|
| Uniform random over valid actions | 100% | 51,786 |
| Mean-reversion heuristic | 100% | 300,027 |

These numbers are not directly comparable as policy improvements because the MDP
changed. Their increase confirms that the old liquidation cliff was economically
dominant. It also exposed two distinct economic effects that the old bug had hidden:

- The stationary price process contains real, predictable mean-reversion profit.
- With the old 100-unit fee, repeated expeditions were effectively free wealth.

The first effect is intentional in the current learning task. The second conflicted
with the design goal that expeditions be risky, high-friction sourcing.

## Expedition calibration

The M2 defaults use `launch_fee_base = 800.0` and
`expedition_min_cash = 1600.0`. The fee is paid in full on success or localized
failure and scales with pre-launch fatigue. The liquidity reserve covers the largest
fee possible below the fatigue eligibility ceiling (1560 at the defaults).

These values follow an explicit equilibrium-price criterion rather than a desired
terminal-wealth result. Including the post-launch hazard probability, a rested
expedition has expected inventory value about 883 and expected net value about +83.
At pre-launch fatigue 25, the expected inventory value falls to about 792 while the
fee rises to 1000, for expected net value about -208. Rest therefore has economic
meaning: a rested launch is tempting, but a moderately fatigued launch is not a
guaranteed-profit action.

## Candidate frozen development baseline

The reproducible diagnostic is written to
`exports/studies/m2_stationary_v1/baseline_dev.json` (generated and ignored). It uses
all 500 development paths `[10000, 10500)` and environment-config SHA-256
`f58152cc3c2e23caf16eab34e5e5ca33d9486f01d2a52686905aa5c2bb85768c`.

| Policy | Median | P10 | P90 | Loss rate | Expedition rate | Survival |
|---|---:|---:|---:|---:|---:|---:|
| HOLD only | 10,000 | 10,000 | 10,000 | 0.0% | 0.0% | 100% |
| Uniform random over valid actions | 24,710 | 4,379 | 79,829 | 24.0% | 11.3% | 100% |
| Mean-reversion heuristic | 300,027 | 235,672 | 379,363 | 0.2% | 0.0% | 100% |
| Rested-expedition comparator | 10,452 | 5,047 | 27,935 | 47.0% | 5.5% | 100% |

Here, “loss” means terminal net worth below the initial 10,000; it is intentionally
different from ruin. The expedition comparator's broad distribution and 47% loss
rate show that sourcing is no longer guaranteed wealth, while its slightly positive
median preserves a reason to explore it. The heuristic's much larger result comes
from trading the exogenous mean-reverting prices, not expeditions.

## Remaining M2 sequence

1. **Trade integrity:** complete. Masks, exact fills, ledger accounting, sidecar
   versioning, and deterministic regressions have focused coverage.
2. **Economic diagnostics:** complete on development seeds. Action frequencies,
   expedition use, loss/ruin rates, and terminal distributions are recorded above.
3. **Freeze the environment:** run the full fast and slow validation suites and an
   RL-specific review, then promote this candidate baseline to frozen.
4. **Predeclare game training:** choose independent DQN training seeds and validation
   selection rules under `m2_game_v1` without opening the final test block.
5. **Train and select:** compare independent trained policies with the frozen
   heuristic on validation paths.
6. **Evaluate once:** open `[100000, 100500)` for the selected policies and report all
   seeds. The design target is median terminal net worth at least 1.8 times the
   heuristic, with no hidden bankruptcy or action-degeneracy regression.

## M2 exit criteria

- No mask-valid non-HOLD action is a zero-effect loophole.
- All legal trades preserve cash/inventory bounds and exact action semantics.
- Liquidation proceeds are finite, positive, monotonic, and capacity-scale invariant.
- Seeded baseline diagnostics are reproducible and documented.
- The environment contract is frozen before DQN training begins.
- Final claims keep training-seed variation separate from evaluation-path variation.
