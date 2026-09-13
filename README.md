# Aurix Exchange

Aurix Exchange is a reinforcement-learning and economic-simulation learning project.
Its long-term destination is a systemic market game, but its current strength is a
controlled research benchmark where an RL agent can be measured against a computed
optimal policy rather than only against hand-written heuristics.

That distinction drives the repository design: the two RL environments and the
detailed economic simulation are independent tracks with different contracts.

## Detailed economic simulation

`src.economy.Economy` now provides a runnable systemic economy with six populated
nodes, thirteen commodities, fourteen production recipes, finite local markets,
warehouses, fleets, delayed shipments, seasonal labor, research dependencies,
strikes, audits, trade bans, intelligence, and competing enterprises. Currency and
goods are accounted for across markets, private property, and cargo in transit.

```powershell
# Run a full four-season model year.
python -m src.economy --days 120 --seed 7

# Operate an enterprise interactively.
python -m src.economy --interactive --seed 7

# Save a run and its diagnostic report.
python -m src.economy --days 30 --save exports/economy/world.json --report exports/economy/report.json
```

See [`docs/economy_guide.md`](docs/economy_guide.md) for commands, economic rules,
modeling assumptions, and deterministic save/resume. This is a command-driven
Python simulation. Its rivals are heuristic policies; a new Gymnasium adapter,
learned policies for this world, ONNX export, and the Godot client remain separate
integration work.

## The two RL tracks

| Track | Exact OU benchmark | Stationary game prototype |
|---|---|---|
| Environment | `OUTradingEnv` | `AurixExchangeEnv` |
| Purpose | Controlled RL experiments with ground truth | Explore the eventual game economy |
| Observation | `[price deviation, invested fraction, time]` (3 values by default) | Cash, inventories, prices, fatigue, phase, cooldown (10 values) |
| Actions | HOLD, BUY, SELL (3) | HOLD, BUY/SELL x 3 commodities, expedition (8) |
| Objective | Finite-horizon, undiscounted log growth | Discounted shaped reward over continuing 200-step windows |
| Best comparison | Regret against dynamic programming | Seeded heuristics and economic diagnostics |

The exact benchmark is intentionally small. Its log price follows an exactly
discretized Ornstein-Uhlenbeck (OU) process, which mean-reverts toward a fixed
long-run level. Because transaction costs are proportional and wealth scale does not
affect the decision, the state reduces to price deviation `z`, invested fraction `f`,
and time. That makes backward dynamic programming practical.

This gives the project an unusually useful experimental control:

- DP computes the best achievable policy on the chosen grid.
- MPC independently checks the DP solution using a separate implementation.
- Common random numbers expose candidate policies to the same market shocks.
- **Regret** is optimal expected log growth minus the learner's expected log growth;
  lower is better, and zero is optimal within measurement/discretization error.

The game prototype is richer but not an exact benchmark. It models Byrinium, Herbs,
Tools, inventory, fatigue, expeditions, four daily phases, cooldowns, and validity
masks. It remains stationary and single-agent so those mechanics can be understood
before endogenous supply, logistics, competitors, and politics are introduced.

## Current status

### Implemented

- A shared pure-math OU core used by the environment, DP solver, MPC oracle, and
  learners, reducing the risk that the "ground truth" solves a different MDP.
- A finite-horizon DP optimum and independent MPC cross-check.
- A common-random-number regret harness and price-of-clipping, discounting, and
  stationarity ablations.
- Dueling Double DQN with prioritized experience replay (PER).
- REINFORCE, A2C, and PPO on the same exact benchmark.
- The ten-observation/eight-action stationary game prototype and heuristic baselines.
- M0 experiment protection: direct DQN/PER tests, unique run manifests and metrics,
  held-out final-test seeds, and fast/manual-oracle CI jobs.
- M1 multiseed replication: a locked four-algorithm/five-seed study with explicit
  sequential execution and resume behavior, durable attempt logs, checkpoint-bound
  held-out metrics, and fail-closed manifest-derived aggregate reports.
- A separate detailed economy with integrated production, logistics, local
  scarcity, political intervention, competitors, an interactive console, event
  journals, deterministic saves, and accounting/causal scenario tests.

Phase 1 and M1 are complete. M1 trained five independent policies for each of DQN,
REINFORCE, A2C, and PPO under a locked protocol; all 20 full-budget runs completed.
Mean held-out regret was `0.2527` for DQN, `0.0846` for REINFORCE, `0.0906` for A2C,
and `0.0828` for PPO. Every observed policy-gradient run beat every observed DQN run,
but five seeds and unequal training budgets do not support a general algorithm-family
ranking. See [`docs/m1_results.md`](docs/m1_results.md) for raw per-seed results,
variation, provenance, and interpretation, and
[`docs/m1_protocol.md`](docs/m1_protocol.md) for the locked design.

### Not implemented yet

- A Gymnasium adapter and learned policies for the detailed economic simulation.
- Recurrent or graph policies and learned multi-agent training. The detailed
  simulator already has endogenous markets, heuristic rivals, and hidden politics.
- A production ONNX exporter and Python/ONNX/C# parity suite.
- A Godot 4.NET project or production game client.

Documents containing C# samples and ONNX contracts describe deployment intent.
The implemented economy's behavior is documented in `docs/economy_guide.md`;
the older specifications are not its executable interface.

## Set up and verify

From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest src/tests -m "not slow"
```

Run the full suite, including slow numerical/oracle checks, before changing the MDP or
solver:

```powershell
python -m pytest src/tests
```

Useful entry points:

```powershell
# Stationary game-prototype baselines
python -m src.tests.run_baselines

# Exact-benchmark ablations and learners
python -m src.solvers.ablations
python -m src.training.train_core
python -m src.training.train_reinforce
python -m src.training.train_a2c
python -m src.training.train_ppo

# Game-prototype DQN
python -m src.training.train_dqn
```

Trainer defaults are full CPU experiments, not quick smoke tests. Check the relevant
configuration dataclass before launching one.

M1 adds an explicit, sequential multiseed workflow. Planning is read-only; `run` is a
separate command because the full default matrix is about 174.65 million environment
step calls:

```powershell
# Inspect four algorithms x seeds 0-4 without starting training
python -m src.training.multiseed plan

# Run, resume if interrupted, and summarize one named study
python -m src.training.multiseed run --experiment-id m1_ou_ladder_v1
python -m src.training.multiseed run --experiment-id m1_ou_ladder_v1 --resume
python -m src.training.multiseed summarize --experiment-id m1_ou_ladder_v1
```

`--profile smoke` verifies the workflow with reduced budgets but does not produce M1
evidence. Full studies also require clean relevant source by default so every policy
in the comparison is trained from the same recorded implementation.

## Experiment artifacts

Generated checkpoints and solver tables live under `exports/` and are intentionally
ignored by Git. M0 standardizes new training output as a self-describing run under:

```text
exports/runs/<algorithm>/seed_<n>/<run-id>/
|-- config.json       # Complete hyperparameters and environment configuration
|-- manifest.json     # Algorithm, schemas, seeds, source revision, and best metric
|-- metrics.jsonl     # Append-only training/evaluation history
|-- best.pt           # Best validation checkpoint
|-- last.pt           # Final training state
`-- aurix_config.json # Game-client sidecar (dqn_game runs only)
```

The exact run-id nesting may evolve, but the invariant is important: model weights
alone are not a reproducible experiment. A checkpoint must remain connected to its
configuration, observation/action schema, training and evaluation seeds, metric
history, and source revision.

Existing flat files such as `exports/core_best.pt` or `exports/ppo_last.pt` are legacy
local artifacts. They support the recorded Phase 1/2 results but should not be used as
the template for new runs. A deliberately blessed deployment bundle may eventually be
versioned separately; generated training outputs should not be force-added casually.

## Documentation authority

When documents disagree, use this order:

1. **Behavior:** executable code and correctness/regression tests are authoritative.
2. **Locked benchmark contract:** [`docs/phase1_plan.md`](docs/phase1_plan.md) defines
   the exact OU objective; [`docs/phase1_results.md`](docs/phase1_results.md) records
   completed Phase 1 evidence.
3. **Replication protocol:** [`docs/m1_protocol.md`](docs/m1_protocol.md) locks the
   M1 algorithms, training seeds, evaluation blocks, and reporting rules.
4. **Replicated benchmark evidence:** [`docs/m1_results.md`](docs/m1_results.md)
   records the completed 20-run M1 study and supersedes single-seed ladder claims.
5. **Historical ladder results:** [`docs/phase2_results.md`](docs/phase2_results.md)
   preserves the exploratory seed-0 policy-gradient experiments and limitations.
6. **Research direction:** [`docs/aurix_rl_roadmap_v2.md`](docs/aurix_rl_roadmap_v2.md)
   explains the benchmark-first strategy. Beyond its locked Phase 1 foundation, the
   roadmap is intentionally revisable as experiments teach us more.
7. **Game design intent:** [`docs/rebalancing.md`](docs/rebalancing.md) describes the
   current Phase A design direction and [`docs/vision.md`](docs/vision.md) the long-term
   product vision. [`docs/future.md`](docs/future.md) and [`docs/spec.md`](docs/spec.md)
   contain useful historical/planned interfaces but may lag the implementation.
8. **Detailed economy:** [`docs/economy_guide.md`](docs/economy_guide.md) documents
   the separate systemic simulation; [`docs/economy_implementation.md`](docs/economy_implementation.md)
   maps its completion requirements to implementation and validation evidence.

The active stationary-prototype repair protocol and its sealed evaluation boundary are
recorded in [`docs/m2_plan.md`](docs/m2_plan.md).

[`AGENTS.md`](AGENTS.md) contains repository-wide engineering constraints. A design
document never silently overrides tested runtime behavior; update the code, tests, and
relevant documentation together when intentionally changing a contract.

## Near-term roadmap and why it is ordered this way

1. **M0 -- protect experiments (complete).** Direct DQN/PER tests, separate
   validation/final-test seeds, and self-describing runs establish the evidence
   contract used by every later milestone.
2. **M1 -- replicate across training seeds (complete).** All 20 predeclared runs
   completed from one clean source state. The tracked result reports every held-out
   regret and across-run variation without inferring a population ranking from five
   seeds.
3. **M2 -- repair and baseline the stationary game prototype (active).** Fix economic loopholes
   such as zero-effect valid actions and problematic liquidation impact before adding
   complexity; an RL agent will exploit the implemented incentives, not the intended
   ones.
4. **M3 -- detailed simulation (implemented as a separate engine).** Typed nodes,
   routes, shipments, facilities, production, conservation checks, and an interactive
   console now exist under `src/economy`. The exact OU core remains the control.
5. **M4/M5 -- grow the policy with the problem.** First compare a flat MLP with shared
   entity/graph encoders in a fully observable world. Add GRU memory only when hidden
   information creates a genuine POMDP; delays represented by visible shipment
   manifests do not by themselves require recurrence.
6. **M6 -- learn in shared markets.** The detailed engine now includes endogenous
   markets and heuristic competitors. Centralized training, decentralized execution,
   MAPPO, and self-play remain future research work.
7. **M7 -- freeze and deploy.** Once observation, action, masking, and recurrent-state
   contracts are stable, add raw-output ONNX export, golden parity vectors, and then
   the Godot client. This avoids repeatedly rebuilding a production interface around a
   changing simulator.

The two tracks continue in parallel: the benchmark remains a regression/control
environment while the game simulation becomes richer. Results from one track must not
be presented as results for the other.
