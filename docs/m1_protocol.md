# M1 protocol: replication across training seeds

**Status:** orchestration and aggregation infrastructure is implemented and verified;
the empirical milestone is not complete until all 20 predeclared runs have finished
and their held-out results have been summarized.

M0 made a single training run reproducible. M1 asks the next scientific question:
does the conclusion survive retraining?

An RL training run is a stochastic optimization process. A different random seed can
change the network initialization, exploratory actions, sampled market histories,
policy samples, minibatch order, and replay-buffer samples. Two runs with identical
hyperparameters can therefore learn meaningfully different policies. Reporting only
the best-looking run would confuse a lucky optimization path with a dependable
algorithm.

## Scope and locked experiment matrix

M1 applies only to the four learners on the exact `OUTradingEnv` benchmark:

| Learner | Training entry point | Predeclared training seeds |
|---|---|---|
| Dueling Double DQN + PER | `src.training.train_core` | 0, 1, 2, 3, 4 |
| REINFORCE | `src.training.train_reinforce` | 0, 1, 2, 3, 4 |
| A2C | `src.training.train_a2c` | 0, 1, 2, 3, 4 |
| PPO | `src.training.train_ppo` | 0, 1, 2, 3, 4 |

This is a `4 algorithms x 5 seeds = 20 policies` experiment. The game-facing
`AurixExchangeEnv` DQN is deliberately excluded: it has a different observation and
action contract, reward, boundary semantics, and evaluation standard. Combining its
scores with the exact benchmark would not produce a meaningful comparison.

The default model, environment, optimizer, and training-budget settings are fixed for
this replication. M1 is not a hyperparameter search. If a defect forces a protocol
change, record it explicitly and restart the affected matrix rather than quietly
mixing incompatible runs.

## What counts as a replicate

**One independently trained policy is one replicate.** In this protocol, that means
one complete trainer run at one of the five predeclared training seeds.

Evaluating one checkpoint on 5,000 market paths does not create 5,000 independent
training replicates. Those paths measure how a *fixed* policy behaves under different
market shocks. Likewise, reevaluating the same checkpoint or copying its files does
not increase the training-seed sample size.

The same five seed labels are used for each algorithm to keep the plan simple, but
seed 2 for PPO is not a formal matched pair with seed 2 for DQN. The algorithms consume
random numbers differently and perform very different numbers of interactions. The
actual common-random-number pairing happens in evaluation, where policies face the
same held-out market paths.

This distinction creates two nested sources of uncertainty:

1. **Rollout or path uncertainty** asks how returns vary across stochastic market
   paths after the policy has been fixed. The common-random-number regret harness
   estimates this inner uncertainty with thousands of paired learner/oracle paths.
2. **Training-seed variation** asks how the learned policy changes when stochastic
   training is repeated from scratch. The five independently trained policies per
   algorithm estimate this outer variation.

A narrow rollout confidence interval can coexist with large training-seed variation.
It can tell us very precisely that one particular policy performs well, without
telling us whether the training algorithm reliably finds such a policy.

## Validation and held-out testing

Every replicate keeps model selection separate from final measurement:

- The validation seed block starts at **10,000**. Periodic validation regret on this
  block chooses `best.pt`. DQN uses 1,000 validation episodes and each policy-gradient
  trainer uses 500, preserving their locked defaults.
- The held-out test seed block starts at **100,000**. All four trainers use 5,000 test
  episodes once, after training, to evaluate the selected `best.pt` checkpoint.

All algorithms and training seeds use those same starts; the policy-gradient
validation paths are the first 500 paths in DQN's 1,000-path block, and the held-out
test block matches exactly. That common-random-number design makes comparisons less
noisy because policies face matched market shocks. It does not make the trained
policies dependent on one another.

The policy-gradient trainers also reserve these literal episode-reset seed intervals
during training: validation is the half-open interval `[10,000, 10,500)`, and testing
is `[100,000, 105,000)`. Their collectors normally draw training paths uniformly from
a much larger integer seed domain. If a draw lands in either reserved interval, the
collector rejects it and draws again. The probability is small for one draw, but the
15 full policy-gradient runs collectively draw 432,000 training episode seeds; without
the guard, about 1.1 collisions would be expected. A held-out contract should be
guaranteed rather than merely likely. REINFORCE, A2C, and PPO record the default-on
`exclude_evaluation_seed_blocks_from_training = true` rule in each run's serialized
training configuration. Turning it off is an explicit protocol deviation and does
not produce canonical M1 evidence.

The held-out test results must not be used to choose a checkpoint, tune a
hyperparameter, decide which seed to retain, or restart an otherwise valid run. Once
test feedback changes a modeling decision, that block has effectively become
validation data and a new test block must be declared for the revised experiment.

Each held-out test record is bound to the exact selected `best.pt` bytes by SHA-256.
Aggregation recomputes that digest, checks that the checkpoint named by the test is
the manifest's best checkpoint, and verifies that its final `selected_as_best`
validation record precedes the test record. This guards against accidentally moving,
replacing, or reporting a different checkpoint after evaluation.

The seed-0 numbers in `phase1_results.md` and `phase2_results.md` remain useful
historical results, but they were produced under the earlier protocol. In particular,
their final evaluation used the old seed convention. They are not silently promoted
to M1 replicates. Seed 0 is retrained and evaluated under this locked M1 protocol just
like seeds 1 through 4.

## Required report

For each algorithm, the M1 report must show:

- all five raw held-out test regrets, identified by training seed and run directory;
- the median regret and the full minimum-to-maximum range;
- the arithmetic mean regret and sample standard deviation, using denominator
  `n - 1`;
- the number of planned, completed, failed, and otherwise incomplete runs;
- any valid training collapse or unusually poor seed, without removing it from the
  aggregate.

The headline value remains test regret, `V* - E[return]`, so lower is better. Paired
regret and its common-random-number interval remain useful path-level diagnostics, as
does decision agreement, but validation metrics must never be substituted for test
metrics in the across-seed table. Test episodes from several policies are not pooled
as though they came from one very large training run.

Raw values matter because five observations are too few for a summary statistic to
show the shape of the distribution. The median and range give an intuitive robust
center and worst-to-best spread. The mean and sample standard deviation provide a
conventional moment summary that will remain comparable if the seed count grows.
Using `n - 1` rather than `n` in the sample variance acknowledges that five runs
estimate a wider population of possible training runs rather than exhaust it.

At `n = 5`, these are descriptive statistics, not a license for algorithm rankings or
statistical-significance claims. The report may say that one observed median is lower
than another, or that a method showed more seed sensitivity in this matrix. It should
not claim that one algorithm is generally superior from five observed training
replicates per algorithm.

### Failures and collapses stay visible

There are two importantly different bad outcomes:

- An **infrastructure failure** means the intended experiment did not finish, for
  example because the process was interrupted or an artifact could not be written.
  Preserve the failed run record and, after fixing the infrastructure, rerun the same
  predeclared seed with a new run ID.
- A **valid training collapse** means the trainer completed correctly but learned a
  poor policy. That result is evidence about algorithm reliability. It remains in the
  raw table and aggregate; it is not replaced with a more favorable seed.

The summary must state how many of the 20 runs are complete. Partial output can be
inspected operationally, but the empirical M1 milestone is not complete and no final
four-algorithm comparison should be claimed until the full matrix is accounted for.

## Workload and execution policy

The default matrix is intentionally substantial. Counting training interactions plus
learner and DP-oracle validation/test rollouts, its approximate number of environment
`step()` calls is:

| Learner | Per training seed | Five seeds |
|---|---:|---:|
| DQN | 5.35 million | 26.75 million |
| REINFORCE | 12.18 million | 60.90 million |
| A2C | 12.18 million | 60.90 million |
| PPO | 5.22 million | 26.10 million |
| **Total** | **34.93 million** | **174.65 million** |

These figures count calls, not equivalent learning updates. REINFORCE and A2C collect
`1,200 x 32 x 200 = 7.68 million` training transitions per seed, while PPO uses
sample reuse and collects `300 x 32 x 200 = 1.92 million`. DQN collects only 150,000
training transitions, but all four methods also perform periodic validation and a
5,000-episode held-out comparison against the DP policy. The DP grid solve itself is
cached and is not counted as an environment rollout.

Because this is a CPU workload, launching all 20 jobs at once could make every job
slower, exhaust memory, obscure failures, and make interruption recovery harder. The
M1 runner therefore plans the full matrix explicitly and executes it sequentially.
Explicit execution is a safety feature: inspecting or summarizing experiments should
never accidentally begin days of training. Unique M0 run directories make sequential
work resumable without overwriting completed evidence.

Only one runner may own a study at a time. An atomic study lock records its process,
host, start time, and command, preventing two `--resume` processes from launching the
same policy. Every attempt also tees its combined output to a durable per-study log,
so a traceback survives after the terminal closes. Relevant-source hashes and
cleanliness are attested immediately before and after every child process. Aggregation
errors stop execution and remain in the experiment history instead of leaving a stale
report behind.

### Runner commands

Inspect the predeclared default matrix without starting training:

```powershell
python -m src.training.multiseed plan
```

Execute the full matrix sequentially under an explicit study ID, then build its
aggregate report:

```powershell
python -m src.training.multiseed run --experiment-id m1_ou_ladder_v1
python -m src.training.multiseed summarize --experiment-id m1_ou_ladder_v1
```

Summarization writes `results.json` for machine-readable provenance and statistics,
`results.csv` for a tidy raw-run table, and `report.md` for human review beside the
study plan. Keeping all three views derived from the same manifests avoids hand-copy
errors between a spreadsheet and the scientific record.

If that study is interrupted, resume the same recorded plan instead of creating a
different matrix:

```powershell
python -m src.training.multiseed run --experiment-id m1_ou_ladder_v1 --resume
```

The plan and run commands accept `--algorithms`, `--seeds`, `--artifact-root`, and
`--study-root` for explicit operational subsets and locations. `--profile smoke`
reduces budgets to verify plumbing; its output is not M1 evidence. The default
`--profile full` is the locked experiment described here.

A full run refuses to begin from dirty relevant source by default. That guard keeps a
20-run study from quietly combining policies trained by different uncommitted code.
`--allow-dirty-source` exists for an intentional diagnostic run, but such a study must
be labeled as development evidence rather than the clean M1 result.

## Completion checklist

Infrastructure:

- [x] The multiseed plan enumerates exactly four OU algorithms and seeds 0-4.
- [x] Every run records its training seed, validation block, test block, source state,
  configuration, metrics, and checkpoints in its M0 artifact directory.
- [x] Planning is non-executing, full execution is explicit and sequential, and an
  interrupted study can resume without replacing completed trials.
- [x] Aggregation derives JSON, CSV, and Markdown from run manifests while keeping
  incomplete and failed trials visible.
- [x] Scientific aggregation validates the declared protocol, source attestations,
  selected-checkpoint hash, and validation-before-test ordering.
- [x] A per-study lock prevents duplicate runners, and every attempt has a durable
  combined stdout/stderr log.
- [x] A reduced one-seed/all-four-algorithm subprocess study verifies the complete
  orchestration and reporting path.

Empirical evidence:

- [ ] All 20 intended full-budget runs are complete or any failures are explicitly
  accounted for.
- [ ] The report contains raw per-seed held-out regrets and the four required summary
  statistics for every algorithm.
- [ ] Training collapses remain in the evidence.
- [ ] Conclusions distinguish rollout uncertainty from training-seed variation.
- [ ] No significance or general algorithm-ranking claim is made from `n = 5`.

Completing the runner and summary tooling completes the **M1 infrastructure**. Filling
the checklist with 20 full default-budget runs completes the **M1 empirical evidence**.
Those are deliberately separate claims.
