# M1 Results — Multiseed Replication

**Status:** Complete (2026-07-17).

M1 asked whether the exact-OU benchmark conclusions survive retraining. Each of the
four frozen learner configurations was trained independently with seeds 0-4, selected
only on its validation block, and evaluated once on the common held-out test block.
One trained, validation-selected policy is one replicate.

## Study identity and integrity

- Experiment: `m1_ou_ladder_v1`
- Track/profile: `exact_ou_benchmark` / `full`
- Source commit: `3404e2b25a348929641ad100e189ce8a88d503d1`
- Source content hash: `1e1e97641b63a80efdb2d195fa9deab357a8051d4f522afe2ef8297c08ce4d18`
- DP configuration hash: `c7e2adda5358b757`
- Environment hash: `b3161d52f017048763fa834e70e74e7a3e6e68726cafb739d9a88c7e081ef45a`
- Validation seeds: `[10000, 10500)`
- Held-out test seeds: `[100000, 105000)`
- Completion: 20 planned, 20 completed, 20 included, zero failures or retries
- Runtime: 1 hour 44 minutes on CPU

All 20 attempts were made from the same clean source state. Their before/after source
hashes match, every child manifest completed, and every selected checkpoint is bound
to its held-out metrics by SHA-256.

## Held-out regret

Lower regret is better. The raw values are in training-seed order 0, 1, 2, 3, 4.
The standard deviation is the sample SD across independently trained policies, using
the `n - 1` denominator.

| Algorithm | Raw regrets, seeds 0-4 | Median [min, max] | Mean | Sample SD |
|---|---|---:|---:|---:|
| DQN | 0.290400, 0.220451, 0.256390, 0.253063, 0.243152 | 0.253063 [0.220451, 0.290400] | 0.252691 | 0.025329 |
| REINFORCE | 0.073050, 0.087222, 0.105198, 0.079559, 0.077722 | 0.079559 [0.073050, 0.105198] | 0.084550 | 0.012622 |
| A2C | 0.101066, 0.095166, 0.078327, 0.095964, 0.082443 | 0.095166 [0.078327, 0.101066] | 0.090593 | 0.009700 |
| PPO | 0.097445, 0.057358, 0.098467, 0.069537, 0.091329 | 0.091329 [0.057358, 0.098467] | 0.082827 | 0.018411 |

Every observed policy-gradient run had lower regret than every observed DQN run under
these frozen configurations. Among the policy-gradient methods, PPO had the lowest
mean and best individual result, REINFORCE had the lowest median, and A2C had the
smallest sample SD. Those different summaries do not justify declaring a general
winner from five training seeds.

The DQN policies had the highest average exact-action agreement with the DP policy
(61.3%, versus 46.9%-52.5% for the policy-gradient methods) despite having worse
regret. Agreement weights every action equally; regret captures the economic cost of
mistakes. Matching frequent low-consequence decisions can coexist with missing rare,
valuable trades.

All 100,000 held-out policy episodes had zero loss and ruin rates. Every learned
policy nevertheless retained a positive paired-regret gap larger than its rollout
95% half-width, so the benchmark is not solved. That statement concerns market-path
uncertainty for fixed policies, not population uncertainty across training runs.

## What this result does and does not establish

M1 establishes that the policy-gradient advantage seen in the exploratory seed-0
ladder was not confined to one initialization. It also demonstrates why multiseed
replication matters: PPO has the best mean while REINFORCE has the best median, and a
single seed can reverse their apparent order.

This is a comparison of frozen trainer configurations, not a proof that one algorithm
family is intrinsically superior or more sample-efficient. DQN used 150,000 training
interactions, PPO used 1.92 million, and REINFORCE/A2C each used 7.68 million. No
population confidence interval, p-value, or general ranking is inferred from `n = 5`.

The generated machine-readable evidence remains under
`exports/studies/m1_ou_ladder_v1/` locally. It is ignored by Git by design; this file
is the durable tracked summary.

## Boundary for future exact-OU experiments

The test block `[100000, 105000)` was valid for this predeclared M1 claim, but its
outcomes have now been observed. Any future exact-OU design or tuning decision informed
by M1 must predeclare a fresh held-out test block and exclude it from training. The M1
block may be reused only for auditing or exact frozen-protocol reproduction, not as a
new unbiased final test.
