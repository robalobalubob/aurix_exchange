"""Price-of-X ablations for the clean OU core (phase1_plan §3.3, roadmap_v2 §1.0).

Three former "decisions" become *measured* regret numbers, each the optimality gap of
a policy trained under a distorted objective when scored on the canonical one
(finite-horizon, undiscounted, unclipped E[ln(W_T/W_0)]):

  price of clipping       — optimum of the clipped-reward twin, scored unclipped.
  price of discounting    — optimum of the gamma=0.99 twin, scored undiscounted.
  price of stationarity   — the time-aware optimum's t=0 slice frozen and applied at
                            every step (i.e. ignoring the horizon), scored canonically.

All three are computed with the DP solver + regret harness; no learner is involved.

Usage:
    python -m src.solvers.ablations
"""
from __future__ import annotations

from dataclasses import dataclass

from src.env.ou_core import OUCoreConfig
from src.solvers import dp, regret


@dataclass
class PriceOfX:
    price_of_clipping: regret.RegretReport
    price_of_discounting: regret.RegretReport
    price_of_stationarity: regret.RegretReport


def compute(
    grid: dp.DPGrid | None = None,
    n_episodes: int = 5_000,
    seed0: int = 10_000,
    clip_value: float = 1.0,
    discount: float = 0.99,
) -> PriceOfX:
    grid = grid or dp.DPGrid()

    # Canonical ground truth (the yardstick every policy is scored against).
    canonical = dp.solve(OUCoreConfig(), grid)
    seeds = regret.make_seed_block(n_episodes, seed0)
    pistar_returns, _ = regret.rollout(regret.from_dp(canonical), canonical.cfg, seeds)

    def score(policy: regret.Policy) -> regret.RegretReport:
        return regret.evaluate_regret(
            policy, canonical, n_episodes=n_episodes, seed0=seed0,
            pistar_returns=pistar_returns,
        )

    # 1. Clipping: optimum of the clipped twin, scored on the canonical objective.
    clipped = dp.solve(
        OUCoreConfig(reward_clip=True, reward_clip_value=clip_value), grid
    )
    price_clip = score(regret.from_dp(clipped))

    # 2. Discounting: optimum of the gamma<1 twin, scored undiscounted.
    discounted = dp.solve(OUCoreConfig(gamma=discount), grid)
    price_disc = score(regret.from_dp(discounted))

    # 3. Stationarity: freeze the time-aware optimum's far-from-horizon (t=0) slice and
    #    apply it at every step, discarding the horizon information.
    def stationary_policy(obs, t):
        return canonical.policy_at(0, float(obs[0]), float(obs[1]))

    price_stat = score(stationary_policy)

    return PriceOfX(price_clip, price_disc, price_stat)


def main() -> None:
    out = compute()
    print("=== Price-of-X ablations (regret vs canonical optimum) ===")
    print(f"config hash {dp.config_hash(OUCoreConfig(), dp.DPGrid())}\n")
    for name, rep in [
        ("price of clipping", out.price_of_clipping),
        ("price of discounting", out.price_of_discounting),
        ("price of stationarity", out.price_of_stationarity),
    ]:
        print(f"--- {name} ---")
        print(f"regret {rep.regret:.4f}  | paired {rep.paired_regret:.4f} "
              f"+/- {rep.paired_ci:.4f}  | agreement {rep.agreement_rate:.1%}")
        print()


if __name__ == "__main__":
    main()
