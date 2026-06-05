"""Greedy-policy evaluation harness for the Aurix Exchange agent.

Summed episode reward is a poor quality signal (it mixes wealth growth with
penalty terms), so this harness reports the metric that actually matters: the
distribution of terminal net worth, plus the bankruptcy rate. Run after training
to judge a checkpoint, and to compare reward-function changes before/after.

Usage:
    python -m src.training.evaluate
    python -m src.training.evaluate --checkpoint exports/dqn_checkpoint.pt --episodes 500
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from src.env.aurix_env import AurixExchangeEnv, EnvConfig, N_ACTIONS, OBS_DIM
from src.models.dqn import DuelingDQN

# Matches the training mask fill so greedy selection ignores invalid actions.
_MASK_FILL = -1.0e9
# Terminal net worth below this counts as ruin (catastrophic failure zeroes assets).
_RUIN_THRESHOLD = 1.0


def _greedy_action(net: DuelingDQN, obs: np.ndarray, mask: np.ndarray) -> int:
    """Masked argmax over Q-values — the deployment-time policy."""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        q = net(obs_t).masked_fill(~mask_t, _MASK_FILL)
        return int(torch.argmax(q, dim=1).item())


def evaluate(
    checkpoint_path: str,
    n_episodes: int = 500,
    seed0: int = 10_000,
    config: EnvConfig | None = None,
) -> dict:
    """Run the greedy policy over ``n_episodes`` seeded episodes and collect metrics.

    Returns a dict of summary statistics; does not print. Each episode uses a
    distinct seed so the result reflects the policy across many market draws.
    """
    cfg = config or EnvConfig()
    net = DuelingDQN(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()

    final_net_worth = np.empty(n_episodes, dtype=np.float64)
    episode_reward = np.empty(n_episodes, dtype=np.float64)
    action_counts: Counter[int] = Counter()

    for ep in range(n_episodes):
        seed = seed0 + ep
        env = AurixExchangeEnv(config=cfg, seed=seed)
        obs, info = env.reset(seed=seed)
        mask = info["action_mask"]
        total_reward = 0.0
        while True:
            action = _greedy_action(net, obs, mask)
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            total_reward += reward
            if terminated or truncated:
                break
        final_net_worth[ep] = info["net_worth"]
        episode_reward[ep] = total_reward

    start = cfg.initial_cash
    return {
        "n_episodes": n_episodes,
        "start_net_worth": start,
        "nw_mean": float(np.mean(final_net_worth)),
        "nw_median": float(np.median(final_net_worth)),
        "nw_p10": float(np.percentile(final_net_worth, 10)),
        "nw_p90": float(np.percentile(final_net_worth, 90)),
        "median_growth_x": float(np.median(final_net_worth) / start),
        "bankruptcy_rate": float(np.mean(final_net_worth < _RUIN_THRESHOLD)),
        "loss_rate": float(np.mean(final_net_worth < start)),
        "reward_mean": float(np.mean(episode_reward)),
        "action_counts": dict(action_counts.most_common()),
    }


def _print_report(m: dict) -> None:
    """Render the metrics dict as a readable console block."""
    print(f"episodes              {m['n_episodes']}")
    print(f"start net worth       {m['start_net_worth']:>12,.0f}")
    print(f"final net worth mean  {m['nw_mean']:>12,.0f}")
    print(f"final net worth median{m['nw_median']:>12,.0f}  ({m['median_growth_x']:.2f}x)")
    print(f"  P10 / P90           {m['nw_p10']:>12,.0f} / {m['nw_p90']:,.0f}")
    print(f"bankruptcy rate       {m['bankruptcy_rate']:>11.1%}")
    print(f"loss rate (< start)   {m['loss_rate']:>11.1%}")
    print(f"mean episode reward   {m['reward_mean']:>12.3f}")
    print(f"action counts         {m['action_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Aurix agent.")
    parser.add_argument("--checkpoint", default="exports/dqn_checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed0", type=int, default=10_000)
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, n_episodes=args.episodes, seed0=args.seed0)
    _print_report(metrics)


if __name__ == "__main__":
    main()
