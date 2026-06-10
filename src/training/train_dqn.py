from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.env.aurix_env import AurixExchangeEnv, N_ACTIONS, OBS_DIM, export_config
from src.models.dqn import DuelingDQN
from src.training.evaluate import evaluate_net
from src.training.replay_buffer import PrioritizedReplayBuffer

# Large negative fill for masked-out action Q-values prior to argmax / max.
_MASK_FILL = -1.0e9


@dataclass(frozen=True)
class TrainConfig:
    # Schedule
    total_steps: int = 200_000
    warmup_steps: int = 2_000
    train_every: int = 1
    target_sync: int = 1_000

    # Optimisation
    batch_size: int = 64
    lr: float = 5.0e-4
    gamma: float = 0.99
    grad_clip: float = 10.0

    # Replay / PER
    buffer_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # Bookkeeping
    log_every: int = 1_000
    # ``checkpoint_path`` is the always-saved last checkpoint; ``best_path`` is saved
    # only when the periodic greedy seeded eval improves (save-best, not save-last).
    checkpoint_path: str = "exports/dqn_last.pt"
    best_path: str = "exports/dqn_best.pt"
    config_path: str = "exports/aurix_config.json"
    seed: Optional[int] = 0

    # Periodic greedy, seeded evaluation that drives save-best (deterministic).
    eval_every: int = 10_000
    eval_episodes: int = 200
    eval_seed0: int = 10_000


def _linear_anneal(start: float, end: float, frac: float) -> float:
    frac = min(max(frac, 0.0), 1.0)
    return start + (end - start) * frac


def _masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a copy of q with invalid actions filled to a large negative."""
    return q.masked_fill(~mask, _MASK_FILL)


def select_action(
    net: DuelingDQN,
    obs: np.ndarray,
    mask: np.ndarray,
    eps: float,
    rng: np.random.Generator,
) -> int:
    """Masked epsilon-greedy action selection."""
    valid = np.flatnonzero(mask)
    if rng.random() < eps:
        return int(rng.choice(valid))

    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        q = _masked_q(net(obs_t), mask_t)
        return int(torch.argmax(q, dim=1).item())


def compute_loss(
    online: DuelingDQN,
    target: DuelingDQN,
    batch: dict,
    gamma: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Double DQN loss with action-masked targets. Returns (loss, td_errors)."""
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
    next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32)
    next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool)
    weights = torch.as_tensor(batch["weights"], dtype=torch.float32)

    # Q(s, a) for the actions actually taken
    q_taken = online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Online net selects the best *valid* next action (Double DQN)
        next_q_online = _masked_q(online(next_obs), next_masks)
        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
        # Target net evaluates that action
        next_q_target = target(next_obs).gather(1, next_actions).squeeze(1)
        target_value = rewards + gamma * next_q_target * (1.0 - dones)

    td_errors = target_value - q_taken
    # PER importance-sampling weighted Huber loss
    elementwise = nn.functional.smooth_l1_loss(
        q_taken, target_value, reduction="none"
    )
    loss = (weights * elementwise).mean()

    return loss, td_errors.detach().cpu().numpy()


def train(cfg: TrainConfig) -> DuelingDQN:
    # Independent child streams for action selection and the PER buffer, both derived
    # from cfg.seed so the whole run is reproducible (see replay_buffer seeding fix).
    act_seed, buf_seed = np.random.SeedSequence(cfg.seed).spawn(2)
    rng = np.random.default_rng(act_seed)
    torch.manual_seed(cfg.seed if cfg.seed is not None else 0)

    env = AurixExchangeEnv(seed=cfg.seed)
    online = DuelingDQN(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    target = DuelingDQN(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = PrioritizedReplayBuffer(
        cfg.buffer_capacity, N_ACTIONS, alpha=cfg.per_alpha,
        rng=np.random.default_rng(buf_seed),
    )

    obs, info = env.reset()
    mask = info["action_mask"]

    running_loss = 0.0
    loss_count = 0
    episode_return = 0.0
    episode_returns: list[float] = []
    # Save-best signal: median terminal net worth from the greedy seeded eval, not the
    # noisy last-20-training-episodes window (which was tracked but never acted on).
    best_eval_score = float("-inf")

    pbar = tqdm(
        range(1, cfg.total_steps + 1),
        desc="train",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    for step in pbar:
        eps = _linear_anneal(
            cfg.eps_start, cfg.eps_end, step / cfg.eps_decay_steps
        )
        action = select_action(online, obs, mask, eps, rng)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]
        episode_return += reward

        # Only `terminated` zeroes the bootstrap; `truncated` does not.
        buffer.add(obs, action, reward, next_obs, terminated, next_mask)

        obs, mask = next_obs, next_mask

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, info = env.reset()
            mask = info["action_mask"]

        # Learn
        if step >= cfg.warmup_steps and step % cfg.train_every == 0:
            beta = _linear_anneal(
                cfg.per_beta_start,
                cfg.per_beta_end,
                step / cfg.total_steps,
            )
            batch = buffer.sample(cfg.batch_size, beta)
            loss, td_errors = compute_loss(online, target, batch, cfg.gamma)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
            optimizer.step()

            buffer.update_priorities(batch["indices"], td_errors)

            running_loss += float(loss.item())
            loss_count += 1

        # Sync target network
        if step % cfg.target_sync == 0:
            target.load_state_dict(online.state_dict())

        # Periodic greedy seeded evaluation -> save-best. Done after warmup, when the
        # policy is worth evaluating.
        if step >= cfg.warmup_steps and step % cfg.eval_every == 0:
            metrics = evaluate_net(
                online,
                n_episodes=cfg.eval_episodes,
                seed0=cfg.eval_seed0,
                config=env.cfg,
            )
            online.train()  # evaluate_net() puts the net in eval mode
            score = metrics["nw_median"]
            if score > best_eval_score:
                best_eval_score = score
                _save_checkpoint(online, cfg.best_path)
            pbar.write(
                f"step {step:>7} | EVAL median_nw {score:>12,.0f} "
                f"({metrics['median_growth_x']:.2f}x) | "
                f"bankruptcy {metrics['bankruptcy_rate']:.1%} | "
                f"best_nw {best_eval_score:>12,.0f}"
            )

        # Log
        if step % cfg.log_every == 0:
            avg_loss = running_loss / max(loss_count, 1)
            recent = episode_returns[-20:]
            avg_ret = float(np.mean(recent)) if recent else float("nan")

            # Live metrics on the bar itself; full line written above it.
            pbar.set_postfix(
                eps=f"{eps:.3f}",
                loss=f"{avg_loss:.4f}",
                ret=f"{avg_ret:.2f}",
                best_nw=f"{best_eval_score:,.0f}",
                buf=len(buffer),
                ep=len(episode_returns),
                refresh=False,
            )
            pbar.write(
                f"step {step:>7} | eps {eps:5.3f} | "
                f"loss {avg_loss:8.5f} | avg_return {avg_ret:8.3f} | "
                f"buffer {len(buffer):>6} | episodes {len(episode_returns):>5}"
            )
            running_loss = 0.0
            loss_count = 0

    pbar.close()
    # Always save the last checkpoint; best.pt holds the best eval seen during training.
    _save_checkpoint(online, cfg.checkpoint_path)
    if best_eval_score == float("-inf"):
        # Training never reached an eval (e.g. total_steps < warmup); best == last.
        _save_checkpoint(online, cfg.best_path)
    export_config(env.cfg, cfg.config_path)
    print(f"saved last -> {cfg.checkpoint_path}, best -> {cfg.best_path}")
    print(f"saved config sidecar -> {cfg.config_path}")
    return online


def _save_checkpoint(net: DuelingDQN, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
    print(f"saved checkpoint -> {path}")


def main() -> None:
    train(TrainConfig())


if __name__ == "__main__":
    main()
