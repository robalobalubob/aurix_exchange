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
    checkpoint_path: str = "exports/dqn_checkpoint.pt"
    config_path: str = "exports/aurix_config.json"
    seed: Optional[int] = 0


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
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed if cfg.seed is not None else 0)

    env = AurixExchangeEnv(seed=cfg.seed)
    online = DuelingDQN(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    target = DuelingDQN(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = PrioritizedReplayBuffer(
        cfg.buffer_capacity, N_ACTIONS, alpha=cfg.per_alpha
    )

    obs, info = env.reset()
    mask = info["action_mask"]

    running_loss = 0.0
    loss_count = 0
    episode_return = 0.0
    episode_returns: list[float] = []
    best_avg_return = float("-inf")

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

        # Log
        if step % cfg.log_every == 0:
            avg_loss = running_loss / max(loss_count, 1)
            recent = episode_returns[-20:]
            avg_ret = float(np.mean(recent)) if recent else float("nan")
            if recent and avg_ret > best_avg_return:
                best_avg_return = avg_ret

            # Live metrics on the bar itself; full line written above it.
            pbar.set_postfix(
                eps=f"{eps:.3f}",
                loss=f"{avg_loss:.4f}",
                ret=f"{avg_ret:.2f}",
                best=f"{best_avg_return:.2f}",
                buf=len(buffer),
                ep=len(episode_returns),
                refresh=False,
            )
            pbar.write(
                f"step {step:>7} | eps {eps:5.3f} | "
                f"loss {avg_loss:8.5f} | avg_return {avg_ret:8.3f} | "
                f"best {best_avg_return:8.3f} | "
                f"buffer {len(buffer):>6} | episodes {len(episode_returns):>5}"
            )
            running_loss = 0.0
            loss_count = 0

    pbar.close()
    _save_checkpoint(online, cfg.checkpoint_path)
    export_config(env.cfg, cfg.config_path)
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
