from __future__ import annotations

import numpy as np

OBS_DIM = 10


class SumTree:
    """Fixed-capacity binary sum-tree for O(log n) priority-weighted sampling.

    Leaves hold transition priorities; internal nodes hold the sum of their
    children. ``total()`` returns the root sum, and ``get(value)`` walks the
    tree to find the leaf whose cumulative range contains ``value``.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Internal nodes occupy [0, capacity-1); leaves occupy [capacity-1, 2*capacity-1)
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self._tree[0])

    def update(self, leaf_idx: int, priority: float) -> None:
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # Propagate the change up to the root
        parent = (tree_idx - 1) // 2
        while True:
            self._tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, value: float) -> tuple[int, float]:
        """Return (leaf_idx, priority) for the leaf whose range contains value."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self._tree):
                break
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, float(self._tree[idx])


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al. 2016).

    Stores transitions with per-sample priorities ``p = (|delta| + eps) ** alpha``
    and draws minibatches in proportion to those priorities, returning
    importance-sampling weights to correct the induced bias. The next-state
    action mask is stored so the Double DQN target can exclude invalid actions.
    """

    def __init__(
        self,
        capacity: int,
        n_actions: int,
        alpha: float = 0.6,
        eps: float = 1e-6,
        obs_dim: int = OBS_DIM,
    ) -> None:
        self.capacity = capacity
        self.n_actions = n_actions
        self.alpha = alpha
        self.eps = eps
        self.obs_dim = obs_dim

        self._tree = SumTree(capacity)
        self._max_priority = 1.0
        self._size = 0
        self._next = 0

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._next_masks = np.zeros((capacity, n_actions), dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        i = self._next
        self._obs[i] = obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._dones[i] = float(done)
        self._next_masks[i] = next_mask

        # New transitions enter at max priority so they are seen at least once
        self._tree.update(i, self._max_priority ** self.alpha)

        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> dict:
        """Draw a prioritized minibatch.

        Returns a dict of numpy arrays plus the sampled leaf indices (for the
        subsequent priority update) and the importance-sampling weights.
        """
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        total = self._tree.total()
        segment = total / batch_size
        for k in range(batch_size):
            # Stratified sampling: one draw per equal-probability segment
            lo = segment * k
            hi = segment * (k + 1)
            value = np.random.uniform(lo, hi)
            leaf_idx, priority = self._tree.get(value)
            indices[k] = leaf_idx
            priorities[k] = priority

        probs = priorities / total
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max()

        return {
            "obs": self._obs[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_obs[indices],
            "dones": self._dones[indices],
            "next_masks": self._next_masks[indices],
            "indices": indices,
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(td_errors) + self.eps
        for leaf_idx, priority in zip(indices, priorities):
            self._tree.update(int(leaf_idx), float(priority) ** self.alpha)
            self._max_priority = max(self._max_priority, float(priority))
