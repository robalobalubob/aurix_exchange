"""Focused tests for prioritized replay and its sum-tree index.

These checks protect the mechanics that decide which past transitions the DQN
learns from.  A replay bug can change the effective training distribution even
when the environment, network, and loss function are all correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.training.replay_buffer import PrioritizedReplayBuffer, SumTree


def _add_numbered_transitions(
    buffer: PrioritizedReplayBuffer,
    count: int,
) -> None:
    """Populate ``buffer`` with transitions that reveal sampled indices."""
    for index in range(count):
        obs = np.full(buffer.obs_dim, index, dtype=np.float64)
        next_obs = obs + 0.5
        next_mask = np.zeros(buffer.n_actions, dtype=np.bool_)
        next_mask[index % buffer.n_actions] = True
        buffer.add(
            obs=obs,
            action=index % buffer.n_actions,
            reward=float(index) + 0.25,
            next_obs=next_obs,
            done=index % 2 == 0,
            next_mask=next_mask,
        )


@pytest.mark.correctness
def test_sum_tree_updates_total_and_cumulative_lookup() -> None:
    """Leaf changes must reach the root and preserve cumulative sampling."""
    tree = SumTree(capacity=4)
    for leaf_idx, priority in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(leaf_idx, priority)

    assert tree.total() == pytest.approx(10.0)
    probes = [0.5, 1.5, 4.5, 8.5]
    expected = [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]
    assert [tree.get(value) for value in probes] == expected

    tree.update(1, 6.0)

    assert tree.total() == pytest.approx(14.0)
    probes = [0.5, 3.0, 8.0, 12.0]
    expected = [(0, 1.0), (1, 6.0), (2, 3.0), (3, 4.0)]
    assert [tree.get(value) for value in probes] == expected


@pytest.mark.correctness
def test_seeded_prioritized_sampling_is_reproducible() -> None:
    """Equal seeds and replay histories must produce equal sample streams."""
    buffers = [
        PrioritizedReplayBuffer(
            capacity=8,
            n_actions=4,
            obs_dim=3,
            rng=np.random.default_rng(2026),
        )
        for _ in range(2)
    ]
    for buffer in buffers:
        _add_numbered_transitions(buffer, count=8)
        buffer.update_priorities(
            np.arange(8, dtype=np.int64),
            np.linspace(0.1, 2.0, 8, dtype=np.float64),
        )

    for _ in range(3):
        left = buffers[0].sample(batch_size=5, beta=0.7)
        right = buffers[1].sample(batch_size=5, beta=0.7)
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.correctness
def test_priority_updates_follow_absolute_td_error() -> None:
    """PER priority is ``(|TD error| + eps) ** alpha`` for each leaf."""
    buffer = PrioritizedReplayBuffer(
        capacity=4,
        n_actions=2,
        alpha=0.5,
        eps=0.25,
        obs_dim=3,
        rng=np.random.default_rng(0),
    )
    _add_numbered_transitions(buffer, count=4)

    indices = np.array([0, 2], dtype=np.int64)
    td_errors = np.array([-3.0, 0.0], dtype=np.float64)
    buffer.update_priorities(indices, td_errors)

    expected_total = np.sqrt(3.25) + 1.0 + np.sqrt(0.25) + 1.0
    assert buffer._tree.total() == pytest.approx(expected_total)
    assert buffer._max_priority == pytest.approx(3.25)

    # Once full, the next insertion overwrites leaf zero. It should receive the
    # largest priority seen so it is not starved before its first update.
    buffer.add(
        obs=np.zeros(3, dtype=np.float32),
        action=0,
        reward=0.0,
        next_obs=np.ones(3, dtype=np.float32),
        done=False,
        next_mask=np.ones(2, dtype=np.bool_),
    )
    assert buffer._tree.total() == pytest.approx(expected_total)


@pytest.mark.correctness
def test_sample_shapes_dtypes_and_normalized_weights() -> None:
    """A sampled batch must match the tensor contract consumed by DQN loss."""
    buffer = PrioritizedReplayBuffer(
        capacity=6,
        n_actions=4,
        obs_dim=3,
        rng=np.random.default_rng(17),
    )
    _add_numbered_transitions(buffer, count=6)

    batch = buffer.sample(batch_size=4, beta=0.6)

    expected = {
        "obs": ((4, 3), np.float32),
        "actions": ((4,), np.int64),
        "rewards": ((4,), np.float32),
        "next_obs": ((4, 3), np.float32),
        "dones": ((4,), np.float32),
        "next_masks": ((4, 4), np.bool_),
        "indices": ((4,), np.int64),
        "weights": ((4,), np.float32),
    }
    for key, (shape, dtype) in expected.items():
        assert batch[key].shape == shape
        assert batch[key].dtype == dtype

    assert np.all(batch["weights"] > 0.0)
    assert np.all(batch["weights"] <= 1.0)
    assert float(batch["weights"].max()) == pytest.approx(1.0)
