"""Extract Phase 2 policy-structure data for visualization.

Sweeps the (z, f) state plane at t=0 and records, for the DP optimum and every
ladder checkpoint (DQN, REINFORCE, A2C, PPO):

- the greedy action map (the learned decision boundaries vs the no-trade band);
- the oracle-priced cost of each decision, Q*(s, a*) - Q*(s, a_policy), via
  ``dp.one_step_q`` — "where the regret lives" as a heatmap;
- the greedy-REINFORCE state-visitation density over the same grid (CRN eval
  seed block), to show which of those costs are actually incurred.

Output: ``exports/phase2_visual_data.json`` (regenerable; not committed).

Usage:
    python -m notebooks.phase2_policy_maps
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.env.ou_core import OUCoreConfig, OUTradingEnv
from src.models.dqn import DuelingDQN
from src.models.policy import ActorCriticNet, PolicyNet
from src.solvers import dp, regret
from src.training.train_reinforce import greedy_policy

DP_CACHE = "exports/dp_c7e2adda5358b757.npz"
OUT_PATH = "exports/phase2_visual_data.json"

N_Z, Z_LO, Z_HI = 121, -3.0, 3.0
N_F = 81
DENSITY_EPISODES = 300


def _load_nets(core: OUCoreConfig) -> dict[str, torch.nn.Module]:
    d, a = core.obs_dim, core.n_actions
    nets: dict[str, torch.nn.Module] = {
        "dqn": DuelingDQN(obs_dim=d, n_actions=a),
        "reinforce": PolicyNet(obs_dim=d, n_actions=a),
        "a2c": ActorCriticNet(obs_dim=d, n_actions=a),
        "ppo": ActorCriticNet(obs_dim=d, n_actions=a),
    }
    paths = {
        "dqn": "exports/core_best.pt",
        "reinforce": "exports/reinforce_best.pt",
        "a2c": "exports/a2c_best.pt",
        "ppo": "exports/ppo_best.pt",
    }
    for name, net in nets.items():
        net.load_state_dict(torch.load(paths[name]))
        net.eval()
    return nets


def _grid_actions(net: torch.nn.Module, obs: torch.Tensor) -> np.ndarray:
    """Greedy action per grid row; scoring vector is Q for the DQN, logits for
    the policy nets — argmax semantics are identical."""
    with torch.no_grad():
        out = net(obs)
        scores = out[0] if isinstance(out, tuple) else out
        return torch.argmax(scores, dim=1).numpy()


def main() -> None:
    core = OUCoreConfig()
    res = dp.DPResult.load(DP_CACHE)
    nets = _load_nets(core)

    z_axis = np.linspace(Z_LO, Z_HI, N_Z)
    f_axis = np.linspace(0.0, 1.0, N_F)
    zz, ff = np.meshgrid(z_axis, f_axis, indexing="ij")
    flat_z, flat_f = zz.ravel(), ff.ravel()

    obs = torch.from_numpy(
        np.stack([flat_z, flat_f, np.zeros_like(flat_z)], axis=1).astype(np.float32)
    )

    actions = {
        name: _grid_actions(net, obs).reshape(N_Z, N_F)
        for name, net in nets.items()
    }
    actions["pistar"] = np.array(
        [res.policy_at(0, z, f) for z, f in zip(flat_z, flat_f)], dtype=np.int64
    ).reshape(N_Z, N_F)

    # Oracle Q per grid point prices every policy's decisions in one pass.
    print("pricing decisions against oracle Q ...")
    q_grid = np.array(
        [dp.one_step_q(res, 0, z, f) for z, f in zip(flat_z, flat_f)]
    )  # (N_Z * N_F, n_actions)
    q_best = q_grid.max(axis=1)
    costs = {
        name: (q_best - q_grid[np.arange(len(flat_z)), acts.ravel()])
        .reshape(N_Z, N_F)
        for name, acts in actions.items()
    }

    # Visitation density of the scored (greedy, CRN seed block) REINFORCE agent.
    print("rolling greedy REINFORCE for visitation density ...")
    pol = greedy_policy(nets["reinforce"])
    env = OUTradingEnv(config=core)
    visits_z, visits_f = [], []
    for seed in regret.make_seed_block(DENSITY_EPISODES):
        ob, _ = env.reset(seed=int(seed))
        t = 0
        while True:
            visits_z.append(float(ob[0]))
            visits_f.append(float(ob[1]))
            ob, _, term, trunc, _ = env.step(pol(ob, t))
            t += 1
            if term or trunc:
                break
    z_edges = np.linspace(Z_LO, Z_HI, N_Z + 1)
    f_edges = np.linspace(0.0, 1.0, N_F + 1)
    density, _, _ = np.histogram2d(visits_z, visits_f, bins=[z_edges, f_edges])

    payload = {
        "meta": {
            "config_hash": res.config_hash,
            "t_slice": 0,
            "z_axis": [round(float(v), 4) for v in z_axis],
            "f_axis": [round(float(v), 4) for v in f_axis],
            "action_names": ["HOLD", "BUY", "SELL"],
            "density_episodes": DENSITY_EPISODES,
        },
        "actions": {k: v.astype(int).tolist() for k, v in actions.items()},
        "costs": {k: np.round(v, 5).tolist() for k, v in costs.items()},
        "density": density.astype(int).tolist(),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"wrote {OUT_PATH}")

    # Console sanity summary: mean oracle-priced cost per policy (uniform grid).
    for name in ["pistar", "dqn", "reinforce", "a2c", "ppo"]:
        print(f"{name:>10}: mean grid cost {costs[name].mean():.5f} nats/step, "
              f"disagreement {np.mean(actions[name] != actions['pistar']):.1%}")


if __name__ == "__main__":
    main()
