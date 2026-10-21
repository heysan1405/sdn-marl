"""Seeded episodic actor-critic training for either live routing policy."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from marl_live import OBSERVATION_SCHEMA_VERSION
from marl_live.config import RoutingConfig, FEATURE_NAMES
from marl_live.environment import RoutingEnv
from marl_live.policy import RoutingPolicy


def train(agent, episodes, seed, out_dir, learning_rate, hidden_dim, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = RoutingConfig(learning_rate=learning_rate, hidden_dim=hidden_dim)
    env = RoutingEnv(config)
    policy = RoutingPolicy(hidden_dim=hidden_dim, device=device)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=learning_rate)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    for episode in range(episodes):
        episode_seed = seed + episode
        topo_rng = random.Random(episode_seed ^ 0x45A17)
        cross_link_probability = topo_rng.uniform(
            config.train_cross_link_probability_min,
            config.train_cross_link_probability_max,
        )
        scenario = (("normal", "congestion")[episode % 2] if agent == "congestion"
                    else ("failure", "combined")[episode % 2])
        observation = env.reset(episode_seed, scenario=scenario, agent=agent,
                                cross_link_probability=cross_link_probability)
        log_probs, values, entropies, rewards = [], [], [], []
        selected_utils = []
        done = False
        final_info = {}
        invalid_actions = 0
        decisions = 0
        policy.model.train()
        while not done:
            dist, value = policy.distribution(observation)
            action = dist.sample()
            action_idx = int(action.item())
            if not bool(observation["action_mask"][action_idx]):
                invalid_actions += 1
                raise RuntimeError("masked policy sampled an invalid action")
            selected_utils.append(float(observation["candidate_features"][action_idx, 0]))
            decisions += 1
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value)
            observation, reward, done, final_info = env.step(action_idx)
            rewards.append(float(reward))
        # Fixed scale improves optimization stability and is recorded in metadata.
        returns, running = [], 0.0
        for reward in reversed(rewards):
            running = reward / config.reward_scale + config.gamma * running
            returns.append(running)
        returns.reverse()
        targets = torch.tensor(returns, dtype=torch.float32, device=device)
        value_tensor = torch.stack(values)
        advantage = targets - value_tensor.detach()
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        value_loss = F.smooth_l1_loss(value_tensor, targets)
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
        optimizer.step()
        rows.append({
            "episode": episode + 1, "seed": episode_seed, "scenario": scenario,
            "cross_link_probability": cross_link_probability,
            "reward": sum(rewards), "delivered": int(final_info.get("delivered", False)),
            "hops": max(0, len(final_info.get("path", [])) - 1),
            "path": json.dumps(final_info.get("path", [])),
            "first_hop_node": (final_info.get("path", [None, None])[1]
                               if len(final_info.get("path", [])) > 1 else None),
            "invalid_actions": invalid_actions,
            "decisions": decisions,
            "loop": int(final_info.get("loop", False)),
            "dead_end": int(final_info.get("failed", False)),
            "mean_selected_link_utilization": float(np.mean(selected_utils)) if selected_utils else None,
            "max_selected_link_utilization": max(selected_utils) if selected_utils else None,
            "sum_weighted_path_cost": float(sum(
                config.hop_cost + config.congestion_weight * util for util in selected_utils)),
            "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()), "optimizer_updates": 1,
        })
    runtime = time.perf_counter() - start
    with (out_dir / "training_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "agent": agent, "seed": seed, "episodes": episodes,
        "optimizer_updates": episodes, "runtime_seconds": runtime,
        "algorithm": "episodic categorical actor-critic with REINFORCE policy gradient",
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "action_schema": "candidate-neighbor index; failed/visited/unreachable/max-hop candidates masked",
        "reward_normalization": "discounted learning targets divided by fixed reward_scale=10; environment rewards retained raw in CSV",
        "action_mask": "failed, visited, no-path-to-destination, and max-hop actions are masked in both training and inference",
        "congestion_reward_objective": "negative sum per hop of 1 + 3 * observed background utilization, matching established weighted-Dijkstra baseline; delivery bonus and dead-end penalty; offered traffic is never reduced",
        "failure_reward_objective": "negative hop_penalty per hop; delivery bonus and dead-end penalty",
        "training_config": {**config.__dict__, "episodes": episodes, "seed": seed,
                            "learning_rate": learning_rate, "hidden_dim": hidden_dim},
        "runtime_environment": "NetworkX generated training graph; not live network measurements",
        "environment_settings": {
            "nodes": 8, "source": 0, "destination": 7,
            "branches": [[0, 1, 4, 7], [0, 2, 5, 7], [0, 3, 6, 7]],
            "cross_link_probability_range": [config.train_cross_link_probability_min,
                                            config.train_cross_link_probability_max],
            "normal_background_mbps_per_branch": [4.0, 22.0],
            "congestion_extra_mbps": [100.0, 190.0],
            "failure_count": 1, "training_scenarios": ["normal", "congestion"]
                if agent == "congestion" else ["failure", "combined"],
        },
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "networkx": nx.__version__,
                     "device": str(device)},
    }
    policy.save(out_dir / f"{agent}_policy.pt", metadata)
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("congestion", "failure"), required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, help="new, agent-specific output directory")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    metadata = train(args.agent, args.episodes, args.seed, args.out,
                     args.learning_rate, args.hidden_dim, args.device)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
