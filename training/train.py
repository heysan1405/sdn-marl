"""
Two-agent SDN-MARL training pipeline.

Agents:
    1. Congestion Agent
    2. Failure Agent

Environment:
    simulator.environment.SDNEnvironment
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Support both `python -m training.train` and direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.congestion_agent import CongestionAgent
from agents.failure_agent import FailureAgent
from simulator.environment import SDNEnvironment

from training.config import (
    AGENT_HIDDEN_DIM,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_FREQUENCY,
    DEVICE,
    EDGE_FEATURE_DIM,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    GNN_DROPOUT,
    GNN_HEADS,
    GNN_HIDDEN_DIM,
    GNN_LAYERS,
    GRAPH_EMBEDDING_DIM,
    LEARNING_RATE,
    LOG_FREQUENCY,
    MAX_STEPS_PER_EPISODE,
    NUM_EPISODES,
    REPLAY_CAPACITY,
    TARGET_UPDATE_FREQUENCY,
)


# ============================================================
# Training configuration
# ============================================================

TRAINING_TOPOLOGIES = [
    "UsCarrier.gml",
    "Colt.gml",
    "GtsCe.gml",
    "TataNld.gml",
    "Pern.gml",
    "VtlWavenet2011.gml",
    "VtlWavenet2008.gml",
    "RedBestel.gml",
    "Ulaknet.gml",
    "Surfnet.gml",
    "Uunet.gml",
    "Bellcanada.gml",
    "Belnet2006.gml",
    "Belnet2003.gml",
    "Belnet2004.gml",
    "Internetmci.gml",
    "Rediris.gml",
    "Easynet.gml",
]

TOPOLOGY_DIR = os.path.join(
    "InternetTopologyZoo",
    "gml",
)

TRAFFIC_SCENARIO = "normal"

NUM_DEMANDS = 10

K_PATHS = 5

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# State validation
# ============================================================

def validate_state(state):
    """
    Validate the state returned by SDNEnvironment.

    Expected state:
        node_features   [N, 7]
        edge_index      [2, E]
        edge_features   [E, 5]
        demand_features [1, 1]
    """

    required_keys = {
        "node_features",
        "edge_index",
        "edge_features",
        "demand_features",
    }

    missing = required_keys - set(state.keys())

    if missing:
        raise KeyError(
            f"State is missing required keys: {sorted(missing)}"
        )

    node_features = torch.as_tensor(
        state["node_features"]
    )

    edge_index = torch.as_tensor(
        state["edge_index"]
    )

    edge_features = torch.as_tensor(
        state["edge_features"]
    )

    demand_features = torch.as_tensor(
        state["demand_features"]
    )

    if node_features.dim() != 2:
        raise ValueError(
            "node_features must be 2-dimensional."
        )

    if node_features.shape[1] != 7:
        raise ValueError(
            "node_features must have shape [N, 7]. "
            f"Got {tuple(node_features.shape)}."
        )

    if edge_index.dim() != 2:
        raise ValueError(
            "edge_index must be 2-dimensional."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape [2, E]. "
            f"Got {tuple(edge_index.shape)}."
        )

    if edge_features.dim() != 2:
        raise ValueError(
            "edge_features must be 2-dimensional."
        )

    if edge_features.shape[1] != EDGE_FEATURE_DIM:
        raise ValueError(
            "edge_features must have shape [E, 5]. "
            f"Got {tuple(edge_features.shape)}."
        )

    if edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError(
            "Number of edge features must match "
            "number of edges in edge_index."
        )

    if demand_features.dim() != 2:
        raise ValueError(
            "demand_features must be 2-dimensional."
        )

    if demand_features.shape != (1, 1):
        raise ValueError(
            "demand_features must have shape [1, 1]. "
            f"Got {tuple(demand_features.shape)}."
        )

    return True


# ============================================================
# Environment creation
# ============================================================

def create_environment(topology_name, episode, traffic_scale=1.0):
    """Create an SDN environment for one training episode."""

    topology_path = os.path.join(
        TOPOLOGY_DIR,
        topology_name,
    )

    if not os.path.isfile(topology_path):
        raise FileNotFoundError(
            f"Topology file not found: {topology_path}"
        )

    return SDNEnvironment(
        topology_path=topology_path,
        traffic_scenario=TRAFFIC_SCENARIO,
        num_demands=NUM_DEMANDS,
        max_steps=MAX_STEPS_PER_EPISODE,
        k_paths=K_PATHS,
        random_seed=SEED + episode,
        traffic_scale=traffic_scale,
    )


# ============================================================
# Save training history
# ============================================================

def save_training_history(history, filepath):
    """Save training statistics to CSV."""

    directory = os.path.dirname(filepath)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "episode",
                "topology",
                "total_reward",
                "average_reward",
                "congestion_loss",
                "failure_loss",
                "congestion_epsilon",
                "failure_epsilon",
                "steps",
                "failure_injected",
                "congestion_action_0", "congestion_action_1", "congestion_action_2",
                "failure_action_0", "failure_action_1", "failure_action_2",
                "congestion_updates_total",
                "failure_updates_total",
            ]
        )

        for row in history:

            writer.writerow(
                [
                    row["episode"],
                    row["topology"],
                    row["total_reward"],
                    row["average_reward"],
                    row["congestion_loss"],
                    row["failure_loss"],
                    row["congestion_epsilon"],
                    row["failure_epsilon"],
                    row["steps"],
                    row["failure_injected"],
                    *row["congestion_action_counts"],
                    *row["failure_action_counts"],
                    row["congestion_updates"],
                    row["failure_updates"],
                ]
            )


# ============================================================
# Main training function
# ============================================================

def main():
    global NUM_EPISODES, MAX_STEPS_PER_EPISODE, NUM_DEMANDS
    global BATCH_SIZE, CHECKPOINT_DIR, SEED, TRAFFIC_SCENARIO

    parser = argparse.ArgumentParser(description="Train the existing congestion and failure GNN-DQN agents.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--num-demands", type=int, default=NUM_DEMANDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scenario", choices=["normal", "heavy", "single_hotspot", "multiple_hotspots", "traffic_spike", "random"], default=TRAFFIC_SCENARIO)
    parser.add_argument("--topologies", nargs="+", default=TRAINING_TOPOLOGIES,
                        help="Topology Zoo GML filenames rotated across episodes.")
    parser.add_argument("--failure-probability", type=float, default=0.5,
                        help="Per-episode probability of injecting one random link failure.")
    parser.add_argument("--traffic-scale", type=float, default=1.0,
                        help="Explicit simulator-generated demand multiplier; useful for stress scenarios.")
    parser.add_argument("--traffic-scales", type=float, nargs="+", default=None,
                        help="Optional episode-rotated traffic multipliers for controlled intensity diversity.")
    parser.add_argument("--train-agent", choices=["both", "congestion", "failure"], default="both",
                        help="Train both DQNs jointly, or isolate one with the other agent held at no-op.")
    parser.add_argument("--output-dir", default="results/training_dqn_seed42")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in output-dir.")
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.num_demands, args.batch_size) < 1:
        parser.error("episodes, max-steps, num-demands and batch-size must be positive")
    if not 0.0 <= args.failure_probability <= 1.0:
        parser.error("failure-probability must be between 0 and 1")
    if not np.isfinite(args.traffic_scale) or args.traffic_scale <= 0:
        parser.error("traffic-scale must be a finite positive number")
    if args.traffic_scales is not None and any(not np.isfinite(x) or x <= 0 for x in args.traffic_scales):
        parser.error("all traffic-scales must be finite positive numbers")
    if not args.topologies or any(
        Path(name).name != name or not name.lower().endswith(".gml")
        for name in args.topologies
    ):
        parser.error("topologies must be one or more GML filenames, without directory components")
    missing_topologies = [name for name in args.topologies if not os.path.isfile(os.path.join(TOPOLOGY_DIR, name))]
    if missing_topologies:
        parser.error(f"topology file(s) not found under {TOPOLOGY_DIR}: {', '.join(missing_topologies)}")
    NUM_EPISODES, MAX_STEPS_PER_EPISODE = args.episodes, args.max_steps
    NUM_DEMANDS, BATCH_SIZE = args.num_demands, args.batch_size
    SEED, TRAFFIC_SCENARIO, CHECKPOINT_DIR = args.seed, args.scenario, args.output_dir
    if os.path.isdir(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR) and not args.overwrite:
        parser.error(f"output directory is not empty: {CHECKPOINT_DIR}; choose a new directory or pass --overwrite")

    set_seed(SEED)
    run_started = time.perf_counter()

    # ========================================================
    # Output directory
    # ========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )
    run_config = vars(args).copy()
    run_config["device"] = str(DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_config["topologies"] = args.topologies
    traffic_scales = args.traffic_scales or [args.traffic_scale]
    run_config["traffic_scales"] = traffic_scales
    with open(os.path.join(CHECKPOINT_DIR, "config.json"), "w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, indent=2)

    # ========================================================
    # Congestion Agent
    # ========================================================

    congestion_agent = CongestionAgent(
        node_feature_dim=7,
        edge_feature_dim=EDGE_FEATURE_DIM,
        demand_feature_dim=1,

        gnn_hidden_dim=GNN_HIDDEN_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
        gnn_layers=GNN_LAYERS,
        gnn_heads=GNN_HEADS,
        gnn_dropout=GNN_DROPOUT,

        hidden_dim=AGENT_HIDDEN_DIM,

        learning_rate=LEARNING_RATE,
        gamma=GAMMA,

        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,

        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,

        target_update_frequency=TARGET_UPDATE_FREQUENCY,

        device=DEVICE,
    )

    # ========================================================
    # Failure Agent
    # ========================================================

    failure_agent = FailureAgent(
        node_feature_dim=7,
        edge_feature_dim=EDGE_FEATURE_DIM,
        demand_feature_dim=1,

        gnn_hidden_dim=GNN_HIDDEN_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
        gnn_layers=GNN_LAYERS,
        gnn_heads=GNN_HEADS,
        gnn_dropout=GNN_DROPOUT,

        hidden_dim=AGENT_HIDDEN_DIM,

        learning_rate=LEARNING_RATE,
        gamma=GAMMA,

        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,

        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,

        target_update_frequency=TARGET_UPDATE_FREQUENCY,

        device=DEVICE,
    )

    # ========================================================
    # Print configuration
    # ========================================================

    print("=" * 70)
    print("TWO-AGENT SDN-MARL TRAINING")
    print("=" * 70)

    print(
        f"Episodes:              {NUM_EPISODES}"
    )

    print(
        f"Max steps:             {MAX_STEPS_PER_EPISODE}"
    )

    print(
        f"Training topologies:   {args.topologies}"
    )

    print(
        f"Traffic scenario:      {TRAFFIC_SCENARIO}"
    )

    print(
        f"Number of demands:     {NUM_DEMANDS}"
    )

    print(
        f"K paths:               {K_PATHS}"
    )

    print(
        f"Batch size:            {BATCH_SIZE}"
    )

    print(
        f"Learning rate:         {LEARNING_RATE}"
    )

    print(
        f"Device:                {congestion_agent.device}"
    )

    print("=" * 70)

    history = []

    # ========================================================
    # Training
    # ========================================================

    for episode in range(
        1,
        NUM_EPISODES + 1,
    ):

        # ----------------------------------------------------
        # Rotate through training topologies
        # ----------------------------------------------------

        topology_name = args.topologies[
            (episode - 1)
            % len(args.topologies)
        ]

        # ----------------------------------------------------
        # Create environment for this episode
        # ----------------------------------------------------

        env = create_environment(
            topology_name,
            episode,
            traffic_scale=traffic_scales[(episode - 1) % len(traffic_scales)],
        )

        try:

            state = env.reset()
            injected_failure = None
            if random.Random(SEED + episode * 31).random() < args.failure_probability:
                link_load = {}
                for path in env.simulator.active_paths.values():
                    for u, v in zip(path[:-1], path[1:]):
                        edge = tuple(sorted((u, v)))
                        link_load[edge] = link_load.get(edge, 0) + 1
                if link_load:
                    max_load = max(link_load.values())
                    candidates = sorted(edge for edge, load in link_load.items() if load == max_load)
                    injected_failure = random.Random(SEED + episode * 97).choice(candidates)
                    env.inject_link_failure(*injected_failure)
                state = env._build_state()

            validate_state(state)

            done = False
            steps = 0
            total_reward = 0.0

            congestion_losses = []
            failure_losses = []
            congestion_action_counts = [0, 0, 0]
            failure_action_counts = [0, 0, 0]

            # =================================================
            # Episode loop
            # =================================================

            while (
                not done
                and steps < MAX_STEPS_PER_EPISODE
            ):

                # ---------------------------------------------
                # Validate current state
                # ---------------------------------------------

                validate_state(state)

                # ---------------------------------------------
                # Agent decisions
                # ---------------------------------------------

                congestion_action = (
                    congestion_agent.select_action(
                        state,
                        training=True,
                    )
                ) if args.train_agent in ("both", "congestion") else 0

                failure_action = (
                    failure_agent.select_action(
                        state,
                        training=True,
                    )
                ) if args.train_agent in ("both", "failure") else 0

                congestion_action_counts[congestion_action] += 1
                failure_action_counts[failure_action] += 1

                # ---------------------------------------------
                # Environment step
                # ---------------------------------------------

                (
                    next_state,
                    reward,
                    done,
                    info,
                ) = env.step(
                    congestion_action,
                    failure_action,
                )

                # ---------------------------------------------
                # Validate next state
                # ---------------------------------------------

                validate_state(next_state)

                # ---------------------------------------------
                # Store congestion experience
                # ---------------------------------------------

                if args.train_agent in ("both", "congestion"):
                    congestion_agent.remember(state, congestion_action, reward, next_state, done)

                # ---------------------------------------------
                # Store failure experience
                # ---------------------------------------------

                if args.train_agent in ("both", "failure"):
                    failure_agent.remember(state, failure_action, reward, next_state, done)

                # ---------------------------------------------
                # Learn
                # ---------------------------------------------

                congestion_loss = congestion_agent.learn() if args.train_agent in ("both", "congestion") else None

                failure_loss = failure_agent.learn() if args.train_agent in ("both", "failure") else None

                if congestion_loss is not None:
                    congestion_losses.append(
                        congestion_loss
                    )

                if failure_loss is not None:
                    failure_losses.append(
                        failure_loss
                    )

                # ---------------------------------------------
                # Advance state
                # ---------------------------------------------

                state = next_state

                total_reward += float(reward)

                steps += 1

        finally:

            # Release the episode environment.
            del env

        # ====================================================
        # Episode statistics
        # ====================================================

        average_reward = (
            total_reward
            / max(steps, 1)
        )

        average_congestion_loss = (
            sum(congestion_losses)
            / len(congestion_losses)
            if congestion_losses
            else 0.0
        )

        average_failure_loss = (
            sum(failure_losses)
            / len(failure_losses)
            if failure_losses
            else 0.0
        )

        result = {
            "episode": episode,
            "topology": topology_name,
            "total_reward": total_reward,
            "average_reward": average_reward,
            "congestion_loss":
                average_congestion_loss,
            "failure_loss":
                average_failure_loss,
            "congestion_epsilon":
                congestion_agent.epsilon,
            "failure_epsilon":
                failure_agent.epsilon,
            "steps": steps,
            "failure_injected": str(injected_failure) if injected_failure else "",
            "congestion_action_counts": congestion_action_counts,
            "failure_action_counts": failure_action_counts,
            "congestion_updates": congestion_agent.learn_steps,
            "failure_updates": failure_agent.learn_steps,
        }

        history.append(result)

        # ====================================================
        # Logging
        # ====================================================

        if (
            episode == 1
            or episode % LOG_FREQUENCY == 0
        ):

            print(
                f"Episode {episode:4d}/{NUM_EPISODES} | "
                f"Topology: {topology_name:20s} | "
                f"Reward: {total_reward:8.3f} | "
                f"CongLoss: {average_congestion_loss:.5f} | "
                f"FailLoss: {average_failure_loss:.5f} | "
                f"CongEps: {congestion_agent.epsilon:.3f} | "
                f"FailEps: {failure_agent.epsilon:.3f}"
            )

        # ====================================================
        # Checkpoints
        # ====================================================

        if (
            episode % CHECKPOINT_FREQUENCY
            == 0
        ):

            congestion_path = os.path.join(
                CHECKPOINT_DIR,
                f"congestion_episode_{episode}.pt",
            )

            failure_path = os.path.join(
                CHECKPOINT_DIR,
                f"failure_episode_{episode}.pt",
            )

            if args.train_agent in ("both", "congestion"):
                congestion_agent.save(congestion_path)
            if args.train_agent in ("both", "failure"):
                failure_agent.save(failure_path)

            print(
                f"Saved checkpoint at "
                f"episode {episode}"
            )

    # ========================================================
    # Final models
    # ========================================================

    congestion_final_path = os.path.join(CHECKPOINT_DIR, "congestion_dqn_final.pt")

    failure_final_path = os.path.join(CHECKPOINT_DIR, "failure_dqn_final.pt")

    if args.train_agent in ("both", "congestion"):
        congestion_agent.save(congestion_final_path)

    if args.train_agent in ("both", "failure"):
        failure_agent.save(failure_final_path)

    # ========================================================
    # Training history
    # ========================================================

    history_path = os.path.join(
        CHECKPOINT_DIR,
        "multi_agent_history.csv",
    )

    save_training_history(
        history,
        history_path,
    )

    summary = {
        "episodes": NUM_EPISODES,
        "environment_steps": sum(row["steps"] for row in history),
        "congestion_optimizer_updates": congestion_agent.learn_steps,
        "failure_optimizer_updates": failure_agent.learn_steps,
        "mean_episode_reward": sum(row["total_reward"] for row in history) / max(1, len(history)),
        "training_action_counts": {
            "congestion": [sum(row["congestion_action_counts"][i] for row in history) for i in range(3)],
            "failure": [sum(row["failure_action_counts"][i] for row in history) for i in range(3)],
        },
        "runtime_seconds": time.perf_counter() - run_started,
        "trained_agent": args.train_agent,
        "congestion_checkpoint": congestion_final_path if args.train_agent in ("both", "congestion") else None,
        "failure_checkpoint": failure_final_path if args.train_agent in ("both", "failure") else None,
        "config": run_config,
    }
    with open(os.path.join(CHECKPOINT_DIR, "training_summary.json"), "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2))

    # ========================================================
    # Completion
    # ========================================================

    print()
    print("=" * 70)
    print("TWO-AGENT TRAINING COMPLETE")
    print("=" * 70)

    if args.train_agent in ("both", "congestion"):
        print(f"Congestion model: {congestion_final_path}")
    if args.train_agent in ("both", "failure"):
        print(f"Failure model:    {failure_final_path}")

    print(
        "Training history: "
        f"{history_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
