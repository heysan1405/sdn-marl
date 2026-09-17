"""
Two-agent SDN-MARL training pipeline.

Agents:
    1. Congestion Agent
    2. Failure Agent

Environment:
    simulator.environment.SDNEnvironment
"""

import csv
import os
import random

import numpy as np
import torch

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

def create_environment(topology_name, episode):
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
                ]
            )


# ============================================================
# Main training function
# ============================================================

def main():

    set_seed(SEED)

    # ========================================================
    # Output directory
    # ========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

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
        f"Training topologies:   {TRAINING_TOPOLOGIES}"
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

        topology_name = TRAINING_TOPOLOGIES[
            (episode - 1)
            % len(TRAINING_TOPOLOGIES)
        ]

        # ----------------------------------------------------
        # Create environment for this episode
        # ----------------------------------------------------

        env = create_environment(
            topology_name,
            episode,
        )

        try:

            state = env.reset()

            validate_state(state)

            done = False
            steps = 0
            total_reward = 0.0

            congestion_losses = []
            failure_losses = []

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
                )

                failure_action = (
                    failure_agent.select_action(
                        state,
                        training=True,
                    )
                )

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

                congestion_agent.remember(
                    state,
                    congestion_action,
                    reward,
                    next_state,
                    done,
                )

                # ---------------------------------------------
                # Store failure experience
                # ---------------------------------------------

                failure_agent.remember(
                    state,
                    failure_action,
                    reward,
                    next_state,
                    done,
                )

                # ---------------------------------------------
                # Learn
                # ---------------------------------------------

                congestion_loss = (
                    congestion_agent.learn()
                )

                failure_loss = (
                    failure_agent.learn()
                )

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

            congestion_agent.save(
                congestion_path
            )

            failure_agent.save(
                failure_path
            )

            print(
                f"Saved checkpoint at "
                f"episode {episode}"
            )

    # ========================================================
    # Final models
    # ========================================================

    congestion_final_path = os.path.join(
        CHECKPOINT_DIR,
        "congestion_final.pt",
    )

    failure_final_path = os.path.join(
        CHECKPOINT_DIR,
        "failure_final.pt",
    )

    congestion_agent.save(
        congestion_final_path
    )

    failure_agent.save(
        failure_final_path
    )

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

    # ========================================================
    # Completion
    # ========================================================

    print()
    print("=" * 70)
    print("TWO-AGENT TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Congestion model: "
        f"{congestion_final_path}"
    )

    print(
        "Failure model:    "
        f"{failure_final_path}"
    )

    print(
        "Training history: "
        f"{history_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()