"""
Two-agent SDN-MARL training pipeline.

Agents:
    1. Congestion Agent
    2. Failure Agent

The current version uses the dummy environment.
"""

import csv
import os

from agents.congestion_agent import CongestionAgent
from agents.failure_agent import FailureAgent

from simulator.dummy_simulator import (
    DummySDNMultiAgentEnvironment,
)

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


def save_training_history(
    history,
    filepath,
):
    """Save training statistics to CSV."""

    directory = os.path.dirname(
        filepath
    )

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
                    row["total_reward"],
                    row["average_reward"],
                    row["congestion_loss"],
                    row["failure_loss"],
                    row["congestion_epsilon"],
                    row["failure_epsilon"],
                    row["steps"],
                ]
            )


def main():

    # ================================================
    # Output directory
    # ================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # ================================================
    # Environment
    # ================================================

    env = DummySDNMultiAgentEnvironment(
        min_nodes=5,
        max_nodes=8,
        max_steps=MAX_STEPS_PER_EPISODE,
        seed=42,
    )

    # ================================================
    # Congestion Agent
    # ================================================

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

    # ================================================
    # Failure Agent
    # ================================================

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

    # ================================================
    # Print configuration
    # ================================================

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

    # ================================================
    # Training
    # ================================================

    for episode in range(
        1,
        NUM_EPISODES + 1,
    ):

        state = env.reset()

        done = False
        steps = 0
        total_reward = 0.0

        congestion_losses = []
        failure_losses = []

        while (
            not done
            and steps
            < MAX_STEPS_PER_EPISODE
        ):

            # ----------------------------------------
            # Agent decisions
            # ----------------------------------------

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

            # ----------------------------------------
            # Environment
            # ----------------------------------------

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(
                congestion_action,
                failure_action,
            )

            # ----------------------------------------
            # Store experiences
            # ----------------------------------------

            congestion_agent.remember(
                state,
                congestion_action,
                reward,
                next_state,
                done,
            )

            failure_agent.remember(
                state,
                failure_action,
                reward,
                next_state,
                done,
            )

            # ----------------------------------------
            # Learn
            # ----------------------------------------

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

            state = next_state

            total_reward += reward

            steps += 1

        # ============================================
        # Episode statistics
        # ============================================

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

        # ============================================
        # Logging
        # ============================================

        if (
            episode == 1
            or episode % LOG_FREQUENCY == 0
        ):

            print(
                f"Episode {episode:4d}/{NUM_EPISODES} | "
                f"Reward: {total_reward:8.3f} | "
                f"CongLoss: "
                f"{average_congestion_loss:.5f} | "
                f"FailLoss: "
                f"{average_failure_loss:.5f} | "
                f"CongEps: "
                f"{congestion_agent.epsilon:.3f} | "
                f"FailEps: "
                f"{failure_agent.epsilon:.3f}"
            )

        # ============================================
        # Checkpoints
        # ============================================

        if (
            episode
            % CHECKPOINT_FREQUENCY
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

    # ================================================
    # Final models
    # ================================================

    congestion_agent.save(
        os.path.join(
            CHECKPOINT_DIR,
            "congestion_final.pt",
        )
    )

    failure_agent.save(
        os.path.join(
            CHECKPOINT_DIR,
            "failure_final.pt",
        )
    )

    # ================================================
    # Training history
    # ================================================

    save_training_history(
        history,
        os.path.join(
            CHECKPOINT_DIR,
            "multi_agent_history.csv",
        ),
    )

    print()
    print("=" * 70)
    print("TWO-AGENT TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Congestion model: "
        f"{CHECKPOINT_DIR}/congestion_final.pt"
    )

    print(
        "Failure model:    "
        f"{CHECKPOINT_DIR}/failure_final.pt"
    )

    print(
        "Training history: "
        f"{CHECKPOINT_DIR}/multi_agent_history.csv"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()