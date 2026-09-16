"""
Training pipeline for the SDN GNN + DQN routing agent.

This version uses DummySDNRoutingEnvironment so the complete
Person 2 pipeline can be tested before Person 1's real environment
is available.
"""

import csv
import os

from agents.routing_agent import RoutingAgent
from simulator.dummy_simulator import DummySDNRoutingEnvironment

from training.config import (
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
    PATH_FEATURE_DIM,
    REPLAY_CAPACITY,
    ROUTING_HIDDEN_DIM,
    TARGET_UPDATE_FREQUENCY,
)


def save_training_history(history, filepath):
    """Save episode statistics to a CSV file."""

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "episode",
                "total_reward",
                "average_reward",
                "average_loss",
                "epsilon",
                "steps",
            ]
        )

        for row in history:
            writer.writerow(
                [
                    row["episode"],
                    row["total_reward"],
                    row["average_reward"],
                    row["average_loss"],
                    row["epsilon"],
                    row["steps"],
                ]
            )


def main():

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Create dummy environment
    # ---------------------------------------------------------

    env = DummySDNRoutingEnvironment(
        min_nodes=5,
        max_nodes=8,
        num_candidate_paths=4,
        max_steps=MAX_STEPS_PER_EPISODE,
        seed=42,
    )

    # ---------------------------------------------------------
    # Create DQN routing agent
    # ---------------------------------------------------------

    agent = RoutingAgent(
        node_feature_dim=6,
        edge_feature_dim=EDGE_FEATURE_DIM,
        demand_feature_dim=1,
        path_feature_dim=PATH_FEATURE_DIM,

        gnn_hidden_dim=GNN_HIDDEN_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
        gnn_layers=GNN_LAYERS,
        gnn_heads=GNN_HEADS,
        gnn_dropout=GNN_DROPOUT,

        routing_hidden_dim=ROUTING_HIDDEN_DIM,

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

    # ---------------------------------------------------------
    # Print configuration
    # ---------------------------------------------------------

    print("=" * 70)
    print("SDN GNN + DQN TRAINING")
    print("=" * 70)

    print(f"Episodes:       {NUM_EPISODES}")
    print(f"Max steps:      {MAX_STEPS_PER_EPISODE}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Gamma:          {GAMMA}")
    print(f"Device:         {agent.device}")

    print("=" * 70)

    # ---------------------------------------------------------
    # Training history
    # ---------------------------------------------------------

    history = []

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    for episode in range(1, NUM_EPISODES + 1):

        state = env.reset()

        total_reward = 0.0
        episode_losses = []

        steps = 0
        done = False

        # -----------------------------------------------------
        # Episode loop
        # -----------------------------------------------------

        while not done and steps < MAX_STEPS_PER_EPISODE:

            # Select action using epsilon-greedy policy
            action = agent.select_action(
                state,
                training=True,
            )

            # Execute action in environment
            next_state, reward, done, info = env.step(action)

            # Store transition in replay buffer
            agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            # Learn from replay buffer
            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            # Move to next state
            state = next_state

            total_reward += reward
            steps += 1

        # -----------------------------------------------------
        # Calculate episode statistics
        # -----------------------------------------------------

        average_reward = total_reward / max(steps, 1)

        if episode_losses:
            average_loss = sum(episode_losses) / len(episode_losses)
        else:
            average_loss = 0.0

        episode_result = {
            "episode": episode,
            "total_reward": total_reward,
            "average_reward": average_reward,
            "average_loss": average_loss,
            "epsilon": agent.epsilon,
            "steps": steps,
        }

        history.append(episode_result)

        # -----------------------------------------------------
        # Print progress
        # -----------------------------------------------------

        if episode == 1 or episode % LOG_FREQUENCY == 0:

            print(
                f"Episode {episode:4d}/{NUM_EPISODES} | "
                f"Reward: {total_reward:8.3f} | "
                f"Avg Reward: {average_reward:7.3f} | "
                f"Loss: {average_loss:8.5f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Steps: {steps:3d}"
            )

        # -----------------------------------------------------
        # Save checkpoint
        # -----------------------------------------------------

        if episode % CHECKPOINT_FREQUENCY == 0:

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_episode_{episode}.pt",
            )

            agent.save(checkpoint_path)

            print(
                f"Saved checkpoint: {checkpoint_path}"
            )

    # ---------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------

    final_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "final_model.pt",
    )

    agent.save(final_checkpoint)

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------

    history_path = os.path.join(
        CHECKPOINT_DIR,
        "training_history.csv",
    )

    save_training_history(
        history,
        history_path,
    )

    # ---------------------------------------------------------
    # Final message
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"Final model:      {final_checkpoint}")
    print(f"Training history: {history_path}")
    print(f"Replay buffer:    {len(agent.replay_buffer)}")
    print(f"Final epsilon:    {agent.epsilon:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()