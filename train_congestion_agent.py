import time
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from agents.sdn_env import SDNCongestionEnv
from agents.dqn_agent import DQNAgent


def run_training_loop(episodes=20, steps_per_episode=10, batch_size=16, checkpoint_path="congestion_agent.pt"):
    print("\n==================================================")
    print("   TRAINING CONGESTION AVOIDANCE RL AGENT (DQN)   ")
    print("==================================================")

    # Simulated/Live State Dimensions for Abilene (11 switches, 14 links, 28 directed edges)
    # 28 utils + 28 losses + max_util + mean_util = 58 state dimensions
    state_dim = 58
    action_dim = 29  # 0: baseline, 1..28: penalize congested link 1..28

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.95,
        batch_size=batch_size
    )

    # Initialize environment
    env = SDNCongestionEnv(controller=None, sampling_interval=0.1)
    env.state_dim = state_dim
    env.action_dim = action_dim
    env.link_keys = [f"s{i}_p1->s{j}_p2" for i in range(1, 15) for j in range(1, 15) if i != j][:28]

    # Generate synthetic dynamic traffic load baseline for offline training/benchmarking
    np.random.seed(42)

    total_rewards = []

    for ep in range(1, episodes + 1):
        # Initial state with random bottleneck congestion
        bottleneck_idx = np.random.randint(0, 28)
        sim_utils = np.random.uniform(0.1, 0.4, size=28)
        sim_utils[bottleneck_idx] = np.random.uniform(0.85, 0.98)  # Heavy bottleneck
        sim_losses = np.zeros(28)

        state = np.concatenate([
            sim_utils,
            sim_losses,
            np.array([np.max(sim_utils), np.mean(sim_utils)], dtype=np.float32)
        ])

        ep_reward = 0.0

        for step in range(steps_per_episode):
            action = agent.select_action(state)

            # Simulate environment transition:
            # If action penalizes the bottleneck link, traffic reroutes and max_util drops!
            next_utils = sim_utils.copy()
            if action > 0 and (action - 1) == bottleneck_idx:
                # Traffic steered away! Bottleneck load drops from 95% -> 45%
                next_utils[bottleneck_idx] = max(0.40, next_utils[bottleneck_idx] - 0.45)
                # Alternative link increases slightly
                alt_idx = (bottleneck_idx + 1) % 28
                next_utils[alt_idx] += 0.15

            next_state = np.concatenate([
                next_utils,
                sim_losses,
                np.array([np.max(next_utils), np.mean(next_utils)], dtype=np.float32)
            ])

            reward = env.compute_reward(next_state)
            done = step == (steps_per_episode - 1)

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            sim_utils = next_utils
            ep_reward += reward

        total_rewards.append(ep_reward)

        if ep % 5 == 0:
            agent.update_target_network()

        max_util = state[-2]
        print(f"Episode {ep:2d}/{episodes:2d} | Ep Reward: {ep_reward:7.2f} | Epsilon: {agent.epsilon:.3f} | Max Link Util: {max_util*100:5.1f}%")

    agent.save(checkpoint_path)
    print("\n[Training Complete] Final evaluation reward:", np.mean(total_rewards[-5:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Congestion Avoidance RL Agent")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default="congestion_agent.pt", help="Path to save trained policy weights")
    args = parser.parse_args()

    run_training_loop(episodes=args.episodes, checkpoint_path=args.save_path)
