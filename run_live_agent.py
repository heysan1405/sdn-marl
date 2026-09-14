import time
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from agents.sdn_env import SDNCongestionEnv
from agents.dqn_agent import DQNAgent


def run_live_agent_loop(interval=3.0, checkpoint="congestion_agent.npy"):
    print("\n" + "=" * 65)
    print("   LIVE REAL-TIME CONGESTION AGENT (MININET + RYU ACTIVE)")
    print("=" * 65)

    # Note: Requires generic_controller.py running with Ryu!
    print("Connecting to live Ryu Controller monitoring state...")

    # We import or reference the active Ryu controller instance
    # If running as a Ryu app extension or monitoring handle:
    env = SDNCongestionEnv(sampling_interval=interval)

    # Initialize Agent
    # For Abilene (11 switches, 14 links, 28 directed edges)
    state_dim = 58
    action_dim = 29

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, epsilon_start=0.05)

    if os.path.exists(checkpoint):
        try:
            agent.load(checkpoint)
            print(f"✅ Loaded live trained brain from [{checkpoint}]")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint ({e}), running live with fresh policy.")
    else:
        print("ℹ️  No checkpoint found. Running live exploration mode.")

    print("\n[LIVE AGENT STARTED] Monitoring real-time Mininet traffic every 3 seconds...")
    print("Press Ctrl+C to stop.\n")

    step_count = 0

    try:
        while True:
            step_count += 1

            # 1. Read live state vector directly from active network
            state = env.get_state_vector()
            max_util = state[-2] if len(state) > 2 else 0.0
            mean_util = state[-1] if len(state) > 2 else 0.0

            # 2. Query Agent for action
            action = agent.select_action(state, eval_mode=True)

            # If live max utilization exceeds congestion threshold (50%), select bottleneck link action
            if max_util > 0.50 and len(env.link_keys) > 0:
                # Find index of max utilization link
                num_links = len(env.link_keys)
                utils = state[:num_links]
                max_link_idx = int(np.argmax(utils))
                action = max_link_idx + 1  # 1-indexed link penalty action

            print(f"[Step {step_count:4d}] Live Max Link Util: {max_util*100:5.1f}% | Mean Util: {mean_util*100:5.1f}%")

            if action == 0 or max_util <= 0.20:
                print("           -> Action: Maintain Baseline (No heavy congestion)")
            else:
                if len(env.link_keys) >= action:
                    congested_link = env.link_keys[action - 1]
                    print(f"           -> 🚨 CONGESTION DETECTED ({max_util*100:.1f}%) on [{congested_link}]!")
                    print(f"              Pushing Link Weight Multiplier w=10.0 to Ryu Controller -> REROUTING TRAFFIC LIVE!")

            # 3. Apply action to environment / controller
            next_state, reward, done, info = env.step(action)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[LIVE AGENT STOPPED] Exiting live monitoring loop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Real-Time SDN Congestion Agent")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling & decision interval in seconds")
    parser.add_argument("--checkpoint", type=str, default="congestion_agent.npy", help="Path to policy checkpoint")
    args = parser.parse_args()

    run_live_agent_loop(interval=args.interval, checkpoint=args.checkpoint)
