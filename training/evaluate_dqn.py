"""Fresh-process evaluation for the graph DQN checkpoints."""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.congestion_agent import CongestionAgent
from agents.failure_agent import FailureAgent
from agents.orchestrator import RuleBasedOrchestrator
from simulator.environment import SDNEnvironment
from training.config import (
    AGENT_HIDDEN_DIM, EDGE_FEATURE_DIM, GNN_DROPOUT, GNN_HEADS,
    GNN_HIDDEN_DIM, GNN_LAYERS, GRAPH_EMBEDDING_DIM,
)


def create_agent(agent_type, device):
    cls = CongestionAgent if agent_type == "congestion" else FailureAgent
    return cls(
        node_feature_dim=7, edge_feature_dim=EDGE_FEATURE_DIM,
        demand_feature_dim=1, gnn_hidden_dim=GNN_HIDDEN_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM, gnn_layers=GNN_LAYERS,
        gnn_heads=GNN_HEADS, gnn_dropout=GNN_DROPOUT,
        hidden_dim=AGENT_HIDDEN_DIM, device=device,
    )


def used_link_to_fail(env):
    uses = {}
    for path in env.simulator.active_paths.values():
        for edge in zip(path[:-1], path[1:]):
            key = tuple(sorted(edge))
            uses[key] = uses.get(key, 0) + 1
    if not uses:
        raise RuntimeError("No routed traffic exists for failure injection")
    u, v = min(uses, key=lambda e: (-uses[e], e))
    return u, v


def run_episode(topology, scenario, seed, method, congestion_agent=None,
                failure_agent=None, inject_failure=False, num_demands=12,
                traffic_scale=1.0, traffic_scenario=None):
    started = time.perf_counter()
    env = SDNEnvironment(
        topology_path=topology, traffic_scenario=traffic_scenario or scenario,
        num_demands=num_demands, max_steps=num_demands, k_paths=5,
        random_seed=seed,
        traffic_scale=traffic_scale,
    )
    state = env.reset(seed=seed)
    failed_link = None
    if inject_failure:
        failed_link = used_link_to_fail(env)
        env.inject_link_failure(*failed_link)
        state = env._build_state()
        if method == "shortest_path":
            env.simulator.reroute_all_demands(
                congestion_aware=False, require_alternative=False
            )
            env._refresh_simulator_state()
            state = env._build_state()

    coordinator = RuleBasedOrchestrator()
    done = False
    rewards = []
    def delivered_demand_count():
        return sum(
            1 for demand_id, path in env.simulator.active_paths.items()
            if env.simulator._path_is_usable(path)
        )
    initial_delivered = delivered_demand_count()
    recovery_step = 0 if initial_delivered == len(env.simulator.demands) else None
    unserved_demand_steps = len(env.simulator.demands) - initial_delivered
    action_counts = {"congestion": [0, 0, 0], "failure": [0, 0, 0]}
    overrides = 0
    while not done:
        c_action = congestion_agent.select_action(state, training=False) if congestion_agent and method in {"congestion_dqn", "orchestrated"} else 0
        f_action = failure_agent.select_action(state, training=False) if failure_agent and method in {"failure_dqn", "orchestrated"} else 0
        if method == "orchestrated":
            decision = coordinator.resolve(state, c_action, f_action)
            if (decision.congestion_action, decision.failure_action) != (c_action, f_action):
                overrides += 1
            c_action, f_action = decision.congestion_action, decision.failure_action
        action_counts["congestion"][c_action] += 1
        action_counts["failure"][f_action] += 1
        state, reward, done, _ = env.step(c_action, f_action)
        rewards.append(float(reward))
        delivered_now = delivered_demand_count()
        unserved_demand_steps += len(env.simulator.demands) - delivered_now
        if recovery_step is None and delivered_now == len(env.simulator.demands):
            recovery_step = len(rewards)

    metrics = env.simulator.calculate_network_metrics()
    paths = metrics.get("path_statistics", [])
    delivered_paths = []
    for demand_id in range(len(env.simulator.demands)):
        path = env.simulator.active_paths.get(demand_id)
        if path and env.simulator._path_is_usable(path):
            delivered_paths.append(path)
    loop_count = sum(len(path) != len(set(path)) for path in delivered_paths)
    result = {
        "scenario": scenario, "seed": seed, "method": method,
        "failure_injected": inject_failure,
        "traffic_scale": traffic_scale,
        "failed_link": str(failed_link) if failed_link else "",
        "steps": len(rewards), "total_reward": sum(rewards),
        "runtime_seconds": time.perf_counter() - started,
        "recovery_step": recovery_step,
        "failure_detection_step": 0 if inject_failure else None,
        "recovery_time_steps": recovery_step,
        "unserved_demand_steps": unserved_demand_steps,
        "mean_step_reward": statistics.mean(rewards) if rewards else None,
        "requested_mbps": metrics["total_requested_traffic_mbps"],
        "throughput_mbps": metrics["total_throughput_mbps"],
        "delivery_rate": len(delivered_paths) / max(1, metrics["total_demands"]),
        "mean_link_utilization": metrics["average_link_utilization"],
        "congestion_ratio": metrics["congestion_ratio"],
        "mean_delay_ms": metrics["average_path_delay_ms"],
        "packet_loss": metrics["average_packet_loss"],
        "mean_hops": statistics.mean(len(path) - 1 for path in delivered_paths) if delivered_paths else None,
        "loops": loop_count, "failed_demands": metrics["total_demands"] - len(delivered_paths),
        "congestion_actions": str(action_counts["congestion"]),
        "failure_actions": str(action_counts["failure"]),
        "orchestrator_overrides": overrides,
    }
    return result


def summarize(rows):
    keys = ["delivery_rate", "throughput_mbps", "mean_link_utilization",
            "congestion_ratio", "mean_delay_ms", "packet_loss", "mean_hops",
            "total_reward", "loops", "failed_demands", "recovery_step",
            "unserved_demand_steps", "orchestrator_overrides", "runtime_seconds"]
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        entry = {"method": method, "runs": len(selected)}
        for key in keys:
            values = [row[key] for row in selected if row[key] is not None]
            entry[key + "_mean"] = statistics.mean(values) if values else None
            entry[key + "_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        summaries.append(entry)
    return summaries


def summarize_by_scenario(rows):
    result = []
    for scenario in sorted({row["scenario"] for row in rows}):
        result.extend({"scenario": scenario, **item}
                      for item in summarize([row for row in rows if row["scenario"] == scenario]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--congestion-checkpoint", required=True)
    parser.add_argument("--failure-checkpoint", required=True)
    parser.add_argument("--topology", default="InternetTopologyZoo/gml/Abilene.gml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--demands", type=int, default=12)
    parser.add_argument("--traffic-scale", type=float, default=1.0,
                        help="Explicit simulator-generated traffic multiplier.")
    args = parser.parse_args()
    if args.demands < 1 or not np.isfinite(args.traffic_scale) or args.traffic_scale <= 0:
        parser.error("demands and traffic-scale must be positive (traffic-scale finite)")
    for path in (args.congestion_checkpoint, args.failure_checkpoint, args.topology):
        if not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    if os.path.exists(args.output):
        parser.error(f"refusing to overwrite existing output: {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    congestion = create_agent("congestion", device)
    failure = create_agent("failure", device)
    congestion.load(args.congestion_checkpoint)
    failure.load(args.failure_checkpoint)
    congestion.epsilon = failure.epsilon = 0.0
    congestion.online_gnn.eval(); congestion.online_network.eval()
    failure.online_gnn.eval(); failure.online_network.eval()
    # Explicit first-run inference check, after loading both files in this process.
    probe_env = SDNEnvironment(args.topology, num_demands=1, max_steps=1, random_seed=args.seeds[0])
    probe_state = probe_env.reset(seed=args.seeds[0])
    probe = {"congestion_action": congestion.select_action(probe_state, training=False),
             "failure_action": failure.select_action(probe_state, training=False)}

    rows = []
    # Identical seed/scenario conditions are replayed from fresh environment
    # resets for each baseline/policy. The traffic multiplier is simulator
    # generated stress, never presented as CAIDA measurement.
    scenarios = [
        ("normal", "normal", 1.0, False),
        ("heavy_congestion", "heavy", args.traffic_scale, False),
        ("link_failure", "heavy", 1.0, True),
        ("combined_congestion_failure", "heavy", args.traffic_scale, True),
    ]
    methods = ("no_agent", "shortest_path", "congestion_dqn", "failure_dqn", "orchestrated")
    for scenario, traffic_scenario, traffic_scale, inject_failure in scenarios:
        for seed in args.seeds:
            for method in methods:
                rows.append(run_episode(args.topology, scenario, seed, method,
                                        congestion_agent=congestion, failure_agent=failure,
                                        inject_failure=inject_failure,
                                        num_demands=args.demands,
                                        traffic_scale=traffic_scale,
                                        traffic_scenario=traffic_scenario))
    report = {
        "topology": args.topology, "device": str(device), "seeds": args.seeds,
        "demands_per_episode": args.demands,
        "traffic_scale": args.traffic_scale,
        "checkpoint_learning_updates": {"congestion": congestion.learn_steps, "failure": failure.learn_steps},
        "fresh_process_inference_probe": probe,
        "scenarios": [{"name": name, "traffic_generator_scenario": traffic,
                       "traffic_scale": scale, "failure_injected": failed}
                      for name, traffic, scale, failed in scenarios],
        "metrics_are_synthetic_network_simulator_outputs": True,
        "runs": rows, "summary": summarize(rows),
        "scenario_summary": summarize_by_scenario(rows),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    csv_path = str(Path(args.output).with_suffix(".csv"))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output": args.output, "csv": csv_path,
                      "inference_probe": probe, "summary": report["scenario_summary"]}, indent=2))


if __name__ == "__main__":
    main()
