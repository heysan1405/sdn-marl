"""Paired evaluation with diagnostics and explicit agent/orchestrator ablations."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import networkx as nx
import numpy as np

from marl_live.config import RoutingConfig
from marl_live.environment import RoutingEnv
from marl_live.orchestrator import RoutingOrchestrator
from marl_live.policy import RoutingPolicy


POLICIES = (
    "shortest_path", "failure_shortest_path", "congestion_aware_baseline",
    "congestion_policy_only", "failure_policy_only", "marl_orchestrated",
    "coordination_disabled", "failure_agent_disabled", "congestion_agent_disabled",
)


def _fallback_action(obs):
    valid = [i for i, allowed in enumerate(obs["action_mask"]) if bool(allowed)]
    if not valid:
        return None
    features = obs["candidate_features"]
    nodes = obs["candidate_nodes"]
    return max(valid, key=lambda i: (
        float(features[i, 3]), float(features[i, 5]),
        -float(features[i, 0]) if float(features[i, 1]) > .5 else 0.0,
        -int(nodes[i]),
    ))


def _baseline_path(env, obs, weighted=False):
    graph = nx.Graph()
    graph.add_nodes_from(env.graph.nodes)
    for u, v, data in env.graph.edges(data=True):
        if data.get("is_failed", False):
            continue
        if ((u in env.visited and u != env.current) or
                (v in env.visited and v != env.current)):
            continue
        util = float(data["background_mbps"]) / float(data["capacity_mbps"])
        cost = (env.config.hop_cost + env.config.congestion_weight * util) if weighted else 1.0
        graph.add_edge(u, v, weight=cost)
    try:
        return nx.shortest_path(graph, obs["current_node"], obs["destination"], weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _run(env, seed, scenario, policy_name, congestion_policy=None, failure_policy=None,
         cross_link_probability=.25):
    obs = env.reset(seed, scenario=scenario, agent="congestion",
                    cross_link_probability=cross_link_probability)
    path, selected_utils, selected_costs = [obs["current_node"]], [], []
    decision_count = invalid_actions = fallback_actions = 0
    source_counts = {}
    orchestrator = RoutingOrchestrator()
    termination_reason = "unknown"
    final_info = {}
    done = False
    while not done:
        action, source = None, policy_name
        if policy_name in ("shortest_path", "failure_shortest_path", "congestion_aware_baseline"):
            route = _baseline_path(env, obs, weighted=policy_name == "congestion_aware_baseline")
            action = (obs["candidate_nodes"].index(route[1])
                      if route and len(route) > 1 and route[1] in obs["candidate_nodes"] else None)
        elif policy_name == "marl_orchestrated":
            try:
                decision = orchestrator.decide(obs, congestion_policy, failure_policy)
                action, source = decision.action_index, decision.decision_source
            except RuntimeError:
                source = "no_valid_action"
        elif policy_name == "coordination_disabled":
            # Previous dispatch rule, retained as an ablation: failure actor has
            # exclusive control whenever any failure is present.
            chosen = failure_policy if obs["has_failure"] else congestion_policy
            if chosen is not None:
                action, source = chosen.select_action(obs, stochastic=False), (
                    "failure_policy_priority" if obs["has_failure"] else "congestion_policy_only")
        elif policy_name == "failure_agent_disabled":
            if congestion_policy is not None:
                action, source = congestion_policy.select_action(obs, stochastic=False), "congestion_policy_only"
        elif policy_name == "congestion_agent_disabled":
            if obs["has_failure"] and failure_policy is not None:
                action, source = failure_policy.select_action(obs, stochastic=False), "failure_policy_only"
        elif policy_name in ("congestion_policy_only", "failure_policy_only"):
            selected = congestion_policy if policy_name == "congestion_policy_only" else failure_policy
            if selected is not None:
                action, source = selected.select_action(obs, stochastic=False), policy_name
        else:
            raise ValueError(f"unknown policy {policy_name}")

        decision_count += 1
        if (action is None or not isinstance(action, (int, np.integer)) or
                not 0 <= int(action) < len(obs["candidate_nodes"]) or
                not bool(obs["action_mask"][int(action)])):
            invalid_actions += int(action is not None)
            action = _fallback_action(obs)
            if action is not None:
                fallback_actions += 1
                source = "safety_fallback"
            else:
                termination_reason = "no_safe_reachable_action"
                break
        source_counts[source] = source_counts.get(source, 0) + 1
        action = int(action)
        next_node = obs["candidate_nodes"][action]
        util = float(obs["candidate_features"][action, 0])
        selected_utils.append(util)
        selected_costs.append(env.config.hop_cost + env.config.congestion_weight * util)
        obs, _, done, final_info = env.step(action)
        path.append(next_node)
        termination_reason = ("delivered" if final_info["delivered"] else
                              "dead_end_or_hop_limit" if final_info["failed"] else "in_progress")
    delivered = path[-1] == env.destination
    return {
        "seed": int(seed), "scenario": scenario, "policy": policy_name,
        "topology_seed": int(seed), "traffic_seed": int(seed),
        "cross_link_probability": float(cross_link_probability),
        "delivered": int(delivered), "hops": len(path) - 1 if delivered else None,
        "attempted_hops": len(path) - 1, "path": json.dumps(path),
        "loop": int(len(path) != len(set(path))), "invalid_actions": invalid_actions,
        "invalid_action_rate": invalid_actions / decision_count if decision_count else 0.0,
        "fallback_actions": fallback_actions,
        "mean_selected_link_utilization": float(np.mean(selected_utils)) if selected_utils else None,
        "max_selected_link_utilization": max(selected_utils) if selected_utils else None,
        "sum_weighted_path_cost": float(sum(selected_costs)) if selected_costs else None,
        "decisions": decision_count, "decision_sources": json.dumps(source_counts),
        "dead_end": int(not delivered),
        "termination_reason": termination_reason,
    }


def evaluate(congestion_path, failure_path, out_dir, episodes=100, seed=90000,
             cross_link_probability=.25):
    congestion = RoutingPolicy.load(congestion_path)
    failure = RoutingPolicy.load(failure_path)
    rows = []
    for scenario in RoutingEnv.SCENARIOS:
        for offset in range(episodes):
            case_seed = seed + offset
            for method in POLICIES:
                rows.append(_run(RoutingEnv(), case_seed, scenario, method,
                                 congestion, failure, cross_link_probability))
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "evaluation_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    groups = {}
    for scenario in RoutingEnv.SCENARIOS:
        for method in POLICIES:
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == method]
            hops = [r["hops"] for r in group if r["hops"] is not None]
            util = [r["mean_selected_link_utilization"] for r in group
                    if r["mean_selected_link_utilization"] is not None]
            groups[f"{scenario}/{method}"] = {
                "n": len(group), "deliveries": sum(r["delivered"] for r in group),
                "delivery_rate": statistics.mean(r["delivered"] for r in group),
                "mean_hops_delivered": statistics.mean(hops) if hops else None,
                "mean_selected_background_utilization": statistics.mean(util) if util else None,
                "loop_count": sum(r["loop"] for r in group),
                "invalid_actions": sum(r["invalid_actions"] for r in group),
                "fallback_actions": sum(r["fallback_actions"] for r in group),
                "dead_ends": sum(r["dead_end"] for r in group),
                "mean_control_decisions": statistics.mean(r["decisions"] for r in group),
            }
    (out_dir / "evaluation_summary.json").write_text(json.dumps({
        "seed_start": seed, "episodes_per_scenario": episodes,
        "cross_link_probability": cross_link_probability,
        "measurement_class": "generated NetworkX simulation; not live Mininet telemetry",
        "paired_scenarios": "all policies reset with identical topology/traffic/failure seeds",
        "metric_semantics": "mean background TX-equivalent utilization along the selected route; not post-admission utilization or network-wide congestion",
        "metrics_unavailable": ["throughput", "end-to-end delay", "packet loss", "real recovery time"],
        "baseline_information": "weighted baseline sees the full generated graph and per-edge utilization; policies see local candidate features plus hop/cost-to-go potentials",
        "groups": groups,
    }, indent=2) + "\n", encoding="utf-8")
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--congestion-policy", required=True)
    p.add_argument("--failure-policy", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=90000)
    p.add_argument("--cross-link-probability", type=float, default=.25)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.episodes <= 0:
        p.error("--episodes must be positive")
    evaluate(a.congestion_policy, a.failure_policy, a.out, a.episodes, a.seed,
             a.cross_link_probability)


if __name__ == "__main__":
    main()
