"""Seeded graph routing environment with candidate-level live-compatible state."""
from __future__ import annotations

import random
from typing import Optional

import networkx as nx
import numpy as np

from marl_live.config import RoutingConfig
from marl_live.features import candidate_observation


BRANCHES = ((0, 1, 4, 7), (0, 2, 5, 7), (0, 3, 6, 7))


def make_training_graph(seed: int, cross_link_probability: float = 0.25) -> nx.Graph:
    """Create connected parallel-path graphs with seed-varying capacities."""
    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    for route in BRANCHES:
        graph.add_edges_from(zip(route, route[1:]))
    # Cross links vary candidate degree and path geometry without changing IDs in features.
    for u in range(1, 7):
        for v in range(u + 1, 7):
            if abs(u - v) > 1 and rng.random() < cross_link_probability:
                graph.add_edge(u, v)
    for u, v in graph.edges:
        graph[u][v].update(capacity_mbps=rng.choice((50.0, 75.0, 100.0, 150.0, 200.0)),
                           background_mbps=0.0, is_failed=False)
    return graph


class RoutingEnv:
    """One episode routes a fixed offered flow from node 0 to node 7, hop by hop.

    Every ``step(action)`` advances exactly one switch-to-switch forwarding
    decision. Link counters are simulator-generated in this environment. Live
    inference consumes the same fields from OpenFlow telemetry.
    """
    SCENARIOS = ("normal", "congestion", "failure", "combined")

    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self.graph = None
        self.current = self.destination = None
        self.path = []
        self.visited = set()
        self.step_count = 0
        self.scenario = "normal"
        self.agent = "congestion"
        self.done = False
        self.failed_edges = set()
        self.seed = None

    @staticmethod
    def _edge_key(u, v):
        return tuple(sorted((u, v)))

    def reset(self, seed: int, scenario="normal", agent="congestion", graph=None,
              cross_link_probability=0.25):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}")
        if agent not in ("congestion", "failure"):
            raise ValueError("agent must be 'congestion' or 'failure'")
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        if not 0.0 <= cross_link_probability <= 1.0:
            raise ValueError("cross_link_probability must be in [0, 1]")
        self.cross_link_probability = float(cross_link_probability)
        self.graph = graph.copy() if graph is not None else make_training_graph(
            self.seed, self.cross_link_probability)
        self.scenario, self.agent = scenario, agent
        self.current, self.destination = 0, 7
        self.path, self.visited = [0], {0}
        self.step_count, self.done, self.failed_edges = 0, False, set()
        for _, _, data in self.graph.edges(data=True):
            data["background_mbps"] = 0.0
            data["is_failed"] = False
        # Three generated background demands share the same OD and are placed on
        # separate routes. These are simulator values, not CAIDA measurements.
        branch_loads = [self.rng.uniform(4.0, 22.0) for _ in BRANCHES]
        if scenario in ("congestion", "combined"):
            hot_branch = self.rng.randrange(len(BRANCHES))
            branch_loads[hot_branch] += self.rng.uniform(100.0, 190.0)
        for route, load in zip(BRANCHES, branch_loads):
            for u, v in zip(route, route[1:]):
                self.graph[u][v]["background_mbps"] += load
        if scenario in ("failure", "combined"):
            branch = self.rng.randrange(len(BRANCHES))
            route = BRANCHES[branch]
            u, v = list(zip(route, route[1:]))[self.rng.randrange(len(route) - 1)]
            self.graph[u][v]["is_failed"] = True
            self.failed_edges.add(self._edge_key(u, v))
        return self.observe()

    def observe(self):
        if self.graph is None:
            raise RuntimeError("reset() must be called before observe()")
        observation = candidate_observation(
            self.graph, self.current, self.destination, self.visited,
            has_failure=bool(self.failed_edges),
            capacity_reference_mbps=self.config.capacity_reference_mbps,
            hop_cost=self.config.hop_cost,
            congestion_weight=self.config.congestion_weight)
        if self.step_count >= self.config.max_hops:
            observation["action_mask"][:] = False
        observation["scenario"] = self.scenario
        return observation

    def step(self, action: int):
        if self.graph is None or self.done:
            raise RuntimeError("episode is not active")
        observation = self.observe()
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(observation["candidate_nodes"]):
            raise ValueError("action index is outside the current candidate list")
        action = int(action)
        if not observation["action_mask"][action]:
            raise ValueError("action selects a failed, visited, or masked next hop")
        nxt = observation["candidate_nodes"][action]
        edge = self.graph[self.current][nxt]
        utilization = edge["background_mbps"] / edge["capacity_mbps"]
        reward = -self.config.hop_penalty
        if self.agent == "congestion":
            # Same additive path objective as the non-RL weighted-Dijkstra
            # baseline. No demand is dropped or changed to improve reward.
            reward = -(self.config.hop_cost + self.config.congestion_weight * utilization)
        self.current = nxt
        self.path.append(nxt)
        self.visited.add(nxt)
        self.step_count += 1
        delivered = nxt == self.destination
        no_actions = not any(self.observe()["action_mask"]) if not delivered else False
        timeout = self.step_count >= self.config.max_hops
        failed = bool(no_actions or timeout) and not delivered
        self.done = delivered or failed
        if delivered:
            reward += self.config.delivery_reward
        elif failed:
            reward -= self.config.failure_penalty
        info = {
            "delivered": delivered,
            "failed": failed,
            "path": list(self.path),
            "selected_link_utilization": float(utilization),
            "selected_edge_cost": float(self.config.hop_cost + self.config.congestion_weight * utilization),
            "invalid_action": False,
            "loop": len(self.path) != len(set(self.path)),
            "scenario": self.scenario,
        }
        return self.observe(), float(reward), self.done, info
