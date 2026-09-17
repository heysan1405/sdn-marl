"""
Dummy two-agent SDN environment.

This is only for testing the Person 2 agents before the
real simulator from Person 1 is available.
"""

import random

import numpy as np


class DummySDNMultiAgentEnvironment:

    def __init__(
        self,
        min_nodes=5,
        max_nodes=8,
        max_steps=10,
        seed=None,
    ):

        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_steps = max_steps

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.current_step = 0
        self.num_nodes = None

        self.node_features = None
        self.edge_index = None
        self.edge_features = None
        self.demand_features = None

    def reset(self):

        self.current_step = 0

        self.num_nodes = self.rng.randint(
            self.min_nodes,
            self.max_nodes,
        )

        self._build_topology()
        self._generate_demand()

        return self._get_state()

    def step(
        self,
        congestion_action,
        failure_action,
    ):

        if self.node_features is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        if congestion_action not in [0, 1, 2]:
            raise ValueError(
                "Invalid congestion action."
            )

        if failure_action not in [0, 1, 2]:
            raise ValueError(
                "Invalid failure action."
            )

        reward = 0.0

        average_utilization = float(
            np.mean(
                self.edge_features[:, 0]
            )
        )

        average_packet_loss = float(
            np.mean(
                self.edge_features[:, 3]
            )
        )

        failed_links = float(
            np.sum(
                self.edge_features[:, 4]
            )
        )

        # ================================================
        # Congestion Agent
        # ================================================

        if congestion_action == 0:
            # Maintain
            reward += 0.2

        elif congestion_action == 1:
            # Reroute
            reward += 0.6

            self.edge_features[:, 0] *= 0.92

        elif congestion_action == 2:
            # Rate-limit
            reward += 0.4

            self.demand_features[0, 0] *= 0.95

        # ================================================
        # Failure Agent
        # ================================================

        if failure_action == 0:
            # No action
            reward += 0.1

        elif failure_action == 1:
            # Reroute
            reward += 0.5

            self.edge_features[:, 0] *= 0.90

        elif failure_action == 2:
            # Isolate failed components
            reward += 0.3

            self.edge_features[:, 4] = 0.0

        # ================================================
        # Network penalty
        # ================================================

        reward -= average_utilization

        reward -= average_packet_loss

        reward -= 0.2 * failed_links

        # ================================================
        # Random traffic/network dynamics
        # ================================================

        noise = self.np_rng.normal(
            0.0,
            0.01,
            size=self.edge_features[:, 0].shape,
        )

        self.edge_features[:, 0] += noise

        self.edge_features[:, 0] = np.clip(
            self.edge_features[:, 0],
            0.0,
            1.0,
        )

        self.current_step += 1

        done = (
            self.current_step
            >= self.max_steps
        )

        info = {
            "average_utilization":
                average_utilization,

            "average_packet_loss":
                average_packet_loss,

            "failed_links":
                int(failed_links),

            "congestion_action":
                congestion_action,

            "failure_action":
                failure_action,
        }

        return (
            self._get_state(),
            float(reward),
            done,
            info,
        )

    def _build_topology(self):

        n = self.num_nodes

        edges = []

        # Chain topology
        for i in range(n - 1):

            edges.append(
                [i, i + 1]
            )

            edges.append(
                [i + 1, i]
            )

        self.edge_index = np.array(
            edges,
            dtype=np.int64,
        ).T

        edge_count = len(edges)

        # ================================================
        # Node features
        # ================================================

        self.node_features = np.zeros(
            (n, 7),
            dtype=np.float32,
        )

        for node in range(n):

            degree = sum(
                1
                for edge in edges
                if edge[0] == node
            )

            self.node_features[node] = [
                float(degree),
                self.rng.uniform(
                    0.1,
                    0.8,
                ),
                node / max(n - 1, 1),
                0.0,
                1.0 if node == 0 else 0.0,
                1.0 if node == n - 1 else 0.0,
                0.0,
            ]

        # ================================================
        # Edge features
        # ================================================

        self.edge_features = np.zeros(
            (edge_count, 5),
            dtype=np.float32,
        )

        for i in range(edge_count):

            self.edge_features[i] = [
                self.rng.uniform(
                    0.1,
                    0.7,
                ),
                100.0,
                self.rng.uniform(
                    5.0,
                    15.0,
                ),
                self.rng.uniform(
                    0.0,
                    0.03,
                ),
                0.0,
            ]

    def _generate_demand(self):

        self.demand_features = np.array(
            [
                [
                    self.rng.uniform(
                        10.0,
                        50.0,
                    )
                ]
            ],
            dtype=np.float32,
        )

    def _get_state(self):

        return {
            "node_features":
                self.node_features.copy(),

            "edge_index":
                self.edge_index.copy(),

            "edge_features":
                self.edge_features.copy(),

            "demand_features":
                self.demand_features.copy(),
        }