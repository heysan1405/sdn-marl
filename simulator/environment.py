"""
SDN MARL Environment

Integration layer between:
    - TopologyLoader
    - TrafficGenerator
    - RoutingEngine
    - NetworkSimulator
    - RewardCalculator

State returned to the GNN/RL agents:

    node_features   [N, 7]
        [degree, node_load, normalized_x, normalized_y,
         is_source, is_destination, is_failed]

    edge_index      [2, E]

    edge_features   [E, 5]
        [utilization, capacity, delay, packet_loss, is_failed]

    demand_features [1, 1]
        [traffic_volume]

Actions:

    Congestion Agent:
        0 = maintain
        1 = reroute
        2 = rate-limit

    Failure Agent:
        0 = no action
        1 = reroute
        2 = isolate failed component
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .topology_loader import TopologyLoader
from .traffic_generator import TrafficGenerator
from .routing import RoutingEngine
from .network_simulator import NetworkSimulator
from .reward import RewardCalculator


class SDNEnvironment:
    """
    Environment used by the congestion and failure agents.

    One demand is treated as the current active demand for the
    source/destination node features.

    All demands remain installed in the network, so the graph state
    still reflects network-wide congestion, utilization, delay,
    packet loss and failures.
    """

    # ==============================================================
    # Congestion agent actions
    # ==============================================================

    CONGESTION_MAINTAIN = 0
    CONGESTION_REROUTE = 1
    CONGESTION_RATE_LIMIT = 2

    # ==============================================================
    # Failure agent actions
    # ==============================================================

    FAILURE_NO_ACTION = 0
    FAILURE_REROUTE = 1
    FAILURE_ISOLATE = 2

    # ==============================================================
    # Feature dimensions
    # ==============================================================

    NUM_NODE_FEATURES = 7
    NUM_EDGE_FEATURES = 5

    def __init__(
        self,
        topology_path: str,
        traffic_scenario: str = "normal",
        num_demands: int = 10,
        max_steps: Optional[int] = None,
        k_paths: int = 5,
        random_seed: Optional[int] = 42,
        rate_limit_factor: float = 0.70,
    ):
        """
        Parameters
        ----------
        topology_path:
            Path to a Topology Zoo GML file.

        traffic_scenario:
            Traffic scenario:
                normal
                heavy
                single_hotspot
                multiple_hotspots
                traffic_spike
                random

        num_demands:
            Number of traffic demands generated per episode.

        max_steps:
            Maximum number of environment steps.
            Defaults to num_demands.

        k_paths:
            Number of candidate paths.

        random_seed:
            Random seed for reproducibility.

        rate_limit_factor:
            Traffic multiplier when rate-limit is selected.
            Example: 0.70 means traffic becomes 70% of its
            previous value.
        """

        self.topology_path = os.path.abspath(topology_path)

        self.traffic_scenario = traffic_scenario
        self.num_demands = int(num_demands)

        self.max_steps = (
            int(max_steps)
            if max_steps is not None
            else self.num_demands
        )

        self.k_paths = int(k_paths)

        self.random_seed = random_seed

        self.rate_limit_factor = float(
            rate_limit_factor
        )

        if self.num_demands <= 0:
            raise ValueError(
                "num_demands must be greater than zero"
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero"
            )

        if not 0.0 < self.rate_limit_factor <= 1.0:
            raise ValueError(
                "rate_limit_factor must be in (0, 1]"
            )

        if not os.path.isfile(self.topology_path):
            raise FileNotFoundError(
                f"Topology file not found: "
                f"{self.topology_path}"
            )

        # ----------------------------------------------------------
        # Random number generators
        # ----------------------------------------------------------

        self.rng = random.Random(
            random_seed
        )

        self.np_rng = np.random.default_rng(
            random_seed
        )

        # ----------------------------------------------------------
        # Core components
        # ----------------------------------------------------------

        self.topology_loader = TopologyLoader()

        # TrafficGenerator requires the graph in its constructor,
        # therefore it is created during reset().
        self.traffic_generator = None

        self.routing = None

        self.simulator = None

        self.reward_calculator = RewardCalculator()

        # ----------------------------------------------------------
        # Runtime state
        # ----------------------------------------------------------

        self.graph: Optional[nx.Graph] = None

        self.demands: List[Dict[str, Any]] = []

        self.current_demand_index = 0

        self.current_step = 0

        self.done = False

        self.current_state = None

        self.last_result = None

        self.last_reward_details = None

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(
        self,
        traffic_scenario: Optional[str] = None,
        num_demands: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Start a new episode.

        Returns
        -------
        dict
            {
                "node_features": [N, 7],
                "edge_index": [2, E],
                "edge_features": [E, 5],
                "demand_features": [1, 1]
            }
        """

        # ----------------------------------------------------------
        # Update seed if provided
        # ----------------------------------------------------------

        if seed is not None:
            self.random_seed = seed

            self.rng = random.Random(
                seed
            )

            self.np_rng = np.random.default_rng(
                seed
            )

        scenario = (
            traffic_scenario
            if traffic_scenario is not None
            else self.traffic_scenario
        )

        demand_count = (
            int(num_demands)
            if num_demands is not None
            else self.num_demands
        )

        if demand_count <= 0:
            raise ValueError(
                "num_demands must be greater than zero"
            )

        # ----------------------------------------------------------
        # Load topology
        # ----------------------------------------------------------

        self.graph = self.topology_loader.load(
            self.topology_path
        )

        # ----------------------------------------------------------
        # Create traffic generator
        # ----------------------------------------------------------

        self.traffic_generator = TrafficGenerator(
            graph=self.graph,
            seed=self.random_seed,
        )

        # ----------------------------------------------------------
        # Create routing engine
        # ----------------------------------------------------------

        self.routing = RoutingEngine(
            self.graph,
            k_paths=self.k_paths,
        )

        # ----------------------------------------------------------
        # Create network simulator
        # ----------------------------------------------------------

        self.simulator = NetworkSimulator(
            self.graph,
            k_paths=self.k_paths,
            random_seed=self.random_seed,
        )

        # ----------------------------------------------------------
        # Generate traffic
        # ----------------------------------------------------------

        self.demands = self._generate_demands(
            scenario=scenario,
            num_demands=demand_count,
        )

        if len(self.demands) == 0:
            raise RuntimeError(
                "Traffic generator produced no demands"
            )

        # ----------------------------------------------------------
        # Install traffic into simulator
        # ----------------------------------------------------------

        self.simulator.install_demands(
            self.demands
        )

        # ----------------------------------------------------------
        # Reset episode counters
        # ----------------------------------------------------------

        self.current_demand_index = 0

        self.current_step = 0

        self.done = False

        # ----------------------------------------------------------
        # Calculate initial metrics
        # ----------------------------------------------------------

        self.last_result = (
            self.simulator.calculate_network_metrics()
        )

        self.last_reward_details = None

        # ----------------------------------------------------------
        # Build initial state
        # ----------------------------------------------------------

        self.current_state = self._build_state()

        return self.current_state

    # ==============================================================
    # STEP
    # ==============================================================

    def step(
        self,
        congestion_action: int = 0,
        failure_action: int = 0,
    ) -> Tuple[
        Dict[str, np.ndarray],
        float,
        bool,
        Dict[str, Any],
    ]:
        """
        Execute one environment step.

        Parameters
        ----------
        congestion_action:
            0 = maintain
            1 = reroute
            2 = rate-limit

        failure_action:
            0 = no action
            1 = reroute
            2 = isolate

        Returns
        -------
        next_state
        reward
        done
        info
        """

        if self.simulator is None:
            raise RuntimeError(
                "Environment has not been reset. "
                "Call reset() first."
            )

        if self.done:
            raise RuntimeError(
                "Episode is already done. "
                "Call reset() before step()."
            )

        self._validate_actions(
            congestion_action,
            failure_action,
        )

        current_demand = self._get_current_demand()

        # ----------------------------------------------------------
        # Apply congestion agent action
        # ----------------------------------------------------------

        congestion_info = (
            self._apply_congestion_action(
                congestion_action,
                current_demand,
            )
        )

        # ----------------------------------------------------------
        # Apply failure agent action
        # ----------------------------------------------------------

        failure_info = (
            self._apply_failure_action(
                failure_action,
                current_demand,
            )
        )

        # ----------------------------------------------------------
        # Refresh simulator state
        # ----------------------------------------------------------

        self._refresh_simulator_state()

        # ----------------------------------------------------------
        # Calculate network metrics
        # ----------------------------------------------------------

        self.last_result = (
            self.simulator.calculate_network_metrics()
        )

        # ----------------------------------------------------------
        # Calculate reward
        # ----------------------------------------------------------

        reward = self.reward_calculator.calculate(
            self.last_result
        )

        self.last_reward_details = (
            self.reward_calculator.calculate_detailed(
                self.last_result
            )
        )

        # ----------------------------------------------------------
        # Advance episode
        # ----------------------------------------------------------

        previous_demand_index = (
            self.current_demand_index
        )

        self.current_step += 1

        self.current_demand_index += 1

        # ----------------------------------------------------------
        # Check termination
        # ----------------------------------------------------------

        if self.current_step >= self.max_steps:
            self.done = True

        if (
            self.current_demand_index
            >= len(self.demands)
        ):
            self.done = True

        # ----------------------------------------------------------
        # Build next state
        # ----------------------------------------------------------

        self.current_state = (
            self._build_state()
        )

        # ----------------------------------------------------------
        # Information returned to RL trainer
        # ----------------------------------------------------------

        info = {
            "step": self.current_step,

            "previous_demand_index":
                previous_demand_index,

            "current_demand_index":
                min(
                    self.current_demand_index,
                    len(self.demands) - 1,
                ),

            "congestion_action":
                int(congestion_action),

            "failure_action":
                int(failure_action),

            "congestion_action_info":
                congestion_info,

            "failure_action_info":
                failure_info,

            "metrics":
                self.last_result,

            "reward_components":
                self.last_reward_details,

            "failed_nodes":
                self.simulator.get_failed_nodes(),

            "failed_links":
                self.simulator.get_failed_links(),

            "current_demand":
                dict(current_demand),
        }

        return (
            self.current_state,
            float(reward),
            self.done,
            info,
        )

    # ==============================================================
    # TRAFFIC GENERATION
    # ==============================================================

    def _generate_demands(
        self,
        scenario: str,
        num_demands: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate demands using the project's
        TrafficGenerator API.
        """

        if self.graph is None:
            raise RuntimeError(
                "Graph is not loaded"
            )

        if self.traffic_generator is None:
            self.traffic_generator = (
                TrafficGenerator(
                    graph=self.graph,
                    seed=self.random_seed,
                )
            )

        demands = (
            self.traffic_generator.generate(
                scenario=scenario,
                num_demands=num_demands,
            )
        )

        return [
            self._normalize_demand(demand)
            for demand in demands
        ]

    @staticmethod
    def _normalize_demand(
        demand: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ensure every traffic demand has the format
        expected by NetworkSimulator.
        """

        if "src" not in demand:
            raise KeyError(
                "Traffic demand is missing 'src'"
            )

        if "dst" not in demand:
            raise KeyError(
                "Traffic demand is missing 'dst'"
            )

        if "traffic_mbps" not in demand:
            raise KeyError(
                "Traffic demand is missing "
                "'traffic_mbps'"
            )

        return {
            "src": demand["src"],
            "dst": demand["dst"],
            "traffic_mbps": float(
                demand["traffic_mbps"]
            ),
        }

    # ==============================================================
    # CONGESTION ACTION
    # ==============================================================

    def _apply_congestion_action(
        self,
        action: int,
        demand: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply congestion agent action.
        """

        # ----------------------------------------------------------
        # 0 = maintain
        # ----------------------------------------------------------

        if action == self.CONGESTION_MAINTAIN:
            return {
                "action": "maintain",
                "applied": True,
            }

        demand_id = (
            self.current_demand_index
        )

        # ----------------------------------------------------------
        # 1 = reroute
        # ----------------------------------------------------------

        if action == self.CONGESTION_REROUTE:

            success = (
                self.simulator.reroute_demand(
                    demand_id
                )
            )

            if success:
                self._refresh_simulator_state()

            return {
                "action": "reroute",
                "applied": bool(success),
                "demand_id": demand_id,
            }

        # ----------------------------------------------------------
        # 2 = rate-limit
        # ----------------------------------------------------------

        if action == self.CONGESTION_RATE_LIMIT:

            old_rate = float(
                demand["traffic_mbps"]
            )

            new_rate = (
                old_rate
                * self.rate_limit_factor
            )

            demand["traffic_mbps"] = (
                new_rate
            )

            # Reinstall the complete demand set so the
            # simulator recalculates traffic using the
            # modified current demand.
            self.simulator.install_demands(
                self.demands
            )

            return {
                "action": "rate-limit",
                "applied": True,
                "demand_id": demand_id,
                "old_traffic_mbps": old_rate,
                "new_traffic_mbps": new_rate,
                "factor":
                    self.rate_limit_factor,
            }

        raise ValueError(
            f"Unknown congestion action: {action}"
        )

    # ==============================================================
    # FAILURE ACTION
    # ==============================================================

    def _apply_failure_action(
        self,
        action: int,
        demand: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply failure agent action.
        """

        # ----------------------------------------------------------
        # 0 = no action
        # ----------------------------------------------------------

        if action == self.FAILURE_NO_ACTION:
            return {
                "action": "no_action",
                "applied": True,
            }

        demand_id = (
            self.current_demand_index
        )

        # ----------------------------------------------------------
        # 1 = reroute
        # ----------------------------------------------------------

        if action == self.FAILURE_REROUTE:

            success = (
                self.simulator.reroute_demand(
                    demand_id
                )
            )

            if success:
                self._refresh_simulator_state()

            return {
                "action": "reroute",
                "applied": bool(success),
                "demand_id": demand_id,
            }

        # ----------------------------------------------------------
        # 2 = isolate failed components
        # ----------------------------------------------------------

        if action == self.FAILURE_ISOLATE:

            result = (
                self.simulator
                .isolate_failed_components()
            )

            return {
                "action": "isolate",
                "applied": True,
                "result": result,
            }

        raise ValueError(
            f"Unknown failure action: {action}"
        )

    # ==============================================================
    # SIMULATOR REFRESH
    # ==============================================================

    def _refresh_simulator_state(self) -> None:
        """
        Recalculate traffic-dependent simulator state.

        NetworkSimulator.reroute_demand() updates the selected
        active path but does not itself call _apply_all_traffic().
        """

        if self.simulator is None:
            return

        # The simulator's own public operations already update
        # state when required. Here we refresh traffic explicitly
        # after path changes.
        self.simulator._apply_all_traffic()

    # ==============================================================
    # STATE CONSTRUCTION
    # ==============================================================

    def _build_state(
        self,
    ) -> Dict[str, np.ndarray]:
        """
        Build the graph state expected by the GNN.

        Returns:

            node_features:
                [N, 7]

            edge_index:
                [2, E]

            edge_features:
                [E, 5]

            demand_features:
                [1, 1]
        """

        if (
            self.graph is None
            or self.simulator is None
        ):
            raise RuntimeError(
                "Environment is not initialized"
            )

        demand = (
            self._get_current_demand()
        )

        nodes = list(
            self.graph.nodes()
        )

        node_to_index = {
            node: index
            for index, node in enumerate(nodes)
        }

        # ==========================================================
        # NODE FEATURES
        # ==========================================================

        max_degree = max(
            (
                self.graph.degree(node)
                for node in nodes
            ),
            default=1,
        )

        failed_nodes = set(
            self.simulator.get_failed_nodes()
        )

        node_loads = (
            self._get_node_loads()
        )

        coordinates = (
            self._get_node_coordinates(
                nodes
            )
        )

        node_features = []

        for node in nodes:

            # ----------------------------------------------
            # Degree
            # ----------------------------------------------

            if max_degree > 0:
                normalized_degree = (
                    float(
                        self.graph.degree(node)
                    )
                    / float(max_degree)
                )
            else:
                normalized_degree = 0.0

            # ----------------------------------------------
            # Node load
            # ----------------------------------------------

            node_load = float(
                node_loads.get(
                    node,
                    0.0,
                )
            )

            node_load = float(
                np.clip(
                    node_load,
                    0.0,
                    1.0,
                )
            )

            # ----------------------------------------------
            # Coordinates
            # ----------------------------------------------

            x, y = coordinates[node]

            # ----------------------------------------------
            # Source
            # ----------------------------------------------

            is_source = float(
                node == demand["src"]
            )

            # ----------------------------------------------
            # Destination
            # ----------------------------------------------

            is_destination = float(
                node == demand["dst"]
            )

            # ----------------------------------------------
            # Failure
            # ----------------------------------------------

            is_failed = float(
                node in failed_nodes
            )

            node_features.append(
                [
                    normalized_degree,
                    node_load,
                    x,
                    y,
                    is_source,
                    is_destination,
                    is_failed,
                ]
            )

        node_features = np.asarray(
            node_features,
            dtype=np.float32,
        )

        # ==========================================================
        # EDGE INDEX + EDGE FEATURES
        # ==========================================================

        edge_index = []

        edge_features = []

        failed_links = (
            self._failed_link_set()
        )

        for u, v, data in self.graph.edges(
            data=True
        ):

            edge_index.append(
                [
                    node_to_index[u],
                    node_to_index[v],
                ]
            )

            link_key = (
                self._canonical_link(u, v)
            )

            # ----------------------------------------------
            # Failure
            # ----------------------------------------------

            is_failed = float(
                (
                    link_key
                    in failed_links
                )
                or bool(
                    data.get(
                        "is_failed",
                        False,
                    )
                )
            )

            # ----------------------------------------------
            # Utilization
            # ----------------------------------------------

            utilization = float(
                data.get(
                    "utilization",
                    0.0,
                )
            )

            # ----------------------------------------------
            # Capacity
            # ----------------------------------------------

            capacity = float(
                data.get(
                    "capacity_mbps",
                    data.get(
                        "capacity",
                        0.0,
                    ),
                )
            )

            # ----------------------------------------------
            # Delay
            # ----------------------------------------------

            delay = float(
                data.get(
                    "delay_ms",
                    data.get(
                        "delay",
                        0.0,
                    ),
                )
            )

            # ----------------------------------------------
            # Packet loss
            # ----------------------------------------------

            packet_loss = float(
                data.get(
                    "packet_loss",
                    0.0,
                )
            )

            edge_features.append(
                [
                    utilization,
                    capacity,
                    delay,
                    packet_loss,
                    is_failed,
                ]
            )

        # ----------------------------------------------------------
        # Convert edge index
        # ----------------------------------------------------------

        if edge_index:

            edge_index = np.asarray(
                edge_index,
                dtype=np.int64,
            ).T

        else:

            edge_index = np.empty(
                (2, 0),
                dtype=np.int64,
            )

        # ----------------------------------------------------------
        # Convert edge features
        # ----------------------------------------------------------

        if edge_features:

            edge_features = np.asarray(
                edge_features,
                dtype=np.float32,
            )

        else:

            edge_features = np.empty(
                (
                    0,
                    self.NUM_EDGE_FEATURES,
                ),
                dtype=np.float32,
            )

        # ==========================================================
        # DEMAND FEATURES
        # ==========================================================

        traffic_volume = float(
            demand["traffic_mbps"]
        )

        demand_features = np.asarray(
            [[traffic_volume]],
            dtype=np.float32,
        )

        # ==========================================================
        # FINAL STATE
        # ==========================================================

        return {
            "node_features":
                node_features,

            "edge_index":
                edge_index,

            "edge_features":
                edge_features,

            "demand_features":
                demand_features,
        }

    # ==============================================================
    # NODE LOAD
    # ==============================================================

    def _get_node_loads(
        self,
    ) -> Dict[Any, float]:
        """
        Read node loads calculated by NetworkSimulator.

        NetworkSimulator stores them directly as:

            graph.nodes[node]["node_load"]
        """

        if self.graph is None:
            return {}

        return {
            node: float(
                self.graph.nodes[node].get(
                    "node_load",
                    0.0,
                )
            )
            for node in self.graph.nodes()
        }

    # ==============================================================
    # NODE COORDINATES
    # ==============================================================

    def _get_node_coordinates(
        self,
        nodes: List[Any],
    ) -> Dict[Any, Tuple[float, float]]:
        """
        Extract and normalize node coordinates.

        Topology Zoo files can use different coordinate names.

        Supported names include:

            x / X
            y / Y
            longitude / Longitude
            latitude / Latitude
            lon / Lon
            lat / Lat

        If coordinates are unavailable, a deterministic fallback
        based on node ordering is used.
        """

        if self.graph is None:
            return {
                node: (0.0, 0.0)
                for node in nodes
            }

        raw_coordinates = {}

        x_keys = (
            "x",
            "X",
            "longitude",
            "Longitude",
            "lon",
            "Lon",
        )

        y_keys = (
            "y",
            "Y",
            "latitude",
            "Latitude",
            "lat",
            "Lat",
        )

        # ----------------------------------------------------------
        # Try to extract coordinates
        # ----------------------------------------------------------

        for node in nodes:

            data = self.graph.nodes[node]

            x_value = (
                self._first_numeric_attribute(
                    data,
                    x_keys,
                )
            )

            y_value = (
                self._first_numeric_attribute(
                    data,
                    y_keys,
                )
            )

            if (
                x_value is not None
                and y_value is not None
            ):

                raw_coordinates[node] = (
                    x_value,
                    y_value,
                )

        # ----------------------------------------------------------
        # If every node has coordinates
        # ----------------------------------------------------------

        if len(raw_coordinates) == len(nodes):

            return self._normalize_coordinates(
                raw_coordinates
            )

        # ----------------------------------------------------------
        # Deterministic fallback
        # ----------------------------------------------------------

        fallback = {}

        n = max(
            len(nodes),
            1,
        )

        for index, node in enumerate(nodes):

            if n == 1:
                x = 0.0
            else:
                x = (
                    float(index)
                    / float(n - 1)
                )

            fallback[node] = (
                float(x),
                0.0,
            )

        # Preserve actual coordinates where
        # they exist.
        fallback.update(
            raw_coordinates
        )

        return self._normalize_coordinates(
            fallback
        )

    @staticmethod
    def _first_numeric_attribute(
        data: Dict[str, Any],
        keys: Tuple[str, ...],
    ) -> Optional[float]:
        """
        Return the first valid numeric attribute.
        """

        for key in keys:

            if key not in data:
                continue

            try:

                value = float(
                    data[key]
                )

                if np.isfinite(value):
                    return value

            except (
                TypeError,
                ValueError,
            ):
                continue

        return None

    @staticmethod
    def _normalize_coordinates(
        coordinates: Dict[
            Any,
            Tuple[float, float],
        ],
    ) -> Dict[
        Any,
        Tuple[float, float],
    ]:
        """
        Min-max normalize coordinates independently
        into [0, 1].
        """

        if not coordinates:
            return {}

        xs = np.asarray(
            [
                value[0]
                for value in coordinates.values()
            ],
            dtype=np.float64,
        )

        ys = np.asarray(
            [
                value[1]
                for value in coordinates.values()
            ],
            dtype=np.float64,
        )

        x_min = float(
            xs.min()
        )

        x_max = float(
            xs.max()
        )

        y_min = float(
            ys.min()
        )

        y_max = float(
            ys.max()
        )

        x_range = (
            x_max - x_min
        )

        y_range = (
            y_max - y_min
        )

        normalized = {}

        for node, (x, y) in coordinates.items():

            if x_range > 0:
                nx_value = (
                    (x - x_min)
                    / x_range
                )
            else:
                nx_value = 0.0

            if y_range > 0:
                ny_value = (
                    (y - y_min)
                    / y_range
                )
            else:
                ny_value = 0.0

            normalized[node] = (
                float(nx_value),
                float(ny_value),
            )

        return normalized

    # ==============================================================
    # FAILURE INJECTION
    # ==============================================================

    def inject_link_failure(
        self,
        u: Any,
        v: Any,
    ) -> None:
        """
        Fail a specific link.
        """

        if self.simulator is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        self.simulator.fail_link(
            u,
            v,
        )

        self._refresh_simulator_state()

    def recover_link(
        self,
        u: Any,
        v: Any,
    ) -> None:
        """
        Recover a failed link.
        """

        if self.simulator is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        self.simulator.recover_link(
            u,
            v,
        )

        self._refresh_simulator_state()

    def inject_node_failure(
        self,
        node: Any,
    ) -> None:
        """
        Fail a specific node.
        """

        if self.simulator is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        self.simulator.fail_node(
            node
        )

        self._refresh_simulator_state()

    def recover_node(
        self,
        node: Any,
    ) -> None:
        """
        Recover a failed node.
        """

        if self.simulator is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        self.simulator.recover_node(
            node
        )

        self._refresh_simulator_state()

    def inject_random_link_failure(
        self,
    ) -> Tuple[Any, Any]:
        """
        Randomly fail one healthy link.

        Returns
        -------
        tuple
            (u, v)
        """

        if (
            self.graph is None
            or self.simulator is None
        ):
            raise RuntimeError(
                "Environment has not been reset"
            )

        failed_links = (
            self._failed_link_set()
        )

        healthy_edges = []

        for u, v, data in self.graph.edges(
            data=True
        ):

            key = (
                self._canonical_link(u, v)
            )

            if key in failed_links:
                continue

            if bool(
                data.get(
                    "is_failed",
                    False,
                )
            ):
                continue

            healthy_edges.append(
                (u, v)
            )

        if not healthy_edges:
            raise RuntimeError(
                "No healthy links available"
            )

        edge = self.rng.choice(
            healthy_edges
        )

        self.simulator.fail_link(
            edge[0],
            edge[1],
        )

        self._refresh_simulator_state()

        return edge

    def inject_random_node_failure(
        self,
    ) -> Any:
        """
        Randomly fail one healthy node.

        Returns
        -------
        node
            Failed node.
        """

        if (
            self.graph is None
            or self.simulator is None
        ):
            raise RuntimeError(
                "Environment has not been reset"
            )

        failed_nodes = set(
            self.simulator.get_failed_nodes()
        )

        healthy_nodes = [
            node
            for node in self.graph.nodes()
            if node not in failed_nodes
        ]

        if not healthy_nodes:
            raise RuntimeError(
                "No healthy nodes available"
            )

        node = self.rng.choice(
            healthy_nodes
        )

        self.simulator.fail_node(
            node
        )

        self._refresh_simulator_state()

        return node

    # ==============================================================
    # UTILITY
    # ==============================================================

    def _get_current_demand(
        self,
    ) -> Dict[str, Any]:
        """
        Get the demand represented by the current
        source/destination node features.
        """

        if not self.demands:
            raise RuntimeError(
                "No demands are loaded"
            )

        index = min(
            self.current_demand_index,
            len(self.demands) - 1,
        )

        return self.demands[index]

    def get_current_demand(
        self,
    ) -> Dict[str, Any]:
        """
        Public current-demand accessor.
        """

        return dict(
            self._get_current_demand()
        )

    def get_state(
        self,
    ) -> Dict[str, np.ndarray]:
        """
        Return the current state without stepping.
        """

        if self.current_state is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        return self.current_state

    def get_metrics(
        self,
    ) -> Dict[str, Any]:
        """
        Return current network metrics.
        """

        if self.simulator is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        return (
            self.simulator
            .calculate_network_metrics()
        )

    def _failed_link_set(
        self,
    ) -> set:
        """
        Return failed links as canonical undirected pairs.
        """

        if self.simulator is None:
            return set()

        return {
            self._canonical_link(u, v)
            for u, v
            in self.simulator.get_failed_links()
        }

    @staticmethod
    def _canonical_link(
        u: Any,
        v: Any,
    ) -> Tuple[Any, Any]:
        """
        Canonical representation of an undirected link.
        """

        try:
            return tuple(
                sorted((u, v))
            )

        except TypeError:

            if str(u) <= str(v):
                return (u, v)

            return (v, u)

    @staticmethod
    def _validate_actions(
        congestion_action: int,
        failure_action: int,
    ) -> None:
        """
        Validate both agent actions.
        """

        if congestion_action not in (
            0,
            1,
            2,
        ):
            raise ValueError(
                "Congestion action must be "
                "0, 1 or 2"
            )

        if failure_action not in (
            0,
            1,
            2,
        ):
            raise ValueError(
                "Failure action must be "
                "0, 1 or 2"
            )

    def close(self) -> None:
        """
        Release environment resources.
        """

        self.graph = None

        self.traffic_generator = None

        self.routing = None

        self.simulator = None

        self.demands = []

        self.current_state = None

        self.last_result = None

        self.last_reward_details = None

        self.current_demand_index = 0

        self.current_step = 0

        self.done = False


# ==================================================================
# COMMAND-LINE TEST
# ==================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test SDN MARL environment"
    )

    parser.add_argument(
        "topology",
        help="Path to a GML topology",
    )

    parser.add_argument(
        "--scenario",
        default="normal",
        choices=[
            "normal",
            "heavy",
            "single_hotspot",
            "multiple_hotspots",
            "traffic_spike",
            "random",
        ],
    )

    parser.add_argument(
        "--demands",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Create environment
    # --------------------------------------------------------------

    env = SDNEnvironment(
        topology_path=args.topology,
        traffic_scenario=args.scenario,
        num_demands=args.demands,
        max_steps=args.steps,
        k_paths=5,
        random_seed=args.seed,
    )

    # --------------------------------------------------------------
    # Reset
    # --------------------------------------------------------------

    state = env.reset()

    print()
    print("=" * 70)
    print("SDN MARL ENVIRONMENT TEST")
    print("=" * 70)

    print(
        f"Topology: {args.topology}"
    )

    print(
        f"Scenario: {args.scenario}"
    )

    print(
        f"Demands: {len(env.demands)}"
    )

    print()

    print(
        "node_features:",
        state["node_features"].shape,
    )

    print(
        "edge_index:",
        state["edge_index"].shape,
    )

    print(
        "edge_features:",
        state["edge_features"].shape,
    )

    print(
        "demand_features:",
        state["demand_features"].shape,
    )

    print()

    print(
        "Current demand:",
        env.get_current_demand(),
    )

    print()

    # --------------------------------------------------------------
    # Check expected feature dimensions
    # --------------------------------------------------------------

    assert (
        state["node_features"].shape[1]
        == 7
    )

    assert (
        state["edge_index"].shape[0]
        == 2
    )

    assert (
        state["edge_features"].shape[1]
        == 5
    )

    assert (
        state["demand_features"].shape
        == (1, 1)
    )

    print(
        "State shape validation: PASS"
    )

    # --------------------------------------------------------------
    # Test maintain / no action
    # --------------------------------------------------------------

    next_state, reward, done, info = (
        env.step(
            congestion_action=0,
            failure_action=0,
        )
    )

    print()
    print("=" * 70)
    print("STEP 1: MAINTAIN / NO ACTION")
    print("=" * 70)

    print(
        "Reward:",
        reward,
    )

    print(
        "Done:",
        done,
    )

    print(
        "Metrics:"
    )

    print(
        info["metrics"]
    )

    # --------------------------------------------------------------
    # Test congestion reroute
    # --------------------------------------------------------------

    if not done:

        next_state, reward, done, info = (
            env.step(
                congestion_action=1,
                failure_action=0,
            )
        )

        print()
        print("=" * 70)
        print("STEP 2: CONGESTION REROUTE")
        print("=" * 70)

        print(
            "Reward:",
            reward,
        )

        print(
            "Done:",
            done,
        )

        print(
            "Action info:",
            info["congestion_action_info"],
        )

    # --------------------------------------------------------------
    # Close
    # --------------------------------------------------------------

    env.close()

    print()
    print("=" * 70)
    print("ENVIRONMENT TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()