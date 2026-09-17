"""
Network Simulator for SDN-MARL

Person 1 component.

Responsibilities:
    - Apply traffic demands to network paths
    - Calculate link traffic
    - Calculate link utilization
    - Calculate delay
    - Calculate packet loss
    - Calculate throughput
    - Calculate node load
    - Handle link/node failures
    - Execute Congestion Agent actions
    - Execute Failure Agent actions

Agent actions:

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
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from simulator.topology_loader import TopologyLoader
from simulator.traffic_generator import TrafficGenerator
from simulator.routing import RoutingEngine


class NetworkSimulator:
    """
    Simulates traffic behavior over a NetworkX topology.

    The simulator owns the actual network mechanics.

    The RL agents only decide what action should be performed.
    """

    # --------------------------------------------------------------
    # DEFAULT SIMULATION PARAMETERS
    # --------------------------------------------------------------

    DEFAULT_CAPACITY_MBPS = 1000.0
    DEFAULT_DELAY_MS = 1.0

    # Maximum utilization before the link is considered congested.
    CONGESTION_THRESHOLD = 0.80

    # Utilization at which packet loss starts increasing.
    LOSS_THRESHOLD = 0.90

    # Maximum packet loss used by the simple simulator model.
    MAX_PACKET_LOSS = 0.50

    # Rate-limit factor.
    #
    # Example:
    #     traffic = 100 Mbps
    #     RATE_LIMIT_FACTOR = 0.70
    #
    # New traffic = 70 Mbps
    RATE_LIMIT_FACTOR = 0.70

    def __init__(
        self,
        graph: nx.Graph,
        k_paths: int = 5,
        random_seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        graph:
            NetworkX topology.

        k_paths:
            Number of alternative paths considered during rerouting.

        random_seed:
            Optional random seed for reproducibility.
        """

        self.graph = graph

        self.k_paths = max(1, int(k_paths))

        self.routing = RoutingEngine(
            graph=self.graph,
            k_paths=self.k_paths,
        )

        if random_seed is not None:
            random.seed(random_seed)

        # Store currently active traffic demands.
        self.demands: List[Dict[str, Any]] = []

        # Store the currently selected path for every demand.
        self.active_paths: Dict[int, List[Any]] = {}

        # Store original traffic before rate limiting.
        self.original_traffic: Dict[int, float] = {}

        # Last simulation result.
        self.last_result: Optional[Dict[str, Any]] = None

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(self) -> None:
        """
        Reset traffic and all dynamic network statistics.
        """

        self.demands = []
        self.active_paths = {}
        self.original_traffic = {}
        self.last_result = None

        self._reset_dynamic_edge_state()
        self._reset_dynamic_node_state()

    # ==============================================================
    # DYNAMIC STATE
    # ==============================================================

    def _reset_dynamic_edge_state(self) -> None:
        """
        Reset traffic-related edge attributes.
        """

        for u, v, data in self.graph.edges(data=True):

            data["traffic_mbps"] = 0.0
            data["utilization"] = 0.0
            data["packet_loss"] = 0.0

            # Preserve failure state if already present.
            data["is_failed"] = int(
                bool(data.get("is_failed", 0))
            )

    def _reset_dynamic_node_state(self) -> None:
        """
        Reset traffic-related node attributes.
        """

        for node, data in self.graph.nodes(data=True):

            data["node_load"] = 0.0

            # Preserve failure state if already present.
            data["is_failed"] = int(
                bool(data.get("is_failed", 0))
            )

    # ==============================================================
    # CAPACITY / EDGE HELPERS
    # ==============================================================

    def _capacity(self, u: Any, v: Any) -> float:
        """
        Return link capacity in Mbps.
        """

        return float(
            self.graph[u][v].get(
                "capacity_mbps",
                self.DEFAULT_CAPACITY_MBPS,
            )
        )

    def _base_delay(self, u: Any, v: Any) -> float:
        """
        Return base propagation delay in milliseconds.
        """

        return float(
            self.graph[u][v].get(
                "delay_ms",
                self.DEFAULT_DELAY_MS,
            )
        )

    def _is_edge_failed(self, u: Any, v: Any) -> bool:
        """
        Check whether a link has failed.
        """

        return bool(
            self.graph[u][v].get("is_failed", 0)
        )

    def _is_node_failed(self, node: Any) -> bool:
        """
        Check whether a node has failed.
        """

        return bool(
            self.graph.nodes[node].get("is_failed", 0)
        )

    # ==============================================================
    # PATH VALIDATION
    # ==============================================================

    def _path_is_usable(
        self,
        path: Optional[List[Any]],
    ) -> bool:
        """
        Check whether every node and link on a path is operational.
        """

        if not path:
            return False

        for node in path:

            if self._is_node_failed(node):
                return False

        for u, v in zip(path[:-1], path[1:]):

            if not self.graph.has_edge(u, v):
                return False

            if self._is_edge_failed(u, v):
                return False

        return True

    # ==============================================================
    # PATH FINDING
    # ==============================================================

    def find_path(
        self,
        src: Any,
        dst: Any,
    ) -> Optional[List[Any]]:
        """
        Find a usable shortest path.

        Failed nodes and links are excluded.
        """

        if src not in self.graph:
            return None

        if dst not in self.graph:
            return None

        if self._is_node_failed(src):
            return None

        if self._is_node_failed(dst):
            return None

        path = self.routing.get_path_excluding_failed_components(
            src,
            dst,
        )

        if self._path_is_usable(path):
            return path

        return None

    # ==============================================================
    # DEMAND INSTALLATION
    # ==============================================================

    def install_demands(
        self,
        demands: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Install traffic demands into the simulator.

        Expected format:

            {
                "src": 0,
                "dst": 8,
                "traffic_mbps": 100.0
            }

        Paths are automatically calculated.
        """

        self._reset_dynamic_edge_state()
        self._reset_dynamic_node_state()

        self.demands = []
        self.active_paths = {}
        self.original_traffic = {}

        failed_demands = []

        for index, demand in enumerate(demands):

            src = demand["src"]
            dst = demand["dst"]

            traffic = float(
                demand.get("traffic_mbps", 0.0)
            )

            traffic = max(0.0, traffic)

            path = self.find_path(src, dst)

            demand_copy = {
                "id": index,
                "src": src,
                "dst": dst,
                "traffic_mbps": traffic,
            }

            self.demands.append(demand_copy)

            self.original_traffic[index] = traffic

            if path is None:

                failed_demands.append(index)

            else:

                self.active_paths[index] = path

        self._apply_all_traffic()

        result = self.calculate_network_metrics()

        result["failed_demands"] = failed_demands

        self.last_result = result

        return result

    # ==============================================================
    # APPLY TRAFFIC
    # ==============================================================

    def _apply_all_traffic(self) -> None:
        """
        Apply all active demands to their selected paths.
        """

        # Clear current traffic.
        for u, v, data in self.graph.edges(data=True):
            data["traffic_mbps"] = 0.0

        # Apply every demand.
        for demand_id, path in self.active_paths.items():

            demand = self.demands[demand_id]

            traffic = float(
                demand.get("traffic_mbps", 0.0)
            )

            if not self._path_is_usable(path):
                continue

            for u, v in zip(path[:-1], path[1:]):

                self.graph[u][v]["traffic_mbps"] += traffic

    # ==============================================================
    # LINK UTILIZATION
    # ==============================================================

    def calculate_link_utilization(self) -> None:
        """
        Calculate utilization for every link.

        utilization = traffic / capacity
        """

        for u, v, data in self.graph.edges(data=True):

            if data.get("is_failed", 0):

                data["utilization"] = 1.0
                continue

            capacity = float(
                data.get(
                    "capacity_mbps",
                    self.DEFAULT_CAPACITY_MBPS,
                )
            )

            traffic = float(
                data.get("traffic_mbps", 0.0)
            )

            if capacity <= 0:
                data["utilization"] = 1.0

            else:
                data["utilization"] = traffic / capacity

    # ==============================================================
    # DELAY
    # ==============================================================

    def calculate_link_delay(
        self,
        u: Any,
        v: Any,
    ) -> float:
        """
        Calculate congestion-aware link delay.

        Base delay is taken from the topology.

        A simple queueing approximation is used:

            delay = base_delay / (1 - utilization)

        as utilization approaches 1.

        The result is capped to prevent numerical explosion.
        """

        data = self.graph[u][v]

        base_delay = float(
            data.get(
                "delay_ms",
                self.DEFAULT_DELAY_MS,
            )
        )

        utilization = float(
            data.get("utilization", 0.0)
        )

        if data.get("is_failed", 0):
            return float("inf")

        # Keep the denominator away from zero.
        denominator = max(
            1.0 - min(utilization, 0.99),
            0.01,
        )

        delay = base_delay / denominator

        # Prevent unrealistic huge values.
        delay = min(delay, 10000.0)

        return delay

    def calculate_all_link_delays(self) -> None:
        """
        Store congestion-aware delay for every link.
        """

        for u, v, data in self.graph.edges(data=True):

            data["current_delay_ms"] = (
                self.calculate_link_delay(u, v)
            )

    # ==============================================================
    # PACKET LOSS
    # ==============================================================

    def calculate_link_packet_loss(
        self,
        u: Any,
        v: Any,
    ) -> float:
        """
        Estimate packet loss from link utilization.

        Below 90% utilization:
            packet loss = 0

        Above 90%:
            packet loss increases with congestion.

        Failed link:
            packet loss = 1
        """

        data = self.graph[u][v]

        if data.get("is_failed", 0):
            return 1.0

        utilization = float(
            data.get("utilization", 0.0)
        )

        # Preserve any explicitly supplied baseline loss.
        baseline_loss = float(
            data.get("base_packet_loss", 0.0)
        )

        baseline_loss = min(
            max(baseline_loss, 0.0),
            1.0,
        )

        if utilization <= self.LOSS_THRESHOLD:
            return baseline_loss

        excess = (
            utilization - self.LOSS_THRESHOLD
        ) / max(
            1.0 - self.LOSS_THRESHOLD,
            0.01,
        )

        congestion_loss = min(
            excess * self.MAX_PACKET_LOSS,
            self.MAX_PACKET_LOSS,
        )

        return min(
            1.0,
            baseline_loss + congestion_loss,
        )

    def calculate_all_packet_loss(self) -> None:
        """
        Calculate packet loss for every link.
        """

        for u, v, data in self.graph.edges(data=True):

            data["packet_loss"] = (
                self.calculate_link_packet_loss(u, v)
            )

    # ==============================================================
    # NODE LOAD
    # ==============================================================

    def calculate_node_load(self) -> None:
        """
        Calculate normalized node load.

        Node load is based on traffic passing through incident links.

        For each node:

            node_load =
                average incident-link utilization
        """

        for node in self.graph.nodes:

            if self._is_node_failed(node):

                self.graph.nodes[node]["node_load"] = 1.0
                continue

            incident_utilizations = []

            for neighbor in self.graph.neighbors(node):

                if self.graph.has_edge(node, neighbor):

                    utilization = float(
                        self.graph[node][neighbor].get(
                            "utilization",
                            0.0,
                        )
                    )

                    incident_utilizations.append(
                        min(max(utilization, 0.0), 1.0)
                    )

            if not incident_utilizations:

                load = 0.0

            else:

                load = sum(
                    incident_utilizations
                ) / len(
                    incident_utilizations
                )

            self.graph.nodes[node]["node_load"] = load

    # ==============================================================
    # THROUGHPUT
    # ==============================================================

    def calculate_throughput(self) -> float:
        """
        Calculate total successfully delivered throughput.

        For each demand, throughput is limited by the bottleneck
        capacity and packet loss along its path.
        """

        total_throughput = 0.0

        for demand_id, path in self.active_paths.items():

            demand = self.demands[demand_id]

            requested = float(
                demand.get("traffic_mbps", 0.0)
            )

            if requested <= 0:
                continue

            if not self._path_is_usable(path):
                continue

            bottleneck_capacity = float("inf")
            end_to_end_success = 1.0

            for u, v in zip(path[:-1], path[1:]):

                capacity = self._capacity(u, v)

                bottleneck_capacity = min(
                    bottleneck_capacity,
                    capacity,
                )

                packet_loss = float(
                    self.graph[u][v].get(
                        "packet_loss",
                        0.0,
                    )
                )

                end_to_end_success *= (
                    1.0 - packet_loss
                )

            deliverable = min(
                requested,
                bottleneck_capacity,
            )

            throughput = (
                deliverable
                * end_to_end_success
            )

            total_throughput += throughput

        return total_throughput

    # ==============================================================
    # CONGESTION
    # ==============================================================

    def congested_links(self) -> List[Tuple[Any, Any]]:
        """
        Return links whose utilization exceeds the congestion threshold.
        """

        result = []

        for u, v, data in self.graph.edges(data=True):

            if data.get("is_failed", 0):
                continue

            utilization = float(
                data.get("utilization", 0.0)
            )

            if utilization >= self.CONGESTION_THRESHOLD:

                result.append((u, v))

        return result

    def congestion_ratio(self) -> float:
        """
        Fraction of healthy links that are congested.
        """

        healthy_links = 0
        congested = 0

        for u, v, data in self.graph.edges(data=True):

            if data.get("is_failed", 0):
                continue

            healthy_links += 1

            if float(
                data.get("utilization", 0.0)
            ) >= self.CONGESTION_THRESHOLD:

                congested += 1

        if healthy_links == 0:
            return 0.0

        return congested / healthy_links

    # ==============================================================
    # PATH DELAY
    # ==============================================================

    def path_delay(
        self,
        path: List[Any],
    ) -> float:
        """
        Calculate current end-to-end path delay.
        """

        if not self._path_is_usable(path):
            return float("inf")

        total = 0.0

        for u, v in zip(path[:-1], path[1:]):

            delay = float(
                self.graph[u][v].get(
                    "current_delay_ms",
                    self._base_delay(u, v),
                )
            )

            total += delay

        return total

    # ==============================================================
    # PATH PACKET LOSS
    # ==============================================================

    def path_packet_loss(
        self,
        path: List[Any],
    ) -> float:
        """
        Calculate end-to-end packet loss.
        """

        if not self._path_is_usable(path):
            return 1.0

        success = 1.0

        for u, v in zip(path[:-1], path[1:]):

            loss = float(
                self.graph[u][v].get(
                    "packet_loss",
                    0.0,
                )
            )

            loss = min(max(loss, 0.0), 1.0)

            success *= (1.0 - loss)

        return 1.0 - success

    # ==============================================================
    # NETWORK METRICS
    # ==============================================================

    def calculate_network_metrics(self) -> Dict[str, Any]:
        """
        Calculate the complete current network state.
        """

        self.calculate_link_utilization()

        self.calculate_all_link_delays()

        self.calculate_all_packet_loss()

        self.calculate_node_load()

        throughput = self.calculate_throughput()

        congested = self.congested_links()

        total_requested = sum(
            float(
                demand.get(
                    "traffic_mbps",
                    0.0,
                )
            )
            for demand in self.demands
        )

        total_link_capacity = sum(
            float(
                data.get(
                    "capacity_mbps",
                    self.DEFAULT_CAPACITY_MBPS,
                )
            )
            for _, _, data in self.graph.edges(data=True)
            if not data.get("is_failed", 0)
        )

        average_utilization = 0.0

        healthy_utilizations = [
            float(data.get("utilization", 0.0))
            for _, _, data in self.graph.edges(data=True)
            if not data.get("is_failed", 0)
        ]

        if healthy_utilizations:

            average_utilization = (
                sum(healthy_utilizations)
                / len(healthy_utilizations)
            )

        # Calculate path-level statistics.
        path_delays = []
        path_losses = []

        for path in self.active_paths.values():

            delay = self.path_delay(path)

            loss = self.path_packet_loss(path)

            if delay != float("inf"):

                path_delays.append(delay)

            path_losses.append(loss)

        average_delay = (
            sum(path_delays) / len(path_delays)
            if path_delays
            else 0.0
        )

        average_packet_loss = (
            sum(path_losses) / len(path_losses)
            if path_losses
            else 0.0
        )

        result = {
            "total_requested_traffic_mbps": total_requested,
            "total_throughput_mbps": throughput,
            "total_link_capacity_mbps": total_link_capacity,
            "average_link_utilization": average_utilization,
            "congestion_ratio": self.congestion_ratio(),
            "congested_links": congested,
            "average_path_delay_ms": average_delay,
            "average_packet_loss": average_packet_loss,
            "active_demands": len(self.active_paths),
            "total_demands": len(self.demands),
            "failed_links": self.get_failed_links(),
            "failed_nodes": self.get_failed_nodes(),
            "link_statistics": self.get_link_statistics(),
            "node_statistics": self.get_node_statistics(),
            "path_statistics": self.get_path_statistics(),
        }

        return result

    # ==============================================================
    # LINK STATISTICS
    # ==============================================================

    def get_link_statistics(self) -> List[Dict[str, Any]]:
        """
        Return current state of every link.
        """

        statistics = []

        for u, v, data in self.graph.edges(data=True):

            statistics.append(
                {
                    "src": u,
                    "dst": v,
                    "capacity_mbps": float(
                        data.get(
                            "capacity_mbps",
                            self.DEFAULT_CAPACITY_MBPS,
                        )
                    ),
                    "traffic_mbps": float(
                        data.get(
                            "traffic_mbps",
                            0.0,
                        )
                    ),
                    "utilization": float(
                        data.get(
                            "utilization",
                            0.0,
                        )
                    ),
                    "delay_ms": float(
                        data.get(
                            "current_delay_ms",
                            data.get(
                                "delay_ms",
                                self.DEFAULT_DELAY_MS,
                            ),
                        )
                    ),
                    "packet_loss": float(
                        data.get(
                            "packet_loss",
                            0.0,
                        )
                    ),
                    "is_failed": int(
                        bool(data.get("is_failed", 0))
                    ),
                }
            )

        return statistics

    # ==============================================================
    # NODE STATISTICS
    # ==============================================================

    def get_node_statistics(self) -> List[Dict[str, Any]]:
        """
        Return current state of every node.
        """

        statistics = []

        for node, data in self.graph.nodes(data=True):

            statistics.append(
                {
                    "node": node,
                    "degree": self.graph.degree(node),
                    "node_load": float(
                        data.get(
                            "node_load",
                            0.0,
                        )
                    ),
                    "is_failed": int(
                        bool(data.get("is_failed", 0))
                    ),
                }
            )

        return statistics

    # ==============================================================
    # PATH STATISTICS
    # ==============================================================

    def get_path_statistics(self) -> List[Dict[str, Any]]:
        """
        Return statistics for active demand paths.
        """

        statistics = []

        for demand_id, path in self.active_paths.items():

            demand = self.demands[demand_id]

            statistics.append(
                {
                    "demand_id": demand_id,
                    "src": demand["src"],
                    "dst": demand["dst"],
                    "traffic_mbps": demand["traffic_mbps"],
                    "path": path,
                    "hop_count": max(0, len(path) - 1),
                    "delay_ms": self.path_delay(path),
                    "packet_loss": self.path_packet_loss(path),
                }
            )

        return statistics

    # ==============================================================
    # FAILURE HANDLING
    # ==============================================================

    def fail_link(
        self,
        u: Any,
        v: Any,
    ) -> bool:
        """
        Fail a link.
        """

        if not self.graph.has_edge(u, v):
            return False

        self.graph[u][v]["is_failed"] = 1

        return True

    def recover_link(
        self,
        u: Any,
        v: Any,
    ) -> bool:
        """
        Recover a failed link.
        """

        if not self.graph.has_edge(u, v):
            return False

        self.graph[u][v]["is_failed"] = 0

        return True

    def fail_node(
        self,
        node: Any,
    ) -> bool:
        """
        Fail a node.
        """

        if node not in self.graph:
            return False

        self.graph.nodes[node]["is_failed"] = 1

        return True

    def recover_node(
        self,
        node: Any,
    ) -> bool:
        """
        Recover a failed node.
        """

        if node not in self.graph:
            return False

        self.graph.nodes[node]["is_failed"] = 0

        return True

    def get_failed_links(self) -> List[Tuple[Any, Any]]:
        """
        Return all failed links.
        """

        return [
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if data.get("is_failed", 0)
        ]

    def get_failed_nodes(self) -> List[Any]:
        """
        Return all failed nodes.
        """

        return [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("is_failed", 0)
        ]

    # ==============================================================
    # REROUTING
    # ==============================================================

    def reroute_demand(
        self,
        demand_id: int,
    ) -> bool:
        """
        Reroute one demand using a healthy path.

        Returns True if a new usable path was found.
        """

        if demand_id < 0 or demand_id >= len(self.demands):
            return False

        demand = self.demands[demand_id]

        src = demand["src"]
        dst = demand["dst"]

        old_path = self.active_paths.get(demand_id)

        # Find a healthy path.
        new_path = self.find_path(src, dst)

        if new_path is None:
            self.active_paths.pop(
                demand_id,
                None,
            )
            return False

        # If the path is exactly the same, there was no
        # alternative route.
        if old_path == new_path:

            # It is still a valid route.
            self.active_paths[demand_id] = new_path

            return True

        self.active_paths[demand_id] = new_path

        return True

    def reroute_all_demands(self) -> int:
        """
        Attempt to reroute every active demand.

        Returns number of successfully rerouted demands.
        """

        rerouted = 0

        for demand_id in range(len(self.demands)):

            old_path = self.active_paths.get(
                demand_id
            )

            new_path = self.find_path(
                self.demands[demand_id]["src"],
                self.demands[demand_id]["dst"],
            )

            if new_path is None:
                self.active_paths.pop(
                    demand_id,
                    None,
                )
                continue

            if old_path != new_path:
                rerouted += 1

            self.active_paths[demand_id] = new_path

        self._apply_all_traffic()

        return rerouted

    # ==============================================================
    # RATE LIMITING
    # ==============================================================

    def rate_limit_demands(
        self,
        factor: Optional[float] = None,
    ) -> float:
        """
        Reduce active traffic rates.

        Returns total amount of traffic removed.
        """

        if factor is None:
            factor = self.RATE_LIMIT_FACTOR

        factor = min(
            max(float(factor), 0.0),
            1.0,
        )

        traffic_removed = 0.0

        for demand in self.demands:

            old_traffic = float(
                demand.get(
                    "traffic_mbps",
                    0.0,
                )
            )

            new_traffic = old_traffic * factor

            traffic_removed += (
                old_traffic - new_traffic
            )

            demand["traffic_mbps"] = new_traffic

        self._apply_all_traffic()

        return traffic_removed

    # ==============================================================
    # ISOLATION
    # ==============================================================

    def isolate_failed_components(self) -> Dict[str, Any]:
        """
        Ensure failed components cannot carry traffic.

        Failed links/nodes are not physically removed from the graph.
        Instead, they remain visible through is_failed=1 while the
        routing layer treats them as unavailable.
        """

        removed_demands = []

        for demand_id, path in list(
            self.active_paths.items()
        ):

            if not self._path_is_usable(path):

                removed_demands.append(
                    demand_id
                )

                del self.active_paths[demand_id]

        self._apply_all_traffic()

        return {
            "isolated_links": self.get_failed_links(),
            "isolated_nodes": self.get_failed_nodes(),
            "removed_demands": removed_demands,
        }

    # ==============================================================
    # AGENT ACTIONS
    # ==============================================================

    def execute_actions(
        self,
        congestion_action: int = 0,
        failure_action: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute actions selected by the two RL agents.

        Congestion actions:
            0 = maintain
            1 = reroute
            2 = rate-limit

        Failure actions:
            0 = no action
            1 = reroute
            2 = isolate failed component
        """

        info: Dict[str, Any] = {
            "congestion_action": congestion_action,
            "failure_action": failure_action,
            "rerouted_demands": 0,
            "rate_limited_traffic_mbps": 0.0,
            "isolation": None,
        }

        # ----------------------------------------------------------
        # CONGESTION AGENT
        # ----------------------------------------------------------

        if congestion_action == 0:

            # Maintain current routing.
            pass

        elif congestion_action == 1:

            # Reroute traffic.
            info["rerouted_demands"] = (
                self.reroute_all_demands()
            )

        elif congestion_action == 2:

            # Rate-limit traffic.
            info["rate_limited_traffic_mbps"] = (
                self.rate_limit_demands()
            )

        else:

            raise ValueError(
                "Invalid congestion action. "
                "Expected 0, 1, or 2."
            )

        # ----------------------------------------------------------
        # FAILURE AGENT
        # ----------------------------------------------------------

        if failure_action == 0:

            # No action.
            pass

        elif failure_action == 1:

            # Attempt rerouting around failures.
            info["rerouted_demands"] += (
                self.reroute_all_demands()
            )

        elif failure_action == 2:

            # Isolate failed components.
            info["isolation"] = (
                self.isolate_failed_components()
            )

        else:

            raise ValueError(
                "Invalid failure action. "
                "Expected 0, 1, or 2."
            )

        # ----------------------------------------------------------
        # RECALCULATE NETWORK
        # ----------------------------------------------------------

        self._apply_all_traffic()

        result = self.calculate_network_metrics()

        result["action_info"] = info

        self.last_result = result

        return result

    # ==============================================================
    # STEP
    # ==============================================================

    def step(
        self,
        congestion_action: int = 0,
        failure_action: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute one simulation step.

        This is the function environment.py will eventually call.
        """

        return self.execute_actions(
            congestion_action=congestion_action,
            failure_action=failure_action,
        )


# ==================================================================
# CLI TEST
# ==================================================================

def load_topology(topology_path: str) -> nx.Graph:
    """
    Load topology through the project's TopologyLoader.
    """

    loader = TopologyLoader()

    return loader.load(topology_path)


def print_result(
    result: Dict[str, Any],
) -> None:
    """
    Pretty-print simulator results.
    """

    print()
    print("=" * 70)
    print("NETWORK SIMULATOR RESULT")
    print("=" * 70)

    print(
        f"Requested traffic : "
        f"{result['total_requested_traffic_mbps']:.2f} Mbps"
    )

    print(
        f"Throughput        : "
        f"{result['total_throughput_mbps']:.2f} Mbps"
    )

    print(
        f"Average utilization : "
        f"{result['average_link_utilization']:.4f}"
    )

    print(
        f"Congestion ratio  : "
        f"{result['congestion_ratio']:.4f}"
    )

    print(
        f"Average delay     : "
        f"{result['average_path_delay_ms']:.2f} ms"
    )

    print(
        f"Average packet loss : "
        f"{result['average_packet_loss']:.4f}"
    )

    print(
        f"Active demands    : "
        f"{result['active_demands']}/"
        f"{result['total_demands']}"
    )

    print(
        f"Failed links      : "
        f"{result['failed_links']}"
    )

    print(
        f"Failed nodes      : "
        f"{result['failed_nodes']}"
    )

    print()
    print("LINK STATISTICS")
    print("-" * 70)

    for link in result["link_statistics"]:

        print(
            f"{link['src']} -> {link['dst']} | "
            f"traffic={link['traffic_mbps']:.2f} Mbps | "
            f"capacity={link['capacity_mbps']:.2f} Mbps | "
            f"util={link['utilization']:.3f} | "
            f"delay={link['delay_ms']:.2f} ms | "
            f"loss={link['packet_loss']:.4f} | "
            f"failed={link['is_failed']}"
        )

    print()
    print("PATH STATISTICS")
    print("-" * 70)

    for path in result["path_statistics"]:

        print(
            f"Demand {path['demand_id']}: "
            f"{path['src']} -> {path['dst']} | "
            f"traffic={path['traffic_mbps']:.2f} Mbps | "
            f"path={path['path']} | "
            f"hops={path['hop_count']} | "
            f"delay={path['delay_ms']:.2f} ms | "
            f"loss={path['packet_loss']:.4f}"
        )

    print("=" * 70)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Test the SDN-MARL network simulator."
    )

    parser.add_argument(
        "--topology",
        required=True,
        help="Path to a GML topology.",
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
        help="Traffic scenario.",
    )

    parser.add_argument(
        "--demands",
        type=int,
        default=10,
        help="Number of traffic demands.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # LOAD TOPOLOGY
    # --------------------------------------------------------------

    print("=" * 70)
    print("LOADING TOPOLOGY")
    print("=" * 70)

    graph = load_topology(
        args.topology
    )

    print(
        f"Nodes     : {graph.number_of_nodes()}"
    )

    print(
        f"Edges     : {graph.number_of_edges()}"
    )

    print(
        f"Connected : {nx.is_connected(graph)}"
    )

    print("=" * 70)

    # --------------------------------------------------------------
    # GENERATE TRAFFIC
    # --------------------------------------------------------------

    generator = TrafficGenerator(
        graph=graph,
        seed=args.seed,
    )

    demands = generator.generate(
        scenario=args.scenario,
        num_demands=args.demands,
    )

    print()
    print("TRAFFIC DEMANDS")
    print("-" * 70)

    for index, demand in enumerate(demands):

        print(
            f"{index}: "
            f"{demand['src']} -> "
            f"{demand['dst']} | "
            f"{demand['traffic_mbps']:.2f} Mbps"
        )

    # --------------------------------------------------------------
    # SIMULATOR
    # --------------------------------------------------------------

    simulator = NetworkSimulator(
        graph=graph,
        k_paths=5,
        random_seed=args.seed,
    )

    result = simulator.install_demands(
        demands
    )

    print_result(result)

    # --------------------------------------------------------------
    # ACTION TEST
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("ACTION TEST")
    print("=" * 70)

    print()
    print("Applying:")
    print("Congestion Agent = 0 (maintain)")
    print("Failure Agent    = 0 (no action)")

    result = simulator.step(
        congestion_action=0,
        failure_action=0,
    )

    print(
        f"Throughput after action: "
        f"{result['total_throughput_mbps']:.2f} Mbps"
    )

    print(
        f"Average utilization: "
        f"{result['average_link_utilization']:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()