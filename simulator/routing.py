"""
Routing Engine for SDN-MARL

Responsibilities:
- Shortest path calculation
- K-shortest candidate paths
- Path validation
- Path metrics
- Path selection from an RL action
- Uses TopologyLoader for all GML loading

Person 1 component.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import networkx as nx

from .topology_loader import TopologyLoader


class RoutingEngine:
    """
    Routing engine operating on a NetworkX topology.

    Expected edge attributes:
        capacity_mbps
        weight
        delay_ms
        utilization
        packet_loss
        traffic_mbps
    """

    def __init__(
        self,
        graph: nx.Graph,
        k_paths: int = 5,
    ):
        self.graph = graph
        self.k_paths = max(1, int(k_paths))

    # ------------------------------------------------------------------
    # BASIC PATH OPERATIONS
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        src: Any,
        dst: Any,
    ) -> Optional[List[Any]]:
        """Return the shortest path between source and destination."""

        if src not in self.graph:
            raise ValueError(f"Source node {src} does not exist.")

        if dst not in self.graph:
            raise ValueError(f"Destination node {dst} does not exist.")

        if src == dst:
            return [src]

        try:
            return nx.shortest_path(
                self.graph,
                source=src,
                target=dst,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            return None

    def k_shortest_paths(
        self,
        src: Any,
        dst: Any,
        k: Optional[int] = None,
    ) -> List[List[Any]]:
        """
        Return up to k shortest simple paths.
        """

        if src not in self.graph:
            raise ValueError(f"Source node {src} does not exist.")

        if dst not in self.graph:
            raise ValueError(f"Destination node {dst} does not exist.")

        if src == dst:
            return [[src]]

        k = k if k is not None else self.k_paths
        k = max(1, int(k))

        paths = []

        try:
            generator = nx.shortest_simple_paths(
                self.graph,
                source=src,
                target=dst,
                weight="weight",
            )

            for path in generator:
                paths.append(path)

                if len(paths) >= k:
                    break

        except nx.NetworkXNoPath:
            pass

        return paths

    # ------------------------------------------------------------------
    # PATH VALIDATION
    # ------------------------------------------------------------------

    def is_valid_path(
        self,
        path: List[Any],
    ) -> bool:
        """Check whether a path exists in the current topology."""

        if not path:
            return False

        if len(path) == 1:
            return path[0] in self.graph

        for u, v in zip(path[:-1], path[1:]):
            if not self.graph.has_edge(u, v):
                return False

            edge_data = self.graph[u][v]

            # Failed links cannot be used.
            if edge_data.get("is_failed", 0):
                return False

        return True

    # ------------------------------------------------------------------
    # EDGE METRICS
    # ------------------------------------------------------------------

    def _edge_data(
        self,
        u: Any,
        v: Any,
    ) -> Dict[str, Any]:
        """Safely retrieve edge attributes."""

        if not self.graph.has_edge(u, v):
            raise ValueError(f"Edge {u}--{v} does not exist.")

        return self.graph[u][v]

    def _get_weight(
        self,
        u: Any,
        v: Any,
    ) -> float:
        data = self._edge_data(u, v)
        return float(data.get("weight", 1.0))

    def _get_delay(
        self,
        u: Any,
        v: Any,
    ) -> float:
        data = self._edge_data(u, v)
        return float(data.get("delay_ms", 1.0))

    def _get_capacity(
        self,
        u: Any,
        v: Any,
    ) -> float:
        data = self._edge_data(u, v)
        return float(data.get("capacity_mbps", 1000.0))

    def _get_utilization(
        self,
        u: Any,
        v: Any,
    ) -> float:
        data = self._edge_data(u, v)
        return float(data.get("utilization", 0.0))

    def _get_packet_loss(
        self,
        u: Any,
        v: Any,
    ) -> float:
        data = self._edge_data(u, v)
        return float(data.get("packet_loss", 0.0))

    # ------------------------------------------------------------------
    # PATH METRICS
    # ------------------------------------------------------------------

    def path_cost(
        self,
        path: List[Any],
    ) -> float:
        """Calculate total routing cost."""

        if not self.is_valid_path(path):
            return float("inf")

        if len(path) <= 1:
            return 0.0

        return sum(
            self._get_weight(u, v)
            for u, v in zip(path[:-1], path[1:])
        )

    def hop_count(
        self,
        path: List[Any],
    ) -> int:
        """Return number of links in a path."""

        if not path:
            return 0

        return max(0, len(path) - 1)

    def total_delay(
        self,
        path: List[Any],
    ) -> float:
        """Calculate total propagation/base delay in milliseconds."""

        if not self.is_valid_path(path):
            return float("inf")

        if len(path) <= 1:
            return 0.0

        return sum(
            self._get_delay(u, v)
            for u, v in zip(path[:-1], path[1:])
        )

    def bottleneck_utilization(
        self,
        path: List[Any],
    ) -> float:
        """
        Return maximum link utilization along the path.
        """

        if not self.is_valid_path(path):
            return 1.0

        if len(path) <= 1:
            return 0.0

        utilizations = [
            self._get_utilization(u, v)
            for u, v in zip(path[:-1], path[1:])
        ]

        if not utilizations:
            return 0.0

        return max(utilizations)

    def packet_loss(
        self,
        path: List[Any],
    ) -> float:
        """
        Estimate end-to-end packet loss.

        If individual link loss probabilities are p_i:

            P(loss) = 1 - product(1 - p_i)
        """

        if not self.is_valid_path(path):
            return 1.0

        if len(path) <= 1:
            return 0.0

        success_probability = 1.0

        for u, v in zip(path[:-1], path[1:]):
            loss = self._get_packet_loss(u, v)

            # Keep loss in a valid range.
            loss = min(max(loss, 0.0), 1.0)

            success_probability *= (1.0 - loss)

        return 1.0 - success_probability

    def bottleneck_capacity(
        self,
        path: List[Any],
    ) -> float:
        """Return minimum link capacity along the path."""

        if not self.is_valid_path(path):
            return 0.0

        if len(path) <= 1:
            return float("inf")

        capacities = [
            self._get_capacity(u, v)
            for u, v in zip(path[:-1], path[1:])
        ]

        if not capacities:
            return 0.0

        return min(capacities)

    # ------------------------------------------------------------------
    # PATH INFORMATION
    # ------------------------------------------------------------------

    def get_path_features(
        self,
        path: List[Any],
    ) -> Dict[str, float]:
        """
        Return useful metrics for a path.
        """

        return {
            "hop_count": self.hop_count(path),
            "bottleneck_utilization": self.bottleneck_utilization(path),
            "total_delay_ms": self.total_delay(path),
            "packet_loss": self.packet_loss(path),
            "bottleneck_capacity_mbps": self.bottleneck_capacity(path),
            "routing_cost": self.path_cost(path),
        }

    def get_candidate_paths(
        self,
        src: Any,
        dst: Any,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate paths and their metrics.

        Returns:
            [
                {
                    "path_index": 0,
                    "path": [...],
                    "hop_count": ...,
                    "bottleneck_utilization": ...,
                    "total_delay_ms": ...,
                    "packet_loss": ...,
                    "bottleneck_capacity_mbps": ...,
                    "routing_cost": ...
                },
                ...
            ]
        """

        paths = self.k_shortest_paths(src, dst, k)

        candidates = []

        for index, path in enumerate(paths):
            features = self.get_path_features(path)

            candidates.append(
                {
                    "path_index": index,
                    "path": path,
                    **features,
                }
            )

        return candidates

    # ------------------------------------------------------------------
    # ACTION → PATH
    # ------------------------------------------------------------------

    def action_to_path(
        self,
        candidates: List[Dict[str, Any]],
        action: int,
    ) -> Optional[List[Any]]:
        """
        Convert a path-selection action into a path.

        This function is retained for compatibility with the earlier
        path-selection interface.

        The newer Person 2 interface may use higher-level actions such
        as maintain/reroute/rate-limit instead.
        """

        if not candidates:
            return None

        if action < 0 or action >= len(candidates):
            return None

        path = candidates[action]["path"]

        if not self.is_valid_path(path):
            return None

        return path

    # ------------------------------------------------------------------
    # ALTERNATIVE ROUTING
    # ------------------------------------------------------------------

    def get_alternative_path(
        self,
        src: Any,
        dst: Any,
        current_path: Optional[List[Any]] = None,
    ) -> Optional[List[Any]]:
        """
        Return an alternative path.

        If current_path is provided, the first candidate that differs
        from it is returned.
        """

        candidates = self.k_shortest_paths(
            src,
            dst,
            self.k_paths,
        )

        if not candidates:
            return None

        if current_path is None:
            return candidates[0]

        for path in candidates:
            if path != current_path and self.is_valid_path(path):
                return path

        return None

    def get_path_excluding_failed_components(
        self,
        src: Any,
        dst: Any,
    ) -> Optional[List[Any]]:
        """
        Find a path while ignoring failed links and failed nodes.

        This is useful for the Failure Agent's rerouting action.
        """

        if src not in self.graph or dst not in self.graph:
            return None

        if self.graph.nodes[src].get("is_failed", 0):
            return None

        if self.graph.nodes[dst].get("is_failed", 0):
            return None

        # Build a temporary graph containing only healthy components.
        healthy_graph = self.graph.copy()

        failed_nodes = [
            node
            for node, data in healthy_graph.nodes(data=True)
            if data.get("is_failed", 0)
        ]

        healthy_graph.remove_nodes_from(failed_nodes)

        failed_edges = [
            (u, v)
            for u, v, data in healthy_graph.edges(data=True)
            if data.get("is_failed", 0)
        ]

        healthy_graph.remove_edges_from(failed_edges)

        if src not in healthy_graph or dst not in healthy_graph:
            return None

        try:
            return nx.shortest_path(
                healthy_graph,
                source=src,
                target=dst,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            return None


# ======================================================================
# TOPOLOGY LOADING
# ======================================================================

def load_graph(topology_path: str) -> nx.Graph:
    """
    Load topology using the project's TopologyLoader.

    TopologyLoader.load() handles:
    - duplicate GML edges
    - node normalization
    - capacity extraction
    - default network attributes
    """
    loader = TopologyLoader()
    return loader.load(topology_path)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the SDN-MARL routing engine."
    )

    parser.add_argument(
        "--topology",
        required=True,
        help="Path to a GML topology file.",
    )

    parser.add_argument(
        "--src",
        required=True,
        type=int,
        help="Source node.",
    )

    parser.add_argument(
        "--dst",
        required=True,
        type=int,
        help="Destination node.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of candidate paths.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # LOAD TOPOLOGY
    # --------------------------------------------------------------

    graph = load_graph(args.topology)

    # --------------------------------------------------------------
    # VALIDATE NODES
    # --------------------------------------------------------------

    if args.src not in graph:
        raise ValueError(
            f"Source node {args.src} does not exist. "
            f"Available nodes: {list(graph.nodes())}"
        )

    if args.dst not in graph:
        raise ValueError(
            f"Destination node {args.dst} does not exist. "
            f"Available nodes: {list(graph.nodes())}"
        )

    # --------------------------------------------------------------
    # CREATE ROUTING ENGINE
    # --------------------------------------------------------------

    router = RoutingEngine(
        graph=graph,
        k_paths=args.k,
    )

    # --------------------------------------------------------------
    # HEADER
    # --------------------------------------------------------------

    print("=" * 70)
    print("ROUTING TEST")
    print("=" * 70)

    print(f"Topology nodes : {graph.number_of_nodes()}")
    print(f"Topology edges : {graph.number_of_edges()}")
    print(f"Source         : {args.src}")
    print(f"Destination    : {args.dst}")
    print(f"K              : {args.k}")

    print("=" * 70)

    # --------------------------------------------------------------
    # SHORTEST PATH
    # --------------------------------------------------------------

    shortest = router.shortest_path(
        args.src,
        args.dst,
    )

    print()
    print("SHORTEST PATH")
    print("-" * 70)

    if shortest is None:
        print("No path exists.")

    else:
        print(" -> ".join(map(str, shortest)))
        print(f"Hops  : {router.hop_count(shortest)}")
        print(f"Cost  : {router.path_cost(shortest):.2f}")
        print(f"Delay : {router.total_delay(shortest):.2f} ms")

    # --------------------------------------------------------------
    # CANDIDATE PATHS
    # --------------------------------------------------------------

    candidates = router.get_candidate_paths(
        args.src,
        args.dst,
        args.k,
    )

    print()
    print("CANDIDATE PATHS")
    print("-" * 70)

    if not candidates:
        print("No candidate paths found.")

    else:
        for candidate in candidates:
            path_string = " -> ".join(
                map(str, candidate["path"])
            )

            print(
                f"Path {candidate['path_index']}: "
                f"{path_string}"
            )

            print(
                f"    hops={candidate['hop_count']}, "
                f"cost={candidate['routing_cost']:.2f}, "
                f"delay={candidate['total_delay_ms']:.2f} ms, "
                f"bottleneck_util="
                f"{candidate['bottleneck_utilization']:.3f}, "
                f"loss="
                f"{candidate['packet_loss']:.4f}, "
                f"capacity="
                f"{candidate['bottleneck_capacity_mbps']:.2f} Mbps"
            )

    # --------------------------------------------------------------
    # ACTION MASK
    # --------------------------------------------------------------

    print()
    print("ACTION MASK")
    print("-" * 70)

    action_mask = [
        router.is_valid_path(candidate["path"])
        for candidate in candidates
    ]

    print(action_mask)

    print("=" * 70)


if __name__ == "__main__":
    main()