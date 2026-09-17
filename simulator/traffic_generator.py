
"""
Topology-independent traffic demand generator.

Generates traffic demands for any NetworkX topology.

Supported scenarios:
    - normal
    - heavy
    - single_hotspot
    - multiple_hotspots
    - traffic_spike
    - random

Output format:
    [
        {
            "src": <source_node>,
            "dst": <destination_node>,
            "traffic_mbps": <traffic_volume>
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np


TrafficDemand = Dict[str, Any]


class TrafficGenerator:
    """
    Generate synthetic traffic demands for an arbitrary topology.

    The generator does NOT assume:
        - a fixed number of nodes
        - a particular topology
        - specific node IDs
        - a fixed number of links
    """

    SCENARIOS = {
        "normal",
        "heavy",
        "single_hotspot",
        "multiple_hotspots",
        "traffic_spike",
        "random",
    }

    def __init__(
        self,
        graph: nx.Graph,
        seed: int | None = 42,
        min_traffic_mbps: float = 10.0,
        max_traffic_mbps: float = 500.0,
        demands_per_node: float = 1.0,
    ) -> None:
        if graph is None:
            raise ValueError("graph cannot be None")

        if graph.number_of_nodes() < 2:
            raise ValueError("Topology must contain at least 2 nodes")

        if min_traffic_mbps <= 0:
            raise ValueError("min_traffic_mbps must be greater than 0")

        if max_traffic_mbps < min_traffic_mbps:
            raise ValueError(
                "max_traffic_mbps must be >= min_traffic_mbps"
            )

        if demands_per_node <= 0:
            raise ValueError("demands_per_node must be greater than 0")

        self.graph = graph
        self.min_traffic_mbps = float(min_traffic_mbps)
        self.max_traffic_mbps = float(max_traffic_mbps)
        self.demands_per_node = float(demands_per_node)

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.nodes: List[Any] = list(graph.nodes())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        scenario: str = "normal",
        num_demands: int | None = None,
    ) -> List[TrafficDemand]:
        """
        Generate traffic according to the requested scenario.
        """

        scenario = scenario.lower().strip()

        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Choose from: {sorted(self.SCENARIOS)}"
            )

        if num_demands is None:
            num_demands = max(
                1,
                int(
                    round(
                        self.graph.number_of_nodes()
                        * self.demands_per_node
                    )
                ),
            )

        if num_demands <= 0:
            raise ValueError("num_demands must be greater than 0")

        if scenario == "normal":
            return self._normal(num_demands)

        if scenario == "heavy":
            return self._heavy(num_demands)

        if scenario == "single_hotspot":
            return self._single_hotspot(num_demands)

        if scenario == "multiple_hotspots":
            return self._multiple_hotspots(num_demands)

        if scenario == "traffic_spike":
            return self._traffic_spike(num_demands)

        if scenario == "random":
            return self._random(num_demands)

        raise RuntimeError("Unreachable scenario")

    # ------------------------------------------------------------------
    # Scenario generators
    # ------------------------------------------------------------------

    def _normal(self, num_demands: int) -> List[TrafficDemand]:
        """
        Normal traffic:
        Mostly moderate traffic with occasional larger flows.
        """

        demands = []

        for _ in range(num_demands):
            src, dst = self._random_pair()

            traffic = self.np_rng.lognormal(
                mean=np.log(60.0),
                sigma=0.55,
            )

            traffic = float(
                np.clip(
                    traffic,
                    self.min_traffic_mbps,
                    self.max_traffic_mbps * 0.6,
                )
            )

            demands.append(
                self._make_demand(src, dst, traffic)
            )

        return demands

    def _heavy(self, num_demands: int) -> List[TrafficDemand]:
        """
        Heavy traffic:
        Generates significantly larger flows.
        """

        demands = []

        lower = max(
            self.min_traffic_mbps,
            self.max_traffic_mbps * 0.40,
        )

        for _ in range(num_demands):
            src, dst = self._random_pair()

            traffic = self.rng.uniform(
                lower,
                self.max_traffic_mbps,
            )

            demands.append(
                self._make_demand(src, dst, traffic)
            )

        return demands

    def _single_hotspot(
        self,
        num_demands: int,
    ) -> List[TrafficDemand]:
        """
        Single hotspot:

        One node becomes significantly busier than the rest.
        Traffic is generated either toward or from that node.
        """

        hotspot = self.rng.choice(self.nodes)

        other_nodes = [
            node for node in self.nodes
            if node != hotspot
        ]

        if not other_nodes:
            return []

        demands = []

        for _ in range(num_demands):
            other = self.rng.choice(other_nodes)

            # Randomly choose whether hotspot is source or destination.
            if self.rng.random() < 0.5:
                src = hotspot
                dst = other
            else:
                src = other
                dst = hotspot

            traffic = self.rng.uniform(
                max(
                    self.min_traffic_mbps,
                    self.max_traffic_mbps * 0.50,
                ),
                self.max_traffic_mbps,
            )

            demands.append(
                self._make_demand(src, dst, traffic)
            )

        return demands

    def _multiple_hotspots(
        self,
        num_demands: int,
    ) -> List[TrafficDemand]:
        """
        Multiple hotspots:

        Select several busy nodes and generate traffic
        involving those nodes.
        """

        hotspot_count = max(
            2,
            int(round(len(self.nodes) * 0.20)),
        )

        hotspot_count = min(
            hotspot_count,
            len(self.nodes),
        )

        hotspots = self.rng.sample(
            self.nodes,
            hotspot_count,
        )

        demands = []

        for _ in range(num_demands):
            if self.rng.random() < 0.80:
                src = self.rng.choice(hotspots)

                possible_destinations = [
                    node
                    for node in self.nodes
                    if node != src
                ]

                dst = self.rng.choice(
                    possible_destinations
                )
            else:
                src, dst = self._random_pair()

            traffic = self.rng.uniform(
                max(
                    self.min_traffic_mbps,
                    self.max_traffic_mbps * 0.35,
                ),
                self.max_traffic_mbps,
            )

            demands.append(
                self._make_demand(src, dst, traffic)
            )

        return demands

    def _traffic_spike(
        self,
        num_demands: int,
    ) -> List[TrafficDemand]:
        """
        Traffic spike:

        Most traffic is normal, but a subset of flows suddenly
        becomes much larger.
        """

        demands = []

        for _ in range(num_demands):
            src, dst = self._random_pair()

            if self.rng.random() < 0.30:
                # Spike traffic.
                traffic = self.rng.uniform(
                    max(
                        self.min_traffic_mbps,
                        self.max_traffic_mbps * 0.70,
                    ),
                    self.max_traffic_mbps,
                )
            else:
                # Normal traffic.
                traffic = self.np_rng.lognormal(
                    mean=np.log(60.0),
                    sigma=0.50,
                )

                traffic = float(
                    np.clip(
                        traffic,
                        self.min_traffic_mbps,
                        self.max_traffic_mbps * 0.60,
                    )
                )

            demands.append(
                self._make_demand(src, dst, traffic)
            )

        return demands

    def _random(
        self,
        num_demands: int,
    ) -> List[TrafficDemand]:
        """
        Completely random traffic scenario.
        """

        demands = []

        for _ in range(num_demands):
            src, dst = self._random_pair()

            traffic = self.rng.uniform(
                self.min_traffic_mbps,
                self.max_traffic_mbps,
            )

            demands.append(
                self._make_demand(src, dst, traffic)
            )

        return demands

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_pair(self) -> tuple[Any, Any]:
        """
        Select two distinct nodes.
        """

        src, dst = self.rng.sample(
            self.nodes,
            2,
        )

        return src, dst

    @staticmethod
    def _make_demand(
        src: Any,
        dst: Any,
        traffic_mbps: float,
    ) -> TrafficDemand:
        """
        Create a standardized traffic-demand dictionary.
        """

        return {
            "src": src,
            "dst": dst,
            "traffic_mbps": round(
                float(traffic_mbps),
                3,
            ),
        }


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------

def generate_traffic(
    graph: nx.Graph,
    scenario: str = "normal",
    num_demands: int | None = None,
    seed: int | None = 42,
) -> List[TrafficDemand]:
    """
    Convenience wrapper.
    """

    generator = TrafficGenerator(
        graph=graph,
        seed=seed,
    )

    return generator.generate(
        scenario=scenario,
        num_demands=num_demands,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate topology-independent traffic demands."
    )

    parser.add_argument(
        "--topology",
        required=True,
        help="Path to a GML topology file.",
    )

    parser.add_argument(
        "--scenario",
        default="normal",
        choices=sorted(TrafficGenerator.SCENARIOS),
        help="Traffic scenario.",
    )

    parser.add_argument(
        "--demands",
        type=int,
        default=None,
        help="Number of traffic demands.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--min-traffic",
        type=float,
        default=10.0,
        help="Minimum traffic in Mbps.",
    )

    parser.add_argument(
        "--max-traffic",
        type=float,
        default=500.0,
        help="Maximum traffic in Mbps.",
    )

    args = parser.parse_args()

    # Read topology.
    graph = nx.read_gml(
        args.topology,
        label=None,
    )

    # Convert node IDs to strings/integers consistently.
    # NetworkX may load GML IDs as integers or strings depending
    # on the topology.
    graph = nx.Graph(graph)

    generator = TrafficGenerator(
        graph=graph,
        seed=args.seed,
        min_traffic_mbps=args.min_traffic,
        max_traffic_mbps=args.max_traffic,
    )

    demands = generator.generate(
        scenario=args.scenario,
        num_demands=args.demands,
    )

    print("=" * 70)
    print("TRAFFIC GENERATION")
    print("=" * 70)

    print(f"Topology nodes : {graph.number_of_nodes()}")
    print(f"Topology edges : {graph.number_of_edges()}")
    print(f"Scenario       : {args.scenario}")
    print(f"Demands        : {len(demands)}")

    print("=" * 70)
    print("DEMANDS")
    print("=" * 70)

    for index, demand in enumerate(demands):
        print(
            f"{index:3d}: "
            f"{demand['src']} -> {demand['dst']} | "
            f"{demand['traffic_mbps']:.3f} Mbps"
        )

    print("=" * 70)

    if demands:
        total = sum(
            demand["traffic_mbps"]
            for demand in demands
        )

        average = total / len(demands)

        print(
            f"Total traffic    : {total:.3f} Mbps"
        )
        print(
            f"Average demand   : {average:.3f} Mbps"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()

