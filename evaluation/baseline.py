"""
Shortest-path baseline for SDN evaluation.

The baseline uses the NetworkSimulator's built-in shortest-path
routing instead of the learned Congestion and Failure agents.

NetworkSimulator.install_demands() is responsible for:
    - clearing previous dynamic state
    - calculating shortest paths
    - installing active paths
    - applying traffic
    - calculating network metrics
"""

import os
import random
from typing import Dict

from evaluation.metrics import extract_metrics

from simulator.network_simulator import NetworkSimulator
from simulator.topology_loader import TopologyLoader
from simulator.traffic_generator import TrafficGenerator


class ShortestPathBaseline:
    """
    Shortest-path routing baseline.

    Pipeline:

        topology
            |
            v
        traffic demands
            |
            v
        NetworkSimulator
            |
            |-- shortest-path routing
            |-- traffic application
            |
            v
        network metrics
    """

    def __init__(
        self,
        topology_path,
        traffic_scenario="normal",
        num_demands=10,
        k_paths=5,
        random_seed=42,
    ):
        self.topology_path = topology_path
        self.traffic_scenario = traffic_scenario

        self.num_demands = int(num_demands)

        self.k_paths = max(
            1,
            int(k_paths),
        )

        self.random_seed = random_seed

        if self.num_demands <= 0:
            raise ValueError(
                "num_demands must be greater than zero."
            )

    def load_graph(self):
        """
        Load the topology graph using the project's
        TopologyLoader.
        """

        loader = TopologyLoader()

        graph = loader.load(
            topology_file=self.topology_path
        )

        return graph

    def run(self) -> Dict:
        """
        Execute the shortest-path baseline.

        NetworkSimulator.install_demands() automatically
        calculates and installs shortest paths.

        Returns
        -------
        dict
            Standardized network metrics.
        """

        # ----------------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------------

        random.seed(
            self.random_seed
        )

        # ----------------------------------------------------------
        # Load topology
        # ----------------------------------------------------------

        graph = self.load_graph()

        # ----------------------------------------------------------
        # Generate traffic
        # ----------------------------------------------------------

        traffic_generator = TrafficGenerator(
            graph=graph,
            seed=self.random_seed,
        )

        demands = traffic_generator.generate(
            scenario=self.traffic_scenario,
            num_demands=self.num_demands,
        )

        # ----------------------------------------------------------
        # Create simulator
        # ----------------------------------------------------------

        simulator = NetworkSimulator(
            graph=graph,
            k_paths=self.k_paths,
            random_seed=self.random_seed,
        )

        # ----------------------------------------------------------
        # Install demands
        #
        # NetworkSimulator.install_demands() performs:
        #   1. dynamic-state reset
        #   2. shortest-path calculation
        #   3. active path installation
        #   4. traffic application
        #   5. metric calculation
        # ----------------------------------------------------------

        result = simulator.install_demands(
            demands
        )

        # ----------------------------------------------------------
        # Extract standardized metrics
        # ----------------------------------------------------------

        metrics = extract_metrics(
            result
        )

        # ----------------------------------------------------------
        # Experiment metadata
        # ----------------------------------------------------------

        metrics["method"] = (
            "shortest_path"
        )

        metrics["topology"] = (
            os.path.basename(
                self.topology_path
            )
        )

        metrics["num_demands"] = (
            self.num_demands
        )

        return metrics


def run_shortest_path_baseline(
    topology_path,
    traffic_scenario="normal",
    num_demands=10,
    k_paths=5,
    random_seed=42,
):
    """
    Convenience function for running the
    shortest-path baseline.
    """

    baseline = ShortestPathBaseline(
        topology_path=topology_path,
        traffic_scenario=traffic_scenario,
        num_demands=num_demands,
        k_paths=k_paths,
        random_seed=random_seed,
    )

    return baseline.run()


if __name__ == "__main__":

    topology = os.path.join(
        "InternetTopologyZoo",
        "gml",
        "Abilene.gml",
    )

    results = run_shortest_path_baseline(
        topology_path=topology,
        traffic_scenario="normal",
        num_demands=10,
        k_paths=5,
        random_seed=42,
    )

    print("=" * 70)
    print("SHORTEST-PATH BASELINE")
    print("=" * 70)

    for key, value in results.items():
        print(
            f"{key:25s}: {value}"
        )