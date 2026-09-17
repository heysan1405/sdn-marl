"""
Reward Function for SDN-MARL.

Person 1 component.

The reward evaluates the current network state using:

    - Throughput
    - Congestion
    - Delay
    - Packet loss
    - Failed components

Reward formulation:

    R = wT*T - wC*C - wD*D - wL*L - wF*F

Default weights:

    Throughput  : 0.40
    Congestion  : 0.25
    Delay       : 0.15
    Packet loss : 0.15
    Failure     : 0.05
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import networkx as nx

from simulator.topology_loader import TopologyLoader
from simulator.traffic_generator import TrafficGenerator
from simulator.network_simulator import NetworkSimulator


class RewardCalculator:
    """
    Calculates an RL reward from network performance.

    The calculator does not modify the network.

    It only evaluates the metrics produced by NetworkSimulator.
    """

    # ==============================================================
    # DEFAULT WEIGHTS
    # ==============================================================

    DEFAULT_THROUGHPUT_WEIGHT = 0.40
    DEFAULT_CONGESTION_WEIGHT = 0.25
    DEFAULT_DELAY_WEIGHT = 0.15
    DEFAULT_LOSS_WEIGHT = 0.15
    DEFAULT_FAILURE_WEIGHT = 0.05

    # ==============================================================
    # NORMALIZATION LIMITS
    # ==============================================================

    # Delay at or above this value is considered very poor.
    MAX_DELAY_MS = 100.0

    def __init__(
        self,
        throughput_weight: float = DEFAULT_THROUGHPUT_WEIGHT,
        congestion_weight: float = DEFAULT_CONGESTION_WEIGHT,
        delay_weight: float = DEFAULT_DELAY_WEIGHT,
        loss_weight: float = DEFAULT_LOSS_WEIGHT,
        failure_weight: float = DEFAULT_FAILURE_WEIGHT,
    ):
        """
        Initialize the reward calculator.

        The weights are normalized automatically so their sum is 1.
        """

        weights = {
            "throughput": float(throughput_weight),
            "congestion": float(congestion_weight),
            "delay": float(delay_weight),
            "loss": float(loss_weight),
            "failure": float(failure_weight),
        }

        if any(value < 0 for value in weights.values()):
            raise ValueError(
                "Reward weights cannot be negative."
            )

        total = sum(weights.values())

        if total <= 0:
            raise ValueError(
                "At least one reward weight must be greater than zero."
            )

        self.throughput_weight = (
            weights["throughput"] / total
        )

        self.congestion_weight = (
            weights["congestion"] / total
        )

        self.delay_weight = (
            weights["delay"] / total
        )

        self.loss_weight = (
            weights["loss"] / total
        )

        self.failure_weight = (
            weights["failure"] / total
        )

    # ==============================================================
    # NORMALIZATION
    # ==============================================================

    @staticmethod
    def clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """
        Clamp a value into [minimum, maximum].
        """

        return max(
            minimum,
            min(value, maximum),
        )

    def normalize_throughput(
        self,
        throughput_mbps: float,
        requested_mbps: float,
    ) -> float:
        """
        Normalize throughput.

        0:
            no successful throughput

        1:
            all requested traffic successfully delivered
        """

        throughput = max(
            0.0,
            float(throughput_mbps),
        )

        requested = max(
            0.0,
            float(requested_mbps),
        )

        if requested <= 0:
            return 1.0

        return self.clamp(
            throughput / requested
        )

    def normalize_congestion(
        self,
        congestion_ratio: float,
    ) -> float:
        """
        Normalize congestion.

        The simulator provides congestion_ratio:

            0 = no congested links
            1 = every healthy link is congested

        Therefore it is already normalized.
        """

        return self.clamp(
            float(congestion_ratio)
        )

    def normalize_delay(
        self,
        delay_ms: float,
    ) -> float:
        """
        Normalize average path delay.

        0:
            no delay / ideal

        1:
            delay >= MAX_DELAY_MS
        """

        delay = max(
            0.0,
            float(delay_ms),
        )

        return self.clamp(
            delay / self.MAX_DELAY_MS
        )

    def normalize_packet_loss(
        self,
        packet_loss: float,
    ) -> float:
        """
        Normalize packet loss.

        Packet loss is already represented as:

            0 = no loss
            1 = complete loss
        """

        return self.clamp(
            float(packet_loss)
        )

    def normalize_failures(
        self,
        result: Dict[str, Any],
    ) -> float:
        """
        Calculate normalized failure severity.

        Failure severity is based on the fraction of network
        components that are failed.

        Links and nodes are both considered.
        """

        failed_links = len(
            result.get("failed_links", [])
        )

        failed_nodes = len(
            result.get("failed_nodes", [])
        )

        link_statistics = result.get(
            "link_statistics",
            [],
        )

        node_statistics = result.get(
            "node_statistics",
            [],
        )

        total_links = len(link_statistics)
        total_nodes = len(node_statistics)

        total_components = (
            total_links + total_nodes
        )

        if total_components <= 0:
            return 0.0

        failed_components = (
            failed_links + failed_nodes
        )

        return self.clamp(
            failed_components / total_components
        )

    # ==============================================================
    # REWARD COMPONENTS
    # ==============================================================

    def calculate_components(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate the individual reward components.
        """

        requested = float(
            result.get(
                "total_requested_traffic_mbps",
                0.0,
            )
        )

        throughput = float(
            result.get(
                "total_throughput_mbps",
                0.0,
            )
        )

        congestion = float(
            result.get(
                "congestion_ratio",
                0.0,
            )
        )

        delay = float(
            result.get(
                "average_path_delay_ms",
                0.0,
            )
        )

        packet_loss = float(
            result.get(
                "average_packet_loss",
                0.0,
            )
        )

        throughput_score = (
            self.normalize_throughput(
                throughput,
                requested,
            )
        )

        congestion_score = (
            self.normalize_congestion(
                congestion
            )
        )

        delay_score = (
            self.normalize_delay(
                delay
            )
        )

        loss_score = (
            self.normalize_packet_loss(
                packet_loss
            )
        )

        failure_score = (
            self.normalize_failures(
                result
            )
        )

        return {
            "throughput": throughput_score,
            "congestion": congestion_score,
            "delay": delay_score,
            "packet_loss": loss_score,
            "failure": failure_score,
        }

    # ==============================================================
    # REWARD
    # ==============================================================

    def calculate(
        self,
        result: Dict[str, Any],
    ) -> float:
        """
        Calculate the final scalar reward.
        """

        components = self.calculate_components(
            result
        )

        reward = (
            self.throughput_weight
            * components["throughput"]
        )

        reward -= (
            self.congestion_weight
            * components["congestion"]
        )

        reward -= (
            self.delay_weight
            * components["delay"]
        )

        reward -= (
            self.loss_weight
            * components["packet_loss"]
        )

        reward -= (
            self.failure_weight
            * components["failure"]
        )

        return float(reward)

    # ==============================================================
    # DETAILED REWARD
    # ==============================================================

    def calculate_detailed(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return the reward together with all components.

        Useful for debugging and experiments.
        """

        components = self.calculate_components(
            result
        )

        positive_contribution = (
            self.throughput_weight
            * components["throughput"]
        )

        congestion_penalty = (
            self.congestion_weight
            * components["congestion"]
        )

        delay_penalty = (
            self.delay_weight
            * components["delay"]
        )

        loss_penalty = (
            self.loss_weight
            * components["packet_loss"]
        )

        failure_penalty = (
            self.failure_weight
            * components["failure"]
        )

        reward = (
            positive_contribution
            - congestion_penalty
            - delay_penalty
            - loss_penalty
            - failure_penalty
        )

        return {
            "reward": float(reward),

            "components": components,

            "weights": {
                "throughput": self.throughput_weight,
                "congestion": self.congestion_weight,
                "delay": self.delay_weight,
                "packet_loss": self.loss_weight,
                "failure": self.failure_weight,
            },

            "contributions": {
                "throughput": positive_contribution,
                "congestion_penalty": congestion_penalty,
                "delay_penalty": delay_penalty,
                "loss_penalty": loss_penalty,
                "failure_penalty": failure_penalty,
            },
        }


# ==================================================================
# CLI TEST
# ==================================================================

def load_topology(
    topology_path: str,
) -> nx.Graph:
    """
    Load topology through the project's TopologyLoader.
    """

    loader = TopologyLoader()

    return loader.load(topology_path)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Test the SDN-MARL reward function."
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
    )

    parser.add_argument(
        "--demands",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # LOAD TOPOLOGY
    # --------------------------------------------------------------

    print("=" * 70)
    print("REWARD TEST")
    print("=" * 70)

    graph = load_topology(
        args.topology
    )

    print(
        f"Topology nodes : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Topology edges : "
        f"{graph.number_of_edges()}"
    )

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

    # --------------------------------------------------------------
    # SIMULATE
    # --------------------------------------------------------------

    simulator = NetworkSimulator(
        graph=graph,
        k_paths=5,
        random_seed=args.seed,
    )

    result = simulator.install_demands(
        demands
    )

    # --------------------------------------------------------------
    # REWARD
    # --------------------------------------------------------------

    calculator = RewardCalculator()

    detailed = calculator.calculate_detailed(
        result
    )

    print()
    print("NETWORK METRICS")
    print("-" * 70)

    print(
        f"Requested traffic : "
        f"{result['total_requested_traffic_mbps']:.2f} Mbps"
    )

    print(
        f"Throughput        : "
        f"{result['total_throughput_mbps']:.2f} Mbps"
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
        f"Packet loss       : "
        f"{result['average_packet_loss']:.4f}"
    )

    print()
    print("REWARD COMPONENTS")
    print("-" * 70)

    components = detailed["components"]

    print(
        f"Throughput score  : "
        f"{components['throughput']:.4f}"
    )

    print(
        f"Congestion score  : "
        f"{components['congestion']:.4f}"
    )

    print(
        f"Delay score       : "
        f"{components['delay']:.4f}"
    )

    print(
        f"Packet loss score : "
        f"{components['packet_loss']:.4f}"
    )

    print(
        f"Failure score     : "
        f"{components['failure']:.4f}"
    )

    print()
    print("REWARD CONTRIBUTIONS")
    print("-" * 70)

    contributions = detailed[
        "contributions"
    ]

    print(
        f"Throughput        : "
        f"+{contributions['throughput']:.4f}"
    )

    print(
        f"Congestion        : "
        f"-{contributions['congestion_penalty']:.4f}"
    )

    print(
        f"Delay             : "
        f"-{contributions['delay_penalty']:.4f}"
    )

    print(
        f"Packet loss       : "
        f"-{contributions['loss_penalty']:.4f}"
    )

    print(
        f"Failures          : "
        f"-{contributions['failure_penalty']:.4f}"
    )

    print()
    print("=" * 70)
    print(
        f"FINAL REWARD: "
        f"{detailed['reward']:.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()