"""Reproducible schema and default training settings."""
from dataclasses import dataclass


FEATURE_NAMES = (
    "utilization", "utilization_known", "log_capacity_ratio",
    "hop_progress", "is_failed", "weighted_cost_progress",
)
FEATURE_DIM = len(FEATURE_NAMES)
CAPACITY_REFERENCE_MBPS = 100.0


@dataclass(frozen=True)
class RoutingConfig:
    demand_mbps: float = 25.0
    max_hops: int = 12
    # Congestion reward exactly matches the established weighted-Dijkstra
    # baseline: sum(1 + 3 * observed background utilization), conditioned on delivery.
    hop_cost: float = 1.0
    congestion_weight: float = 3.0
    delivery_reward: float = 2.0
    failure_penalty: float = 2.0
    hop_penalty: float = 0.25
    reward_scale: float = 10.0
    learning_rate: float = 0.001
    hidden_dim: int = 64
    gamma: float = 0.97
    capacity_reference_mbps: float = CAPACITY_REFERENCE_MBPS
    train_cross_link_probability_min: float = 0.15
    train_cross_link_probability_max: float = 0.45
