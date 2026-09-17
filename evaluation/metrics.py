"""
Evaluation metric extraction utilities.

This module converts simulator result dictionaries into a consistent
set of evaluation metrics used by the MARL and shortest-path evaluations.
"""

from typing import Any, Dict, Iterable, Optional


METRIC_NAMES = [
    "max_utilization",
    "average_utilization",
    "delay",
    "packet_loss",
    "throughput",
    "congested_links",
]


def first_value(
    result: Dict[str, Any],
    keys: Iterable[str],
    default: float = 0.0,
) -> float:
    """
    Return the first available metric value from a list of possible keys.

    Parameters
    ----------
    result:
        Simulator result dictionary.
    keys:
        Possible keys for the requested metric.
    default:
        Value returned if none of the keys are present.

    Returns
    -------
    float
        Metric value converted to float when possible.
    """
    for key in keys:
        if key in result and result[key] is not None:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                continue

    return float(default)


def extract_metrics(result: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract the standard evaluation metrics from a simulator result.

    The simulator stores link-level utilization inside ``link_statistics``,
    while several other metrics use different names from the standard
    evaluation names.

    Parameters
    ----------
    result:
        Dictionary returned by the simulator.

    Returns
    -------
    dict
        Standardized evaluation metrics.
    """
    if result is None:
        result = {}

    if not isinstance(result, dict):
        raise TypeError(
            f"Expected simulator result to be dict, got {type(result).__name__}"
        )

    # ---------------------------------------------------------------
    # Maximum link utilization
    # ---------------------------------------------------------------
    # The simulator does not provide max_utilization as a top-level
    # field, so calculate it from link_statistics.
    link_stats = result.get("link_statistics", [])

    if not isinstance(link_stats, list):
        link_stats = []

    utilizations = []

    for link in link_stats:
        if not isinstance(link, dict):
            continue

        if "utilization" not in link:
            continue

        try:
            utilizations.append(float(link["utilization"]))
        except (TypeError, ValueError):
            continue

    max_utilization = max(utilizations) if utilizations else 0.0

    # ---------------------------------------------------------------
    # Average utilization
    # ---------------------------------------------------------------
    average_utilization = first_value(
        result,
        [
            "average_link_utilization",
            "avg_utilization",
        ],
    )

    # ---------------------------------------------------------------
    # Delay
    # ---------------------------------------------------------------
    delay = first_value(
        result,
        [
            "average_path_delay_ms",
            "delay",
            "average_delay",
            "mean_delay",
        ],
    )

    # ---------------------------------------------------------------
    # Packet loss
    # ---------------------------------------------------------------
    packet_loss = first_value(
        result,
        [
            "average_packet_loss",
            "packet_loss",
            "loss_rate",
        ],
    )

    # ---------------------------------------------------------------
    # Throughput
    # ---------------------------------------------------------------
    throughput = first_value(
        result,
        [
            "total_throughput_mbps",
            "throughput",
            "total_throughput",
        ],
    )

    # ---------------------------------------------------------------
    # Congested links
    # ---------------------------------------------------------------
    congested_links_raw = result.get("congested_links", [])

    if isinstance(congested_links_raw, list):
        congested_links = len(congested_links_raw)
    else:
        try:
            congested_links = int(congested_links_raw)
        except (TypeError, ValueError):
            congested_links = 0

    return {
        "max_utilization": max_utilization,
        "average_utilization": average_utilization,
        "delay": delay,
        "packet_loss": packet_loss,
        "throughput": throughput,
        "congested_links": congested_links,
    }


def metrics_from_simulator(simulator: Any) -> Dict[str, float]:
    """
    Extract standardized metrics directly from a simulator instance.

    The function tries the simulator's ``calculate_network_metrics()``
    method first.

    Parameters
    ----------
    simulator:
        NetworkSimulator instance.

    Returns
    -------
    dict
        Standardized evaluation metrics.
    """
    if not hasattr(simulator, "calculate_network_metrics"):
        raise AttributeError(
            "Simulator does not provide calculate_network_metrics()."
        )

    result = simulator.calculate_network_metrics()

    return extract_metrics(result)


def add_metadata(
    metrics: Dict[str, float],
    method: str,
    topology: str,
    num_demands: int,
) -> Dict[str, Any]:
    """
    Add evaluation metadata to a metric dictionary.

    Parameters
    ----------
    metrics:
        Standardized metrics.
    method:
        Evaluation method, e.g. ``marl`` or ``shortest_path``.
    topology:
        Topology filename.
    num_demands:
        Number of traffic demands.

    Returns
    -------
    dict
        Metrics plus metadata.
    """
    output = dict(metrics)

    output["method"] = method
    output["topology"] = topology
    output["num_demands"] = num_demands

    return output