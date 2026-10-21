"""Schema helpers for the archived Datanet/SDN simulation dataset."""

from __future__ import annotations


def link_usage_by_edge(graph, link_usage):
    """Map row-major ``linkUsage.txt`` matrix entries onto graph arcs.

    Datanet stores link state at matrix offset ``source * N + destination``.
    This is independent of NetworkX's edge iteration order. Node labels must
    match the zero-based matrix indices used by the archive.
    """
    node_count = graph.number_of_nodes()
    expected_nodes = set(range(node_count))
    if set(graph.nodes) != expected_nodes:
        raise ValueError(
            "Dataset topology node labels must be contiguous integers 0..N-1"
        )
    if len(link_usage) < node_count * node_count:
        raise ValueError(
            f"linkUsage has {len(link_usage)} matrix fields; expected at least "
            f"{node_count * node_count} for {node_count} nodes"
        )

    mapped = {}
    for source, destination, attributes in graph.edges(data=True):
        metrics = link_usage[source * node_count + destination]
        if metrics is None:
            raise ValueError(
                f"Topology arc {source}->{destination} has no linkUsage measurement"
            )
        if len(metrics) < 8:
            raise ValueError(
                f"Topology arc {source}->{destination} has {len(metrics)} link metrics; "
                "expected 8 (link and queue measurements)"
            )
        mapped[(source, destination)] = {
            "bandwidth": float(attributes.get("bandwidth", 0)),
            "weight": float(attributes.get("weight", 0)),
            "metrics": [float(value) for value in metrics[:8]],
        }
    return mapped


def candidate_features(state, hop_distance):
    """Return the established 11-feature routing input in its fixed order."""
    metrics = state["metrics"]
    return [
        state["bandwidth"],
        state["weight"],
        metrics[0],  # link utilization
        metrics[1],  # link losses
        metrics[2],  # average packet size
        metrics[3],  # queue utilization
        metrics[4],  # queue losses
        metrics[5],  # average port occupancy
        metrics[6],  # maximum queue occupancy
        metrics[7],  # queue average packet size
        hop_distance,
    ]
