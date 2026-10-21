"""Canonical candidate feature construction used in training and deployment."""
import math

import networkx as nx
import numpy as np

from marl_live.config import FEATURE_DIM, CAPACITY_REFERENCE_MBPS


def _shortest_costs(graph, destination, weight):
    reverse = graph.reverse(copy=False) if graph.is_directed() else graph
    return nx.single_source_dijkstra_path_length(reverse, destination, weight=weight)


def candidate_observation(graph, current, destination, visited, has_failure=None,
                          capacity_reference_mbps=CAPACITY_REFERENCE_MBPS,
                          hop_cost=1.0, congestion_weight=3.0):
    """Construct a consistent state and mask.

    Candidate actions are masked when they are failed, already visited, or
    cannot reach the destination without revisiting the current route. Features
    include both hop progress and progress against the same additive edge cost
    optimized by the non-RL congestion baseline.
    """
    healthy = graph.__class__()
    healthy.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if not bool(data.get("is_failed", False)):
            healthy.add_edge(u, v, **data)

    # The live and simulated objective treats unknown utilization as zero while
    # keeping its missingness bit explicit. This is a documented route-cost
    # fallback, not an assertion that the unmeasured utilization is zero.
    for _, _, data in healthy.edges(data=True):
        data["_routing_cost"] = float(hop_cost) + float(congestion_weight) * (
            float(data["utilization"]) if data.get("utilization") is not None else
            (float(data.get("background_mbps", 0.0)) / float(data["capacity_mbps"])
             if data.get("capacity_mbps") and "background_mbps" in data else 0.0)
        )
        data["_routing_cost"] = max(float(hop_cost), data["_routing_cost"])
    route_graph = healthy.copy()
    route_graph.remove_nodes_from(node for node in visited
                                  if node in route_graph and node != current)
    try:
        hop_to_dest = nx.single_source_shortest_path_length(
            route_graph.reverse(copy=False) if route_graph.is_directed() else route_graph, destination
        )
        weighted_to_dest = _shortest_costs(route_graph, destination, "_routing_cost")
    except nx.NodeNotFound:
        hop_to_dest, weighted_to_dest = {}, {}

    candidates = sorted(graph.neighbors(current), key=lambda n: str(n))
    rows, mask = [], []
    failed_present = False
    current_hops = hop_to_dest.get(current)
    current_cost = weighted_to_dest.get(current)
    for neighbor in candidates:
        edge = graph[current][neighbor]
        capacity = edge.get("capacity_mbps")
        known = edge.get("utilization") is not None or (
            capacity is not None and float(capacity) > 0 and "background_mbps" in edge
        )
        if edge.get("utilization") is not None:
            utilization = float(edge["utilization"])
        elif known:
            utilization = float(edge["background_mbps"]) / float(capacity)
        else:
            utilization = 0.0
        failed = bool(edge.get("is_failed", False))
        visited_flag = neighbor in visited
        failed_present |= failed
        capacity_feature = (math.log1p(float(capacity)) / math.log1p(capacity_reference_mbps)
                            if capacity is not None and float(capacity) > 0 else 0.0)

        # Remove already traversed nodes for the feasibility check; this
        # prevents valid-local but globally dead-ending choices.
        remaining = route_graph.copy()
        if current in remaining:
            remaining.remove_node(current)
        feasible = neighbor == destination
        if not feasible and neighbor in remaining and destination in remaining:
            feasible = nx.has_path(remaining, neighbor, destination)
        safe = not failed and not visited_flag and feasible

        h_next = hop_to_dest.get(neighbor)
        hop_progress = ((current_hops - 1 - h_next) / max(current_hops, 1)
                        if current_hops is not None and h_next is not None else -1.0)
        c_next = weighted_to_dest.get(neighbor)
        edge_cost = float(hop_cost) + float(congestion_weight) * (utilization if known else 0.0)
        candidate_cost = edge_cost + c_next if c_next is not None else float("inf")
        weighted_progress = ((current_cost - candidate_cost) / max(current_cost, 1.0)
                             if current_cost is not None and math.isfinite(candidate_cost) else -1.0)

        rows.append([utilization if known else 0.0, float(known), capacity_feature,
                     float(hop_progress), float(failed), float(weighted_progress)])
        mask.append(safe)
    return {
        "candidate_features": np.asarray(rows, dtype=np.float32).reshape((-1, FEATURE_DIM)),
        "action_mask": np.asarray(mask, dtype=np.bool_),
        "candidate_nodes": candidates, "current_node": current,
        "destination": destination,
        "has_failure": bool(failed_present if has_failure is None else has_failure),
    }
