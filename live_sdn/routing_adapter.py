"""Pure routing helpers shared by the live Ryu app and unit tests.

This module intentionally does not manufacture simulator-only model features.
Live link utilization is populated only when the operator supplies a known
capacity and two valid OpenFlow counter samples are available.
"""
from __future__ import annotations

import networkx as nx


def build_live_graph(switches, links, failed_links=(), metrics=None):
    """Build a directed graph from Ryu switch/link records.

    ``links`` are objects with ``src``/``dst`` (dpid, port_no), or tuples
    ``(src_dpid, src_port, dst_dpid, dst_port)``. ``metrics`` maps directed
    (src_dpid, dst_dpid) pairs to measured attributes.
    """
    graph = nx.DiGraph()
    for switch in switches:
        dpid = getattr(getattr(switch, "dp", None), "id", switch)
        graph.add_node(int(dpid))
    failed = {tuple(sorted((int(u), int(v)))) for u, v in failed_links}
    metrics = metrics or {}
    for link in links:
        if isinstance(link, tuple):
            src, src_port, dst, dst_port = link
        else:
            src, src_port = link.src.dpid, link.src.port_no
            dst, dst_port = link.dst.dpid, link.dst.port_no
        src, dst = int(src), int(dst)
        if tuple(sorted((src, dst))) in failed:
            continue
        attrs = dict(metrics.get((src, dst), {}))
        graph.add_edge(src, dst, out_port=int(src_port), peer_port=int(dst_port), **attrs)
    return graph


def choose_path(graph, source, destination, congestion_aware=False,
                utilization_weight=4.0, congestion_threshold=0.80):
    """Choose a loop-free path, excluding failed links already absent in graph.

    Congestion costs are used only if every candidate edge has a fresh measured
    utilization value. Unknown metrics fall back to unweighted shortest path.
    """
    source, destination = int(source), int(destination)
    if source == destination:
        return [source]
    if source not in graph or destination not in graph:
        return None
    if not nx.has_path(graph, source, destination):
        return None
    weight_key = None
    if congestion_aware:
        all_known = all("utilization" in data and data["utilization"] is not None
                        for _, _, data in graph.edges(data=True))
        if all_known:
            weight_key = "route_cost"
            for _, _, data in graph.edges(data=True):
                utilization = max(0.0, float(data["utilization"]))
                # Keep every available link traversable; discourage saturated links.
                data[weight_key] = 1.0 + utilization_weight * max(
                    0.0, utilization - congestion_threshold
                )
    return nx.shortest_path(graph, source, destination, weight=weight_key)


def path_outputs(graph, path):
    """Return (switch, output-port) hops; raises if topology mapping is stale."""
    if not path or len(path) == 1:
        return []
    return [(int(u), int(graph[u][v]["out_port"])) for u, v in zip(path, path[1:])]


def simulator_checkpoint_compatibility():
    """Explain why current graph DQNs cannot be safely invoked from live state."""
    return {
        "compatible": False,
        "reason": (
            "DQN actions are global simulator interventions (maintain/reroute/"
            "rate-limit or no-op/reroute/isolate), not next-hop choices. The live "
            "network does not expose simulator node-load, delay, packet-loss, or "
            "offered-demand values with matching semantics. Missing values are not "
            "filled with fabricated defaults."
        ),
    }
