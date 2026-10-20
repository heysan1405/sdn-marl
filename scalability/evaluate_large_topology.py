"""Sampled graph-only routing evaluation for CAIDA Matrix Market topology."""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import networkx as nx

try:
    import resource
except ImportError:  # Windows
    resource = None

from large_topology import read_matrix_market, connected_subgraph, route_distance_constrained


def peak_rss_kib():
    try:
        # Linux reports KiB; Windows reports bytes.
        if resource is None:
            raise RuntimeError("resource module unavailable")
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value / 1024) if value > 10**7 else int(value)
    except Exception:
        try:
            import psutil
            return int(psutil.Process().memory_info().peak_wset / 1024)
        except Exception:
            return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="tech-RL-caida/tech-RL-caida.mtx")
    parser.add_argument("--size", type=int, help="Connected induced subgraph node count")
    parser.add_argument("--full-graph", action="store_true", help="Evaluate sampled routes on all CAIDA nodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--output-dir", default="results/large_topology")
    args = parser.parse_args()
    if args.full_graph == (args.size is not None):
        parser.error("specify exactly one of --full-graph or --size N")
    if args.size is not None and args.size < 2:
        parser.error("--size must be at least 2")
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    full, meta = read_matrix_market(args.matrix)
    if args.full_graph:
        graph = full
        selection = {"selection": "full source graph", "seed": args.seed}
        label = "full"
    else:
        if args.size > len(full):
            parser.error(f"requested {args.size} nodes but source graph has {len(full)}")
        graph, selection = connected_subgraph(full, args.size, args.seed)
        label = str(args.size)
    if not nx.is_connected(graph):
        raise RuntimeError("Evaluation topology is unexpectedly disconnected")

    nodes = list(graph)
    rng = random.Random(args.seed)
    pairs = set()
    pair_limit = min(args.pairs, len(nodes) * (len(nodes) - 1))
    while len(pairs) < pair_limit:
        pairs.add(tuple(rng.sample(nodes, 2)))

    # A destination BFS is computed once and reused for both policies. Only
    # sampled destinations are visited; no all-pairs table is constructed.
    counts = {name: {"delivered": 0, "dead_ends": 0, "loops": 0, "hops": []}
              for name in ("shortest_path_reference", "distance_constrained")}
    for source, destination in pairs:
        distances = nx.single_source_shortest_path_length(graph, destination)
        hops = distances.get(source)
        if hops is None:
            status, hops = "dead_end", 0
        else:
            status = "delivered"
        record = counts["shortest_path_reference"]
        record["delivered"] += status == "delivered"
        record["dead_ends"] += status == "dead_end"
        if status == "delivered":
            record["hops"].append(hops)
        constrained = route_distance_constrained(graph, source, destination, distances)
        record = counts["distance_constrained"]
        record["delivered"] += constrained["status"] == "delivered"
        record["dead_ends"] += constrained["status"] == "dead_end"
        record["loops"] += constrained["status"] == "loop"
        if constrained["status"] == "delivered":
            record["hops"].append(constrained["hops"])

    policies = []
    for name, values in counts.items():
        hops = sorted(values.pop("hops"))
        delivered = values["delivered"]
        policies.append({"policy": name, "pairs": len(pairs), **values,
                         "unreachable_pairs": values["dead_ends"],
                         "delivery_rate": delivered / len(pairs),
                         "mean_hops": sum(hops) / len(hops) if hops else None,
                         "p50_hops": hops[len(hops)//2] if hops else None,
                         "p95_hops": hops[int(.95*(len(hops)-1))] if hops else None})

    degree = [d for _, d in graph.degree()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    topology_path = None
    if not args.full_graph:
        topology_path = out / f"caida_{label}_seed{args.seed}.json"
        selection.update({"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                          "connected": True, "degree_min": min(degree),
                          "degree_mean": sum(degree)/len(degree), "degree_max": max(degree),
                          "source_graph": meta, "edges": [list(e) for e in sorted(graph.edges)]})
        topology_path.write_text(json.dumps(selection, separators=(",", ":")), encoding="utf-8")
    report = {"experiment": "graph-only; topology has no traffic or link-state measurements",
              "seed": args.seed, "topology_file": str(topology_path) if topology_path else None,
              "source_graph": meta, "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
              "connected": True, "degree": {"min": min(degree), "mean": sum(degree)/len(degree), "max": max(degree)},
              "source_destination_pairs": len(pairs), "runtime_seconds": time.perf_counter()-started,
              "max_rss_kib": peak_rss_kib(), "policies": policies,
              "unsupported_learned_policies": {"baseline_model": "requires measured link features absent from CAIDA",
                "congestion_dqn": "requires simulator graph state and synthetic demands; not a topology-only policy",
                "failure_dqn": "requires failure observations; no failures are encoded in CAIDA",
                "orchestrated": "requires compatible agent observations"},
              "congestion_metrics": None}
    report_path = out / f"evaluation_{label}_seed{args.seed}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
