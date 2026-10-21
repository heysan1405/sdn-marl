import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topology_loader import TopologyLoader
from dataset_parser import DatasetParser
from routing_loader import RoutingLoader
from dataset_schema import candidate_features, link_usage_by_edge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CongestionAgent(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def normalize_features(x):
    x = torch.nan_to_num(
        x, nan=0.0, posinf=1e6, neginf=-1e6
    )
    x[:, 0] /= 100000.0
    x[:, 4] /= 2000.0
    x[:, 9] /= 2000.0
    x[:, 10] /= 50.0
    return x


def load_model(path, device):
    model = CongestionAgent().to(device)
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {path}")
    return model


def link_congestion(metrics):
    # Same congestion definition used by the congestion-aware training.
    utilization = np.clip(float(metrics[0]), 0.0, 1.0)
    occupancy = np.clip(float(metrics[5]) / 32.0, 0.0, 1.0)
    max_queue = np.clip(float(metrics[6]) / 32.0, 0.0, 1.0)
    return float((utilization + occupancy + max_queue) / 3.0)


def load_context(split, size, archive_name, simulation_index):
    root = "scalability"
    parser = DatasetParser(root)
    topology_loader = TopologyLoader(root)
    routing_loader = RoutingLoader(root)

    data = parser.load_simulation(
        split=split,
        size=size,
        archive_name=archive_name,
        simulation_index=simulation_index,
    )
    simulation = data["simulation"]

    graph = topology_loader.load(
        split=split,
        size=size,
        graph_name=simulation["graph_file"],
    )
    routes = routing_loader.load(
        split=split,
        size=size,
        routing_name=simulation["routing_file"],
    )

    distances = dict(nx.all_pairs_shortest_path_length(graph))

    if data["stability"]["status"] != "OK":
        raise ValueError(
            f"Simulation {simulation['simulation_id']} is not stable: "
            f"{data['stability']['status']}"
        )
    link_state = link_usage_by_edge(graph, data["link_usage"])

    print(f"Graph: {simulation['graph_file']}")
    print(f"Routing: {simulation['routing_file']}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Reference routes: {len(routes)}")

    return graph, routes, distances, link_state


def make_candidates(graph, distances, link_state, current, destination, enforce_progress=False):
    candidates = []
    current_distance = distances.get(current, {}).get(destination, float('inf'))

    for neighbor in graph.successors(current):
        state = link_state.get((current, neighbor))
        if state is None:
            continue

        hop_distance = distances.get(neighbor, {}).get(
            destination, float("inf")
        )
        if enforce_progress and not (hop_distance < current_distance):
            continue

        metrics = state["metrics"]
        features = candidate_features(state, hop_distance)

        candidates.append({
            "neighbor": neighbor,
            "features": features,
            "congestion": link_congestion(metrics),
        })

    # Avoid choosing a node from which the destination is unreachable.
    reachable = [
        c for c in candidates
        if np.isfinite(c["features"][10])
    ]
    return reachable


@torch.inference_mode()
def choose_next_hop(model, candidates, device):
    if not candidates:
        return None

    x = torch.tensor(
        [c["features"] for c in candidates],
        dtype=torch.float32,
        device=device,
    )
    x = normalize_features(x)
    scores = model(x)
    index = int(torch.argmax(scores).item())
    return candidates[index]


def rollout(model, graph, distances, link_state, source, destination, enforce_progress=False):
    if source == destination:
        return {"status": "delivered", "hops": 0, "congestion": []}

    current = source
    visited = {current}
    congestion_values = []
    max_hops = max(2, graph.number_of_nodes() * 2)

    for _ in range(max_hops):
        candidates = make_candidates(
            graph, distances, link_state, current, destination,
            enforce_progress=enforce_progress
        )
        selected = choose_next_hop(model, candidates, DEVICE)

        if selected is None:
            return {
                "status": "dead_end",
                "hops": len(congestion_values),
                "congestion": congestion_values,
            }

        next_node = selected["neighbor"]
        congestion_values.append(selected["congestion"])

        if next_node == destination:
            return {
                "status": "delivered",
                "hops": len(congestion_values),
                "congestion": congestion_values,
            }

        if next_node in visited:
            return {
                "status": "loop",
                "hops": len(congestion_values),
                "congestion": congestion_values,
            }

        if not graph.has_edge(current, next_node):
            return {
                "status": "dead_end",
                "hops": len(congestion_values),
                "congestion": congestion_values,
            }

        current = next_node
        visited.add(current)

    return {
        "status": "hop_limit",
        "hops": len(congestion_values),
        "congestion": congestion_values,
    }


def reference_congestion(path, link_state):
    values = []
    for u, v in zip(path[:-1], path[1:]):
        state = link_state.get((u, v))
        if state is not None:
            values.append(link_congestion(state["metrics"]))
    return values


def evaluate_model(name, model, graph, routes, distances, link_state, enforce_progress=False):
    delivered = 0
    loops = 0
    dead_ends = 0
    hop_limits = 0
    delivered_hops = []
    all_hops = []
    path_congestions = []
    completed = 0
    unreachable_reference = 0

    for (source, destination), reference_path in routes.items():
        if reference_path is None or len(reference_path) < 2:
            continue

        if source not in graph or destination not in graph:
            continue

        if destination not in distances.get(source, {}):
            unreachable_reference += 1
            continue

        result = rollout(
            model, graph, distances, link_state, source, destination,
            enforce_progress=enforce_progress
        )

        all_hops.append(result["hops"])

        if result["status"] == "delivered":
            delivered += 1
            completed += 1
            delivered_hops.append(result["hops"])

        elif result["status"] == "loop":
            loops += 1
        elif result["status"] == "dead_end":
            dead_ends += 1
        else:
            hop_limits += 1

        if result["status"] == "delivered" and result["congestion"]:
            path_congestions.append(float(np.mean(result["congestion"])))

    total = delivered + loops + dead_ends + hop_limits
    delivery_rate = 100.0 * delivered / total if total else 0.0

    print(f"\n--- {name} ---")
    print(f"Evaluated routes:       {total}")
    print(f"Delivered:              {delivered} ({delivery_rate:.2f}%)")
    print(f"Loops:                  {loops}")
    print(f"Dead ends:              {dead_ends}")
    print(f"Hop-limit terminations: {hop_limits}")
    print(f"Unreachable references skipped: {unreachable_reference}")

    if delivered_hops:
        print(f"Mean hops (delivered):  {np.mean(delivered_hops):.3f}")
    else:
        print("Mean hops (delivered):  N/A")

    if path_congestions:
        print(
            "Mean recorded path congestion (delivered): "
            f"{np.mean(path_congestions):.5f}"
        )
    else:
        print("Mean recorded path congestion (delivered): N/A")

    return {
        "name": name,
        "total": total,
        "delivered": delivered,
        "delivery_rate": delivery_rate,
        "loops": loops,
        "dead_ends": dead_ends,
        "hop_limits": hop_limits,
        "mean_hops": float(np.mean(delivered_hops)) if delivered_hops else float("nan"),
        "mean_congestion": float(np.mean(path_congestions)) if path_congestions else float("nan"),
    }


def main():
    global DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--simulation-index", type=int, default=0)
    parser.add_argument(
        "--baseline", default="models/congestion_agent.pt"
    )
    parser.add_argument(
        "--aware", default="models/congestion_agent_aware.pt"
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    baseline = load_model(args.baseline, DEVICE)
    aware = load_model(args.aware, DEVICE)

    graph, routes, distances, link_state = load_context(
        split="test",
        size=args.size,
        archive_name=args.archive,
        simulation_index=args.simulation_index,
    )

    results = []
    results.append(evaluate_model(
        "Baseline", baseline, graph, routes, distances, link_state
    ))
    results.append(evaluate_model(
        "Congestion-aware", aware, graph, routes, distances, link_state
    ))
    results.append(evaluate_model(
        "Congestion-aware + distance constraint",
        aware, graph, routes, distances, link_state,
        enforce_progress=True
    ))

    print("\n=== COMPARISON ===")
    print(
        f"{'Metric':<30} {'Baseline':>14} {'Congestion-aware':>18}"
    )
    print("-" * 64)

    for label, key, fmt in [
        ("Delivery rate (%)", "delivery_rate", ".2f"),
        ("Looped routes", "loops", "d"),
        ("Dead-end routes", "dead_ends", "d"),
        ("Mean hops (delivered)", "mean_hops", ".3f"),
        ("Mean recorded congestion", "mean_congestion", ".5f"),
    ]:
        a, b, c = results
        av, bv, cv = a[key], b[key], c[key]
        print(
            f"{label:<30} {format(av, fmt):>14} "
            f"{format(bv, fmt):>18} {format(cv, fmt):>22}"
        )


if __name__ == "__main__":
    main()
