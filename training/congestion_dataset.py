import networkx as nx
from pathlib import Path

try:
    from .topology_loader import TopologyLoader
    from .dataset_parser import DatasetParser
    from .routing_loader import RoutingLoader
    from .dataset_schema import candidate_features, link_usage_by_edge
except ImportError:  # direct script execution from the training directory
    from topology_loader import TopologyLoader
    from dataset_parser import DatasetParser
    from routing_loader import RoutingLoader
    from dataset_schema import candidate_features, link_usage_by_edge


class CongestionDatasetBuilder:

    def __init__(self, dataset_root="scalability"):
        self.dataset_root = Path(dataset_root)

        self.topology_loader = TopologyLoader(dataset_root)
        self.parser = DatasetParser(dataset_root)
        self.routing_loader = RoutingLoader(dataset_root)

    def build_from_archive(
        self,
        split,
        size,
        archive_name,
        simulation_index=0,
        verbose=True,
    ):

        # =========================================================
        # LOAD SIMULATION
        # =========================================================

        data = self.parser.load_simulation(
            split=split,
            size=size,
            archive_name=archive_name,
            simulation_index=simulation_index,
            verbose=verbose,
        )

        simulation = data["simulation"]

        if data["stability"]["status"] != "OK":
            raise ValueError(
                f"Simulation {simulation['simulation_id']} is not stable: "
                f"{data['stability']['status']}"
            )

        graph_file = simulation["graph_file"]
        routing_file = simulation["routing_file"]

        if verbose:
            print("=" * 70)
            print("BUILDING CONGESTION DATASET")
            print("=" * 70)
            print("Archive :", archive_name)
            print("Simulation:", simulation_index)
            print("Graph   :", graph_file)
            print("Routing :", routing_file)

        # =========================================================
        # LOAD TOPOLOGY
        # =========================================================

        graph = self.topology_loader.load(
            split=split,
            size=size,
            graph_name=graph_file
        )

        # =========================================================
        # PRECOMPUTE SHORTEST-PATH HOP DISTANCES
        # =========================================================

        hop_distances = dict(
            nx.all_pairs_shortest_path_length(graph)
        )

        # =========================================================
        # LOAD ROUTING
        # =========================================================

        routes = self.routing_loader.load(
            split=split,
            size=size,
            routing_name=routing_file
        )

        # =========================================================
        # BUILD LINK STATE
        # =========================================================

        link_state = link_usage_by_edge(graph, data["link_usage"])

        # =========================================================
        # BUILD CANDIDATE-BASED TRAINING SAMPLES
        # =========================================================

        samples = []

        for (source, destination), path in routes.items():

            if path is None or len(path) < 2:
                continue

            # -----------------------------------------------------
            # Every hop in the observed route becomes one
            # next-hop decision.
            # -----------------------------------------------------

            for i in range(len(path) - 1):

                current = path[i]
                next_hop = path[i + 1]

                # -------------------------------------------------
                # Find every possible outgoing link from current
                # node.
                # -------------------------------------------------

                candidates = []

                for neighbor in graph.successors(current):

                    edge = (current, neighbor)

                    if edge not in link_state:
                        continue

                    state = link_state[edge]
                    # -------------------------------------------------
                    # Distance from candidate neighbor to destination.
                    #
                    # This is measured in number of hops using the
                    # topology graph.
                    # -------------------------------------------------

                    if destination in hop_distances.get(
                        neighbor,
                        {}
                    ):
                        hop_distance = hop_distances[
                            neighbor
                        ][destination]

                    else:
                        # Candidate cannot reach destination.
                        hop_distance = float("inf")

                    # -------------------------------------------------
                    # Feature vector
                    #
                    # 0  bandwidth
                    # 1  link weight
                    # 2  link utilization
                    # 3  link losses
                    # 4  average packet size
                    # 5  queue utilization
                    # 6  queue losses
                    # 7  average port occupancy
                    # 8  maximum queue occupancy
                    # 9  queue average packet size
                    # 10 hop distance to destination
                    # -------------------------------------------------

                    features = candidate_features(state, hop_distance)

                    candidates.append({
                        "neighbor": neighbor,
                        "features": features
                    })

                # -------------------------------------------------
                # Make sure the observed next hop is one of the
                # available candidates.
                # -------------------------------------------------

                candidate_neighbors = [
                    candidate["neighbor"]
                    for candidate in candidates
                ]

                if next_hop not in candidate_neighbors:
                    continue

                # -------------------------------------------------
                # Target is the index of the observed next hop.
                # -------------------------------------------------

                target_index = candidate_neighbors.index(
                    next_hop
                )

                samples.append({
                    "source": source,
                    "current": current,
                    "destination": destination,

                    "candidates": candidates,

                    "target_index": target_index,
                    "target_next_hop": next_hop
                })

        # =========================================================
        # SUMMARY
        # =========================================================

        if verbose:
            print()
            print("Nodes:", graph.number_of_nodes())
            print("Edges:", graph.number_of_edges())
            print("Routes:", len(routes))
            print("Training decisions:", len(samples))

        # =========================================================
        # DISPLAY FIRST 5 SAMPLES
        # =========================================================

        if verbose:
            print()
            print("First 5 candidate-based samples:")

        for i, sample in enumerate(samples[:5] if verbose else []):

            print()
            print(f"Sample {i}")

            print(
                "  Source      :",
                sample["source"]
            )

            print(
                "  Current     :",
                sample["current"]
            )

            print(
                "  Destination :",
                sample["destination"]
            )

            print(
                "  Target      :",
                sample["target_next_hop"]
            )

            print(
                "  Target index:",
                sample["target_index"]
            )

            print("  Candidates:")

            for j, candidate in enumerate(
                sample["candidates"]
            ):

                if j == sample["target_index"]:
                    marker = " <-- TARGET"
                else:
                    marker = ""

                print(
                    f"    [{j}] "
                    f"{sample['current']} -> "
                    f"{candidate['neighbor']} "
                    f"{candidate['features']}"
                    f"{marker}"
                )

        return samples


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    builder = CongestionDatasetBuilder()

    samples = builder.build_from_archive(
        split="train",
        size=45,
        archive_name="results_45_400-2000_0_24.tar.gz",
        simulation_index=0
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        "Total training decisions:",
        len(samples)
    )
