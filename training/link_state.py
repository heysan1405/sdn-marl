from pathlib import Path

from topology_loader import TopologyLoader
from dataset_parser import DatasetParser


class LinkStateLoader:

    def __init__(self, dataset_root="scalability"):
        self.dataset_root = Path(dataset_root)

        self.topology_loader = TopologyLoader(
            dataset_root
        )

        self.parser = DatasetParser(
            dataset_root
        )

    def load(
        self,
        split,
        size,
        graph_file,
        archive_name,
        simulation_index=0
    ):

        # -------------------------------------------------
        # Load topology
        # -------------------------------------------------

        graph = self.topology_loader.load(
            split=split,
            size=size,
            graph_name=graph_file
        )

        # -------------------------------------------------
        # Load simulation
        # -------------------------------------------------

        data = self.parser.load_simulation(
            split=split,
            size=size,
            archive_name=archive_name,
            simulation_index=simulation_index
        )

        usage_fields = data["link_usage"]

        # -------------------------------------------------
        # Extract valid usage entries
        # -------------------------------------------------

        valid_usage = [
            value
            for value in usage_fields
            if value is not None
        ]

        edges = list(graph.edges(data=True))

        if len(edges) != len(valid_usage):

            raise ValueError(
                f"Edge/linkUsage mismatch: "
                f"{len(edges)} edges vs "
                f"{len(valid_usage)} measurements"
            )

        # -------------------------------------------------
        # Map edge → usage
        # -------------------------------------------------

        link_state = {}

        for edge_index, (
            source,
            destination,
            attributes
        ) in enumerate(edges):

            usage = valid_usage[edge_index]

            link_state[(source, destination)] = {

                "source": source,

                "destination": destination,

                "port": attributes.get("port"),

                "bandwidth": attributes.get(
                    "bandwidth"
                ),

                "weight": attributes.get(
                    "weight"
                ),

                "metrics": usage
            }

        return graph, link_state


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    loader = LinkStateLoader()

    graph, link_state = loader.load(
        split="train",
        size=45,
        graph_file="graph-45-1-2.txt",
        archive_name="results_45_400-2000_0_24.tar.gz",
        simulation_index=0
    )

    print("=" * 70)
    print("LINK STATE")
    print("=" * 70)

    print(
        "Nodes:",
        graph.number_of_nodes()
    )

    print(
        "Edges:",
        graph.number_of_edges()
    )

    print(
        "Link states:",
        len(link_state)
    )

    print("\nFirst 10 links:")

    for i, (
        edge,
        state
    ) in enumerate(link_state.items()):

        if i >= 10:
            break

        print()
        print(
            f"{edge[0]} -> {edge[1]}"
        )

        print(
            "Port:",
            state["port"]
        )

        print(
            "Bandwidth:",
            state["bandwidth"]
        )

        print(
            "Metrics:",
            state["metrics"]
        )