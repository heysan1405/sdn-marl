from pathlib import Path
import networkx as nx


class TopologyLoader:

    def __init__(self, dataset_root="scalability"):
        self.dataset_root = Path(dataset_root)

    def load(self, split, size, graph_name):
        """
        Load one topology.

        Example:
            split = "train"
            size = 45
            graph_name = "graph-45-0-2.txt"
        """

        path = (
            self.dataset_root
            / split
            / str(size)
            / "graphs"
            / graph_name
        )

        if not path.exists():
            raise FileNotFoundError(path)

        graph = nx.read_gml(path, label=None)

        return graph

    @staticmethod
    def summarize(graph):

        print("=" * 60)
        print("TOPOLOGY")
        print("=" * 60)

        print("Nodes:", graph.number_of_nodes())
        print("Edges:", graph.number_of_edges())
        print("Directed:", graph.is_directed())

        print("\nNodes:")

        for node, attrs in list(graph.nodes(data=True))[:5]:
            print(
                node,
                {
                    "queue": attrs.get("queueSizes"),
                    "scheduler": attrs.get("schedulingPolicy"),
                    "QoS levels": attrs.get("levelsQoS"),
                }
            )

        print("\nEdges:")

        for u, v, attrs in list(graph.edges(data=True))[:10]:

            print(
                f"{u} -> {v}",
                {
                    "port": attrs.get("port"),
                    "bandwidth": attrs.get("bandwidth"),
                    "weight": attrs.get("weight"),
                }
            )


if __name__ == "__main__":

    loader = TopologyLoader()

    graph = loader.load(
        split="train",
        size=45,
        graph_name="graph-45-0-2.txt"
    )

    loader.summarize(graph)