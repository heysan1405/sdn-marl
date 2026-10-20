from pathlib import Path


class RoutingLoader:

    def __init__(self, dataset_root="scalability"):
        self.dataset_root = Path(dataset_root)

    def load(self, split, size, routing_name):
        path = (
            self.dataset_root
            / split
            / str(size)
            / "routings"
            / routing_name
        )

        if not path.exists():
            raise FileNotFoundError(path)

        routes = {}

        with open(path, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                nodes = [int(x) for x in line.split(";")]

                source = nodes[0]
                destination = nodes[-1]

                routes[(source, destination)] = nodes

        return routes

    @staticmethod
    def summarize(routes):
        print("=" * 60)
        print("ROUTING")
        print("=" * 60)

        print("Number of source-destination pairs:", len(routes))

        print("\nFirst 10 routes:")

        for i, ((src, dst), path) in enumerate(routes.items()):
            if i >= 10:
                break

            print(f"{src} -> {dst} :", path)


if __name__ == "__main__":

    loader = RoutingLoader()

    routes = loader.load(
        split="train",
        size=45,
        routing_name="Routing-45-0-SP-k-34.txt"
    )

    loader.summarize(routes)