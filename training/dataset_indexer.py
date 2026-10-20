from pathlib import Path

from dataset_parser import DatasetParser


class DatasetIndexer:

    def __init__(self, dataset_root="scalability"):
        self.dataset_root = Path(dataset_root)
        self.parser = DatasetParser(dataset_root)

    # =========================================================
    # Load routing file
    # =========================================================

    def load_routing(
        self,
        split,
        size,
        routing_file
    ):

        path = (
            self.dataset_root
            / split
            / str(size)
            / "routings"
            / routing_file
        )

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r") as f:
            lines = f.readlines()

        return self.parser.parse_routing_file(lines)

    # =========================================================
    # Build source-destination records
    # =========================================================

    def build_records(
        self,
        split,
        size,
        archive_name,
        simulation_index=0
    ):

        # -----------------------------------------------------
        # Load simulation
        # -----------------------------------------------------

        data = self.parser.load_simulation(
            split=split,
            size=size,
            archive_name=archive_name,
            simulation_index=simulation_index
        )

        simulation = data["simulation"]

        # -----------------------------------------------------
        # Load routing
        # -----------------------------------------------------

        routes = self.load_routing(
            split,
            size,
            simulation["routing_file"]
        )

        # -----------------------------------------------------
        # Check number of routes
        # -----------------------------------------------------

        expected_routes = size * (size - 1)

        if len(routes) != expected_routes:

            print(
                f"WARNING: expected "
                f"{expected_routes} routes, "
                f"found {len(routes)}"
            )

        # -----------------------------------------------------
        # Build records
        # -----------------------------------------------------

        records = []

        # Field 0 is global.
        # Fields 1.. are source-destination pairs.

        field_index = 1

        for source in range(size):

            for destination in range(size):

                if source == destination:
                    continue

                # ---------------------------------------------
                # Route
                # ---------------------------------------------

                path = routes.get(
                    (source, destination)
                )

                if path is None:

                    print(
                        f"WARNING: missing route "
                        f"{source} -> {destination}"
                    )

                    field_index += 1
                    continue

                # ---------------------------------------------
                # Corresponding dataset fields
                # ---------------------------------------------

                traffic = data["traffic"][
                    field_index
                ]

                link_usage = data["link_usage"][
                    field_index
                ]

                results = data["results"][
                    field_index
                ]

                # ---------------------------------------------
                # Create record
                # ---------------------------------------------

                record = {

                    "source": source,

                    "destination": destination,

                    "path": path,

                    "traffic": traffic,

                    "link_usage": link_usage,

                    "results": results
                }

                records.append(record)

                field_index += 1

        return {
            "simulation": simulation,
            "records": records
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    indexer = DatasetIndexer()

    dataset = indexer.build_records(
        split="train",
        size=45,
        archive_name="results_45_400-2000_0_24.tar.gz",
        simulation_index=0
    )

    print()
    print("=" * 70)
    print("DATASET INDEX")
    print("=" * 70)

    print(
        "\nSimulation:",
        dataset["simulation"]
    )

    print(
        "\nNumber of records:",
        len(dataset["records"])
    )

    print("\nFirst 5 records:")

    for record in dataset["records"][:5]:

        print()
        print(
            f"{record['source']} "
            f"-> "
            f"{record['destination']}"
        )

        print(
            "Path:",
            record["path"]
        )

        print(
            "Traffic:",
            record["traffic"]
        )

        print(
            "Link usage:",
            record["link_usage"]
        )

        print(
            "Results:",
            record["results"]
        )