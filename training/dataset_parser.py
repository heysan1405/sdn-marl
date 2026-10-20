from pathlib import Path
import tarfile


class DatasetParser:

    def __init__(self, dataset_root="scalability"):
        self.dataset_root = Path(dataset_root)

    # =========================================================
    # Find archive
    # =========================================================

    def get_archive(self, split, size, archive_name):

        path = (
            self.dataset_root
            / split
            / str(size)
            / archive_name
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Archive not found: {path}"
            )

        return path

    # =========================================================
    # Read file from tar.gz
    # =========================================================

    @staticmethod
    def read_file(tar, filename):

        member = None

        for m in tar.getmembers():

            if m.name.endswith(filename):
                member = m
                break

        if member is None:
            raise FileNotFoundError(
                f"{filename} not found inside archive"
            )

        file_obj = tar.extractfile(member)

        if file_obj is None:
            raise RuntimeError(
                f"Could not extract {filename}"
            )

        return file_obj.read().decode("utf-8").splitlines()

    # =========================================================
    # Parse input_files.txt
    # =========================================================

    @staticmethod
    def parse_input_files(lines):

        simulations = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split(";")

            if len(parts) < 3:
                continue

            simulation_id = int(parts[0])

            graph_file = parts[1]

            routing_file = parts[2]

            simulations.append({
                "simulation_id": simulation_id,
                "graph_file": graph_file,
                "routing_file": routing_file
            })

        return simulations

    # =========================================================
    # Parse traffic
    # =========================================================

    @staticmethod
    def parse_traffic_field(field):

        field = field.strip()

        if not field:
            return None

        # Global traffic field
        if "|" in field:

            parts = field.split("|")

            global_value = float(parts[0])

            extra = parts[1] if len(parts) > 1 else ""

            return {
                "global_value": global_value,
                "extra": extra
            }

        values = field.split(",")

        parsed = []

        for value in values:

            value = value.strip()

            if value == "":
                parsed.append(None)
            else:
                parsed.append(float(value))

        return parsed

    # =========================================================
    # Parse link usage
    # =========================================================

    @staticmethod
    def parse_link_usage_field(field):

        field = field.strip()

        # -1 means no measurement
        if field == "-1":
            return None

        # Empty field
        if field == "":
            return None

        values = field.split(",")

        parsed = []

        for value in values:

            value = value.strip()

            if value == "":
                parsed.append(None)
            else:
                parsed.append(float(value))

        return parsed

    # =========================================================
    # Parse simulation results
    # =========================================================

    @staticmethod
    def parse_result_field(field):

        field = field.strip()

        if not field:
            return None

        parts = field.split("|")

        # Main values
        main_values = []

        for value in parts[0].split(","):

            value = value.strip()

            if value == "":
                main_values.append(None)
            else:
                main_values.append(float(value))

        # No second section
        if len(parts) == 1:
            return main_values

        # Second section
        extra_values = []

        for value in parts[1].split(","):

            value = value.strip()

            if value == "":
                extra_values.append(None)
            else:
                extra_values.append(float(value))

        return {
            "global": main_values,
            "extra": extra_values
        }

    # =========================================================
    # Parse routing file
    # =========================================================

    @staticmethod
    def parse_routing_file(lines):

        routes = {}

        for line in lines:

            line = line.strip()

            if not line:
                continue

            nodes = [
                int(x)
                for x in line.split(";")
            ]

            if len(nodes) < 2:
                continue

            source = nodes[0]

            destination = nodes[-1]

            routes[(source, destination)] = nodes

        return routes

    # =========================================================
    # Load one simulation
    # =========================================================

    def load_simulation(
        self,
        split,
        size,
        archive_name,
        simulation_index=0
    ):

        archive_path = self.get_archive(
            split,
            size,
            archive_name
        )

        print("=" * 70)
        print("LOADING SIMULATION")
        print("=" * 70)

        print("Archive:", archive_path)

        with tarfile.open(
            archive_path,
            "r:gz"
        ) as tar:

            # -------------------------------------------------
            # Read raw files
            # -------------------------------------------------

            input_lines = self.read_file(
                tar,
                "input_files.txt"
            )

            traffic_lines = self.read_file(
                tar,
                "traffic.txt"
            )

            link_lines = self.read_file(
                tar,
                "linkUsage.txt"
            )

            result_lines = self.read_file(
                tar,
                "simulationResults.txt"
            )

        # -----------------------------------------------------
        # Check simulation index
        # -----------------------------------------------------

        if simulation_index < 0:

            raise ValueError(
                "simulation_index cannot be negative"
            )

        if simulation_index >= len(input_lines):

            raise IndexError(
                f"Simulation index {simulation_index} "
                f"does not exist. "
                f"Available simulations: "
                f"0-{len(input_lines) - 1}"
            )

        # -----------------------------------------------------
        # Simulation metadata
        # -----------------------------------------------------

        simulations = self.parse_input_files(
            input_lines
        )

        simulation = simulations[simulation_index]

        # -----------------------------------------------------
        # Get corresponding simulation lines
        # -----------------------------------------------------

        traffic_fields = traffic_lines[
            simulation_index
        ].split(";")

        link_fields = link_lines[
            simulation_index
        ].split(";")

        result_fields = result_lines[
            simulation_index
        ].split(";")

        # -----------------------------------------------------
        # Check field counts
        # -----------------------------------------------------

        print("\nField counts:")

        print(
            "Traffic:",
            len(traffic_fields)
        )

        print(
            "Link usage:",
            len(link_fields)
        )

        print(
            "Results:",
            len(result_fields)
        )

        if not (
            len(traffic_fields)
            == len(link_fields)
            == len(result_fields)
        ):

            raise ValueError(
                "Traffic, linkUsage and "
                "simulationResults have "
                "different numbers of fields"
            )

        # -----------------------------------------------------
        # Parse
        # -----------------------------------------------------

        traffic = [
            self.parse_traffic_field(field)
            for field in traffic_fields
        ]

        link_usage = [
            self.parse_link_usage_field(field)
            for field in link_fields
        ]

        results = [
            self.parse_result_field(field)
            for field in result_fields
        ]

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {
            "simulation": simulation,
            "traffic": traffic,
            "link_usage": link_usage,
            "results": results
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    parser = DatasetParser()

    data = parser.load_simulation(
        split="train",
        size=45,
        archive_name="results_45_400-2000_0_24.tar.gz",
        simulation_index=0
    )

    print("\n" + "=" * 70)
    print("SIMULATION DATA")
    print("=" * 70)

    print("\nSimulation:")
    print(data["simulation"])

    print(
        "\nNumber of traffic fields:",
        len(data["traffic"])
    )

    print(
        "Number of link usage fields:",
        len(data["link_usage"])
    )

    print(
        "Number of result fields:",
        len(data["results"])
    )

    print("\nGlobal traffic:")
    print(data["traffic"][0])

    print("\nFirst traffic entry:")
    print(data["traffic"][1])

    print("\nFirst link usage entry:")
    print(data["link_usage"][1])

    print("\nGlobal result:")
    print(data["results"][0])

    print("\nFirst result entry:")
    print(data["results"][1])