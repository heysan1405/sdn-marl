from topology_loader import TopologyLoader
from dataset_parser import DatasetParser


# ============================================================
# Load topology
# ============================================================

topology_loader = TopologyLoader()

graph = topology_loader.load(
    split="train",
    size=45,
    graph_name="graph-45-1-2.txt"
)


# ============================================================
# Load simulation
# ============================================================

parser = DatasetParser()

data = parser.load_simulation(
    split="train",
    size=45,
    archive_name="results_45_400-2000_0_24.tar.gz",
    simulation_index=0
)


# ============================================================
# Basic topology information
# ============================================================

print("=" * 70)
print("ACTUAL SIMULATION TOPOLOGY")
print("=" * 70)

print("Nodes :", graph.number_of_nodes())
print("Edges :", graph.number_of_edges())
print("Directed :", graph.is_directed())

print("\nFirst 25 topology edges:")

for i, (u, v, attrs) in enumerate(graph.edges(data=True)):

    if i >= 25:
        break

    print(
        f"{i}: {u} -> {v} | attrs={dict(attrs)}"
    )


# ============================================================
# Link usage
# ============================================================

link_usage = data["link_usage"]

valid_indices = [
    i
    for i, value in enumerate(link_usage)
    if value is not None
]


print("\n" + "=" * 70)
print("LINK USAGE")
print("=" * 70)

print("Total fields :", len(link_usage))
print("Valid fields :", len(valid_indices))

print("\nFirst 25 valid field indices:")

print(valid_indices[:25])


# ============================================================
# Test our OLD source-destination interpretation
# ============================================================

print("\n" + "=" * 70)
print("OLD INDEXING CHECK")
print("=" * 70)

matches = 0
mismatches = 0

for index in valid_indices:

    pair_index = index - 1

    source = pair_index // 44

    offset = pair_index % 44

    destination = (
        offset
        if offset < source
        else offset + 1
    )

    if graph.has_edge(source, destination):
        matches += 1
    else:
        mismatches += 1


print("Matches    :", matches)
print("Mismatches :", mismatches)


# ============================================================
# Check whether valid field count equals edge count
# ============================================================

print("\n" + "=" * 70)
print("EDGE COUNT CHECK")
print("=" * 70)

print(
    "Topology edges:",
    graph.number_of_edges()
)

print(
    "Valid linkUsage:",
    len(valid_indices)
)

if graph.number_of_edges() == len(valid_indices):

    print(
        "\nMATCH: number of topology edges "
        "equals number of valid linkUsage entries."
    )

else:

    print(
        "\nNO MATCH: edge count differs."
    )