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


link_usage = data["link_usage"]


# ============================================================
# Get valid linkUsage entries
# ============================================================

valid = [
    (i, value)
    for i, value in enumerate(link_usage)
    if value is not None
]


# ============================================================
# Print topology edges and linkUsage together
# ============================================================

print("=" * 80)
print("TOPOLOGY EDGE ORDER")
print("=" * 80)

edges = list(graph.edges(data=True))

for i, (u, v, attrs) in enumerate(edges):

    print(
        f"EDGE {i:3d}: "
        f"{u:2d} -> {v:2d} "
        f"port={attrs.get('port')} "
        f"bw={attrs.get('bandwidth')}"
    )


print("\n" + "=" * 80)
print("LINK USAGE ORDER")
print("=" * 80)

for i, (field_index, value) in enumerate(valid):

    print(
        f"USAGE {i:3d}: "
        f"field_index={field_index:4d} "
        f"value={value}"
    )

    if i >= 30:
        break


# ============================================================
# IMPORTANT TEST
# ============================================================

print("\n" + "=" * 80)
print("EDGE ↔ FIELD INDEX HYPOTHESIS")
print("=" * 80)

print(
    "\nIf linkUsage follows graph edge order, "
    "then the following should correspond:\n"
)

for edge_index, (u, v, attrs) in enumerate(edges[:30]):

    field_index, usage = valid[edge_index]

    print(
        f"EDGE {edge_index:3d}: "
        f"{u:2d}->{v:2d} "
        f"<--> "
        f"FIELD {field_index:4d}"
    )