from dataset_parser import DatasetParser


parser = DatasetParser()

data = parser.load_simulation(
    split="train",
    size=45,
    archive_name="results_45_400-2000_0_24.tar.gz",
    simulation_index=0
)

link_usage = data["link_usage"]

print("=" * 70)
print("LINK USAGE ANALYSIS")
print("=" * 70)

print("\nTotal fields:", len(link_usage))

valid = []

for i, value in enumerate(link_usage):

    if value is not None:
        valid.append((i, value))


print("Valid fields:", len(valid))
print("None fields:", len(link_usage) - len(valid))

print("\nFirst 20 valid fields:")

for index, value in valid[:20]:

    print(
        f"[{index}] "
        f"length={len(value)} "
        f"value={value}"
    )


print("\nField index analysis:")

for index, value in valid[:20]:

    if index == 0:
        print(
            f"[{index}] GLOBAL"
        )
        continue

    # Convert field index to source/destination
    pair_index = index - 1

    source = pair_index // 44

    destination_offset = pair_index % 44

    destination = (
        destination_offset
        if destination_offset < source
        else destination_offset + 1
    )

    print(
        f"[{index}] "
        f"{source} -> {destination} "
        f"{value}"
    )