from pathlib import Path
import tarfile


DATASET = Path("scalability")

ARCHIVE = (
    DATASET
    / "train"
    / "45"
    / "results_45_400-2000_0_24.tar.gz"
)


def inspect_file(tar, filename):

    member = None

    for m in tar.getmembers():
        if m.name.endswith(filename):
            member = m
            break

    if member is None:
        print(f"{filename}: NOT FOUND")
        return

    f = tar.extractfile(member)

    if f is None:
        return

    lines = f.read().decode("utf-8").splitlines()

    print()
    print("=" * 70)
    print(filename)
    print("=" * 70)

    print("Number of simulations:", len(lines))

    if lines:

        first = lines[0]

        print("First line length:", len(first))

        fields = first.split(";")

        print("Number of fields:", len(fields))

        print("\nFirst 10 fields:")

        for i, field in enumerate(fields[:10]):
            print(f"[{i}] {field}")


def main():

    print("=" * 70)
    print("DATASET RESULT INSPECTION")
    print("=" * 70)

    print("Archive:", ARCHIVE)

    with tarfile.open(ARCHIVE, "r:gz") as tar:

        print("\nFiles:")

        for member in tar.getmembers():
            print(" ", member.name)

        inspect_file(tar, "traffic.txt")
        inspect_file(tar, "linkUsage.txt")
        inspect_file(tar, "simulationResults.txt")
        inspect_file(tar, "stability.txt")


if __name__ == "__main__":
    main()