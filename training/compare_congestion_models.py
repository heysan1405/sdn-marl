
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))

from congestion_dataset import CongestionDatasetBuilder


class CongestionAgent(nn.Module):

    def __init__(self, input_dim=11):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def normalize_features(x):
    x = torch.nan_to_num(
        x, nan=0.0, posinf=1e6, neginf=-1e6
    )

    x[:, 0] /= 100000.0
    x[:, 4] /= 2000.0
    x[:, 9] /= 2000.0
    x[:, 10] /= 50.0

    return x


def calculate_congestion(features):
    """Use raw features: utilization, port occupancy, max queue."""

    utilization = features[:, 2].clamp(0.0, 1.0)
    avg_occupancy = (features[:, 7] / 32.0).clamp(0.0, 1.0)
    max_queue = (features[:, 8] / 32.0).clamp(0.0, 1.0)

    return (utilization + avg_occupancy + max_queue) / 3.0


def load_model(path, device):
    model = CongestionAgent(input_dim=11).to(device)

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def evaluate_model(model, samples, device):
    correct = 0
    total = 0
    selected_congestion_sum = 0.0
    target_congestion_sum = 0.0
    lower_congestion_count = 0

    with torch.no_grad():
        for sample in samples:
            candidates = sample["candidates"]
            target_index = sample["target_index"]

            raw_x = torch.tensor(
                [candidate["features"] for candidate in candidates],
                dtype=torch.float32,
                device=device
            )

            congestion = calculate_congestion(raw_x)
            scores = model(normalize_features(raw_x))
            prediction = torch.argmax(scores).item()

            correct += int(prediction == target_index)
            selected_congestion_sum += congestion[prediction].item()
            target_congestion_sum += congestion[target_index].item()

            if congestion[prediction].item() < congestion[target_index].item():
                lower_congestion_count += 1

            total += 1

    if total == 0:
        return None

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "selected_congestion": selected_congestion_sum / total,
        "target_congestion": target_congestion_sum / total,
        "lower_congestion_percent": 100.0 * lower_congestion_count / total,
    }


def main():
    print("=" * 76)
    print("BASELINE VS CONGESTION-AWARE MODEL")
    print("=" * 76)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_paths = {
        "Baseline": Path("models/congestion_agent.pt"),
        "Congestion-aware": Path("models/congestion_agent_aware.pt"),
    }

    for name, path in model_paths.items():
        if not path.exists():
            print(f"ERROR: {name} model not found: {path}")
            return

    models = {
        name: load_model(path, device)
        for name, path in model_paths.items()
    }

    # Use the same test archive and samples for both models.
    tests = [
        (100, "results_400-2000_0_0.tar.gz", 0),
        (200, "results_400-2000_1_1.tar.gz", 0),
        (300, "results_400-2000_0_0.tar.gz", 0),
    ]

    builder = CongestionDatasetBuilder()

    for size, archive_name, simulation_index in tests:
        print("\n" + "=" * 76)
        print(f"TEST TOPOLOGY: {size} NODES")
        print("=" * 76)
        print("Archive:", archive_name)
        print("Simulation index:", simulation_index)

        try:
            samples = builder.build_from_archive(
                split="test",
                size=size,
                archive_name=archive_name,
                simulation_index=simulation_index
            )
        except Exception as error:
            print(f"Could not load test data: {error}")
            continue

        print("Decisions:", len(samples))

        for name, model in models.items():
            result = evaluate_model(model, samples, device)

            if result is None:
                print(f"{name}: no decisions to evaluate")
                continue

            print(f"\n{name}")
            print("-" * 40)
            print(
                f"Next-hop accuracy: "
                f"{result['accuracy']:.2f}% "
                f"({result['correct']}/{result['total']})"
            )
            print(
                f"Mean selected-link congestion: "
                f"{result['selected_congestion']:.4f}"
            )
            print(
                f"Mean dataset-target-link congestion: "
                f"{result['target_congestion']:.4f}"
            )
            print(
                f"Decisions selecting a strictly less-congested "
                f"link than the dataset target: "
                f"{result['lower_congestion_percent']:.2f}%"
            )

        # Release the test samples before moving to the next topology.
        del samples

    print("\n" + "=" * 76)
    print("COMPARISON COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
