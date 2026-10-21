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
        x,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    )

    x[:, 0] = x[:, 0] / 100000.0
    x[:, 4] = x[:, 4] / 2000.0
    x[:, 9] = x[:, 9] / 2000.0
    x[:, 10] = x[:, 10] / 50.0

    return x


def test_model():

    print("=" * 70)
    print("CONGESTION AGENT TEST")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    model_path = Path("models/congestion_agent.pt")

    print("\nLoading model:")
    print(model_path)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model = CongestionAgent(
        input_dim=checkpoint["input_dim"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # LOAD UNSEEN SIMULATION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING UNSEEN SIMULATION")
    print("=" * 70)

    builder = CongestionDatasetBuilder()

    samples = builder.build_from_archive(
        split="train",
        size=45,
        archive_name="results_45_400-2000_0_24.tar.gz",
        simulation_index=1
    )

    print("\nTesting samples:", len(samples))

    # ---------------------------------------------------------
    # TEST NEXT-HOP ACCURACY
    # ---------------------------------------------------------

    correct = 0
    total = 0

    for sample in samples:

        candidates = sample["candidates"]
        target_index = sample["target_index"]

        x = torch.tensor(
            [candidate["features"] for candidate in candidates],
            dtype=torch.float32,
            device=device
        )

        x = normalize_features(x)

        with torch.no_grad():
            scores = model(x)

        prediction = torch.argmax(scores).item()

        if prediction == target_index:
            correct += 1

        total += 1

    accuracy = (
        100.0 * correct / total
        if total > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("NEXT-HOP RESULTS")
    print("=" * 70)

    print(f"Correct decisions : {correct}")
    print(f"Total decisions   : {total}")
    print(f"Accuracy          : {accuracy:.2f}%")

    # ---------------------------------------------------------
    # SAMPLE PREDICTIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    for i, sample in enumerate(samples[:10]):

        candidates = sample["candidates"]
        target_index = sample["target_index"]

        x = torch.tensor(
            [candidate["features"] for candidate in candidates],
            dtype=torch.float32,
            device=device
        )

        x = normalize_features(x)

        with torch.no_grad():
            scores = model(x)

        prediction_index = torch.argmax(scores).item()

        print(
            f"Sample {i:02d} | "
            f"Candidates: {len(candidates)} | "
            f"Predicted index: {prediction_index} | "
            f"Target index: {target_index} | "
            f"{'CORRECT' if prediction_index == target_index else 'WRONG'}"
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
