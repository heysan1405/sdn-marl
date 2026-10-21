import sys
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------
# IMPORT CONGESTION DATASET BUILDER
# ---------------------------------------------------------

sys.path.append(str(Path(__file__).resolve().parent))

from congestion_dataset import CongestionDatasetBuilder


# =========================================================
# CONGESTION AGENT
# =========================================================

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


# =========================================================
# FEATURE NORMALIZATION
# =========================================================

def normalize_features(x):

    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    )

    # Feature 0: bandwidth
    x[:, 0] = x[:, 0] / 100000.0

    # Feature 4: average packet size
    x[:, 4] = x[:, 4] / 2000.0

    # Feature 9: queue average packet size
    x[:, 9] = x[:, 9] / 2000.0

    # Feature 10: hop distance to destination
    x[:, 10] = x[:, 10] / 50.0

    return x


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CONGESTION AGENT - 200 NODE GENERALIZATION TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # DEVICE
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -----------------------------------------------------
    # LOAD TRAINED MODEL
    # -----------------------------------------------------

    model_path = Path(
        "models/congestion_agent.pt"
    )

    if not model_path.exists():

        print("\nERROR: Model not found:")
        print(model_path)

        return

    model = CongestionAgent(
        input_dim=11
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nModel loaded successfully.")

    # -----------------------------------------------------
    # TEST DATA
    # -----------------------------------------------------

    size = 200

    archive_name = (
        "results_400-2000_1_1.tar.gz"
    )

    simulation_index = 0

    print("\nTest topology:")
    print("Nodes:", size)
    print("Archive:", archive_name)
    print("Simulation:", simulation_index)

    # -----------------------------------------------------
    # BUILD TEST SAMPLES
    # -----------------------------------------------------

    builder = CongestionDatasetBuilder()

    print("\nBuilding test samples...")

    samples = builder.build_from_archive(
        split="test",
        size=size,
        archive_name=archive_name,
        simulation_index=simulation_index
    )

    print(
        "Total decisions:",
        len(samples)
    )

    # -----------------------------------------------------
    # TEST MODEL
    # -----------------------------------------------------

    correct = 0
    total = 0

    print("\nFirst 10 predictions:")
    print("-" * 70)

    with torch.no_grad():

        for i, sample in enumerate(samples):

            candidates = sample["candidates"]

            target_index = sample["target_index"]

            # ---------------------------------------------
            # Convert candidate features to tensor
            # ---------------------------------------------

            x = torch.tensor(
                [
                    candidate["features"]
                    for candidate in candidates
                ],
                dtype=torch.float32,
                device=device
            )

            # ---------------------------------------------
            # Normalize
            # ---------------------------------------------

            x = normalize_features(x)

            # ---------------------------------------------
            # Model prediction
            # ---------------------------------------------

            scores = model(x)

            prediction = torch.argmax(
                scores
            ).item()

            # ---------------------------------------------
            # Accuracy
            # ---------------------------------------------

            if prediction == target_index:

                correct += 1

            total += 1

            # ---------------------------------------------
            # Show first 10 predictions
            # ---------------------------------------------

            if i < 10:

                result = (
                    "CORRECT"
                    if prediction == target_index
                    else "WRONG"
                )

                print(
                    f"Sample {i:02d} | "
                    f"Candidates: {len(candidates)} | "
                    f"Predicted index: {prediction} | "
                    f"Target index: {target_index} | "
                    f"{result}"
                )

    # -----------------------------------------------------
    # CALCULATE ACCURACY
    # -----------------------------------------------------

    accuracy = (
        100.0 * correct / total
        if total > 0
        else 0.0
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("200-NODE TEST RESULTS")
    print("=" * 70)

    print(
        "Correct decisions :",
        correct
    )

    print(
        "Total decisions   :",
        total
    )

    print(
        f"Accuracy          : {accuracy:.2f}%"
    )

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()