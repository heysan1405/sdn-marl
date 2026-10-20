import sys
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim

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


def train_simulation(model, optimizer, criterion, samples, device):

    model.train()

    total_loss = 0.0
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

        scores = model(x)

        target = torch.tensor(
            [target_index],
            dtype=torch.long,
            device=device
        )

        loss = criterion(
            scores.unsqueeze(0),
            target
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        prediction = torch.argmax(scores).item()

        if prediction == target_index:
            correct += 1

        total += 1

    return (
        total_loss / total if total else 0.0,
        100.0 * correct / total if total else 0.0,
        total,
        correct
    )


def train():

    print("=" * 70)
    print("CONGESTION AGENT - CONTINUED STREAMING TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # DEVICE
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------

    topology_sizes = [25, 30, 35, 40, 45, 50]

    simulations_per_size = 10

    # We already completed epochs 1-5.
    # This run continues with epochs 6-10.
    start_epoch = 5
    epochs = 10

    print("\nTopology sizes:", topology_sizes)

    print(
        "Simulations per size:",
        simulations_per_size
    )

    print(
        "Total simulations per epoch:",
        len(topology_sizes) * simulations_per_size
    )

    print(
        "Continuing from epoch:",
        start_epoch + 1
    )

    print(
        "Training until epoch:",
        epochs
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = CongestionAgent(
        input_dim=11
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # LOAD EXISTING MODEL
    # ---------------------------------------------------------

    model_path = Path(
        "models/congestion_agent.pt"
    )

    if model_path.exists():

        print("\nLoading existing model from:")
        print(model_path)

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        # The previous training script saved a dictionary:
        #
        # {
        #     "model_state_dict": ...,
        #     "input_dim": 11
        # }
        #
        # Therefore we must load only model_state_dict.

        if isinstance(checkpoint, dict) and \
           "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            print(
                "Existing model weights loaded successfully."
            )

        else:

            # Safety fallback in case the file contains
            # a raw state_dict instead.

            model.load_state_dict(
                checkpoint
            )

            print(
                "Existing raw state_dict loaded successfully."
            )

        print(
            "Continuing training from epoch 5."
        )

    else:

        print(
            "\nWARNING: Existing model was not found."
        )

        print(
            "Training would start from a new model."
        )

    # ---------------------------------------------------------
    # DATASET BUILDER
    # ---------------------------------------------------------

    builder = CongestionDatasetBuilder()

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    for epoch in range(start_epoch, epochs):

        print("\n" + "=" * 70)
        print(
            f"EPOCH {epoch + 1}/{epochs}"
        )
        print("=" * 70)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        simulations_done = 0

        # -----------------------------------------------------
        # TOPOLOGY SIZES
        # -----------------------------------------------------

        for size in topology_sizes:

            archive_dir = (
                Path("scalability")
                / "train"
                / str(size)
            )

            archives = sorted(
                archive_dir.glob("*.tar.gz")
            )

            if not archives:

                print(
                    f"WARNING: No archives for "
                    f"{size}-node topology."
                )

                continue

            loaded = 0

            # -------------------------------------------------
            # ARCHIVES
            # -------------------------------------------------

            for archive in archives:

                if loaded >= simulations_per_size:
                    break

                # Each archive contains 25 simulations.

                for simulation_index in range(25):

                    if loaded >= simulations_per_size:
                        break

                    print(
                        f"\n[{size} nodes] "
                        f"Simulation {simulation_index} "
                        f"from {archive.name}"
                    )

                    try:

                        # -------------------------------------
                        # BUILD TRAINING DATA
                        # -------------------------------------

                        samples = (
                            builder.build_from_archive(
                                split="train",
                                size=size,
                                archive_name=archive.name,
                                simulation_index=simulation_index
                            )
                        )

                        # -------------------------------------
                        # TRAIN ON THIS SIMULATION
                        # -------------------------------------

                        (
                            loss,
                            accuracy,
                            decisions,
                            correct
                        ) = train_simulation(
                            model,
                            optimizer,
                            criterion,
                            samples,
                            device
                        )

                        # -------------------------------------
                        # ACCUMULATE STATISTICS
                        # -------------------------------------

                        epoch_loss += (
                            loss * decisions
                        )

                        epoch_correct += correct

                        epoch_total += decisions

                        loaded += 1

                        simulations_done += 1

                        print(
                            f"Decisions: {decisions} | "
                            f"Loss: {loss:.4f} | "
                            f"Accuracy: {accuracy:.2f}%"
                        )

                        # -------------------------------------
                        # RELEASE DATA
                        # -------------------------------------

                        del samples

                    except Exception as e:

                        print(
                            f"WARNING: Failed simulation: {e}"
                        )

            print(
                f"\nCompleted {loaded} simulations "
                f"for {size}-node topology."
            )

        # -----------------------------------------------------
        # EPOCH SUMMARY
        # -----------------------------------------------------

        overall_loss = (
            epoch_loss / epoch_total
            if epoch_total
            else 0.0
        )

        overall_accuracy = (
            100.0 * epoch_correct / epoch_total
            if epoch_total
            else 0.0
        )

        print("\n" + "-" * 70)

        print(
            f"EPOCH {epoch + 1} SUMMARY"
        )

        print("-" * 70)

        print(
            "Simulations processed:",
            simulations_done
        )

        print(
            "Total decisions:",
            epoch_total
        )

        print(
            f"Loss: {overall_loss:.4f}"
        )

        print(
            f"Accuracy: {overall_accuracy:.2f}%"
        )

        # -----------------------------------------------------
        # SAVE AFTER EVERY EPOCH
        # -----------------------------------------------------

        output_dir = Path("models")

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": 11,
                "epoch": epoch + 1
            },
            model_path
        )

        print(
            f"Checkpoint saved after epoch {epoch + 1}"
        )

    # ---------------------------------------------------------
    # FINAL SAVE
    # ---------------------------------------------------------

    output_dir = Path("models")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": 11,
            "epoch": epochs
        },
        model_path
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Model saved to:"
    )

    print(model_path)


if __name__ == "__main__":
    train()