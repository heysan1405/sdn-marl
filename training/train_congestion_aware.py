import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent))

from congestion_dataset import CongestionDatasetBuilder


# ============================================================
# CONGESTION-AWARE AGENT
# ============================================================

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


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(x):

    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    )

    # Bandwidth
    x[:, 0] = x[:, 0] / 100000.0

    # Average packet size
    x[:, 4] = x[:, 4] / 2000.0

    # Queue average packet size
    x[:, 9] = x[:, 9] / 2000.0

    # Hop distance
    x[:, 10] = x[:, 10] / 50.0

    return x


# ============================================================
# CONGESTION SCORE
# ============================================================

def calculate_congestion(features):

    """
    Calculate a normalized congestion score for each candidate link.

    Features:

    2 -> link utilization
    7 -> average port occupancy
    8 -> maximum queue occupancy

    We normalize the latter two using the observed queue capacity
    of 32 packets in the dataset.

    C(e) = (U + Q + O) / 3
    """

    utilization = torch.nan_to_num(
        features[:, 2],
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    avg_port_occupancy = torch.nan_to_num(
        features[:, 7],
        nan=0.0,
        posinf=32.0,
        neginf=0.0
    )

    max_queue_occupancy = torch.nan_to_num(
        features[:, 8],
        nan=0.0,
        posinf=32.0,
        neginf=0.0
    )

    # Normalize to approximately [0, 1]
    utilization = torch.clamp(
        utilization,
        min=0.0,
        max=1.0
    )

    avg_port_occupancy = torch.clamp(
        avg_port_occupancy / 32.0,
        min=0.0,
        max=1.0
    )

    max_queue_occupancy = torch.clamp(
        max_queue_occupancy / 32.0,
        min=0.0,
        max=1.0
    )

    congestion = (
        utilization
        + avg_port_occupancy
        + max_queue_occupancy
    ) / 3.0

    return congestion


# ============================================================
# TRAIN ONE SIMULATION
# ============================================================

def train_simulation(
    model,
    optimizer,
    criterion,
    samples,
    device,
    congestion_lambda
):

    model.train()

    total_loss = 0.0
    total_routing_loss = 0.0
    total_congestion_loss = 0.0

    correct = 0
    total = 0

    for sample in samples:

        candidates = sample["candidates"]
        target_index = sample["target_index"]

        # ----------------------------------------------------
        # Candidate features
        # ----------------------------------------------------

        x = torch.tensor(
            [
                candidate["features"]
                for candidate in candidates
            ],
            dtype=torch.float32,
            device=device
        )

        # ----------------------------------------------------
        # Calculate congestion BEFORE normalization
        # ----------------------------------------------------

        congestion = calculate_congestion(x)

        # ----------------------------------------------------
        # Normalize model input
        # ----------------------------------------------------

        x_normalized = normalize_features(x)

        # ----------------------------------------------------
        # Model scores
        # ----------------------------------------------------

        scores = model(x_normalized)

        # ----------------------------------------------------
        # Routing imitation loss
        # ----------------------------------------------------

        target = torch.tensor(
            [target_index],
            dtype=torch.long,
            device=device
        )

        routing_loss = criterion(
            scores.unsqueeze(0),
            target
        )

        # ----------------------------------------------------
        # Candidate probabilities
        # ----------------------------------------------------

        
        # ----------------------------------------------------
        # Hop-aware candidate probabilities
        # ----------------------------------------------------

        # The target next hop provides a reference for the
        # remaining hop distance to the destination.
        target_distance = x[target_index, 10]

        # Consider candidates that are no farther from the
        # destination than the reference next hop.
        eligible = x[:, 10] <= target_distance + 1e-6

        # The reference candidate should always be eligible.
        eligible[target_index] = True

        # Calculate congestion probabilities only over
        # eligible candidates.
        eligible_scores = scores.masked_fill(
            ~eligible, -1e9
        )

        probabilities = torch.softmax(
            eligible_scores, dim=0
        )

        # ----------------------------------------------------
        # Hop-aware expected congestion
        # ----------------------------------------------------

        congestion_loss = torch.sum(
            probabilities * congestion
        )


        # ----------------------------------------------------
        # Combined objective
        # ----------------------------------------------------

        loss = (
            routing_loss
            + congestion_lambda * congestion_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_routing_loss += routing_loss.item()

        total_congestion_loss += congestion_loss.item()

        prediction = torch.argmax(scores).item()

        if prediction == target_index:
            correct += 1

        total += 1

    if total == 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0
        )

    return (
        total_loss / total,
        total_routing_loss / total,
        total_congestion_loss / total,
        100.0 * correct / total,
        total,
        correct
    )


# ============================================================
# TRAINING
# ============================================================

def train():

    print("=" * 70)
    print("CONGESTION-AWARE AGENT TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DATASET CONFIGURATION
    # --------------------------------------------------------

    topology_sizes = [
        25,
        30,
        35,
        40,
        45,
        50
    ]

    simulations_per_size = 10

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    start_epoch = 0

    epochs = 10

    learning_rate = 0.001

    # Strength of congestion penalty.
    #
    # Start relatively small so that the model does not
    # completely abandon the routing behavior learned from
    # the dataset.
    #
    # This is the main hyperparameter we can experiment with.
    # --------------------------------------------------------

    congestion_lambda = 0.2

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
        "Epochs:",
        epochs
    )

    print(
        "Learning rate:",
        learning_rate
    )

    print(
        "Congestion lambda:",
        congestion_lambda
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = CongestionAgent(
        input_dim=11
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # LOAD BASELINE MODEL
    # --------------------------------------------------------

    baseline_path = Path(
        "models/congestion_agent.pt"
    )

    output_path = Path(
        "models/congestion_agent_aware.pt"
    )

    if baseline_path.exists():

        print("\nLoading baseline model:")

        print(baseline_path)

        checkpoint = torch.load(
            baseline_path,
            map_location=device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            print(
                "Baseline model loaded successfully."
            )

        else:

            model.load_state_dict(
                checkpoint
            )

            print(
                "Raw baseline state_dict loaded."
            )

    else:

        print(
            "\nWARNING:"
        )

        print(
            "Baseline model was not found."
        )

        print(
            "Training will start from a new model."
        )

    # --------------------------------------------------------
    # DATASET BUILDER
    # --------------------------------------------------------

    builder = CongestionDatasetBuilder()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        epochs
    ):

        print("\n" + "=" * 70)

        print(
            f"EPOCH {epoch + 1}/{epochs}"
        )

        print("=" * 70)

        epoch_loss = 0.0

        epoch_routing_loss = 0.0

        epoch_congestion_loss = 0.0

        epoch_correct = 0

        epoch_total = 0

        simulations_done = 0

        # ----------------------------------------------------
        # TOPOLOGY SIZES
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # ARCHIVES
            # ------------------------------------------------

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

                        # ------------------------------------
                        # BUILD TRAINING DATA
                        # ------------------------------------

                        samples = (
                            builder.build_from_archive(
                                split="train",
                                size=size,
                                archive_name=archive.name,
                                simulation_index=simulation_index
                            )
                        )

                        # ------------------------------------
                        # TRAIN
                        # ------------------------------------

                        (
                            loss,
                            routing_loss,
                            congestion_loss,
                            accuracy,
                            decisions,
                            correct
                        ) = train_simulation(
                            model,
                            optimizer,
                            criterion,
                            samples,
                            device,
                            congestion_lambda
                        )

                        # ------------------------------------
                        # ACCUMULATE
                        # ------------------------------------

                        epoch_loss += (
                            loss * decisions
                        )

                        epoch_routing_loss += (
                            routing_loss * decisions
                        )

                        epoch_congestion_loss += (
                            congestion_loss * decisions
                        )

                        epoch_correct += correct

                        epoch_total += decisions

                        loaded += 1

                        simulations_done += 1

                        # ------------------------------------
                        # PRINT SIMULATION RESULT
                        # ------------------------------------

                        print(
                            f"Decisions: {decisions} | "
                            f"Loss: {loss:.4f} | "
                            f"Routing: {routing_loss:.4f} | "
                            f"Congestion: {congestion_loss:.4f} | "
                            f"Accuracy: {accuracy:.2f}%"
                        )

                        # ------------------------------------
                        # RELEASE DATA
                        # ------------------------------------

                        del samples

                    except Exception as e:

                        print(
                            f"WARNING: "
                            f"Failed simulation: {e}"
                        )

            print(
                f"\nCompleted {loaded} simulations "
                f"for {size}-node topology."
            )

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        if epoch_total:

            overall_loss = (
                epoch_loss
                / epoch_total
            )

            overall_routing_loss = (
                epoch_routing_loss
                / epoch_total
            )

            overall_congestion_loss = (
                epoch_congestion_loss
                / epoch_total
            )

            overall_accuracy = (
                100.0
                * epoch_correct
                / epoch_total
            )

        else:

            overall_loss = 0.0

            overall_routing_loss = 0.0

            overall_congestion_loss = 0.0

            overall_accuracy = 0.0

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
            f"Total Loss: {overall_loss:.4f}"
        )

        print(
            f"Routing Loss: {overall_routing_loss:.4f}"
        )

        print(
            f"Congestion Loss: "
            f"{overall_congestion_loss:.4f}"
        )

        print(
            f"Accuracy: "
            f"{overall_accuracy:.2f}%"
        )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        output_dir = Path("models")

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_dim":
                    11,

                "epoch":
                    epoch + 1,

                "congestion_lambda":
                    congestion_lambda
            },
            output_path
        )

        print(
            f"Checkpoint saved after epoch "
            f"{epoch + 1}"
        )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    output_dir = Path("models")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                11,

            "epoch":
                epochs,

            "congestion_lambda":
                congestion_lambda
        },
        output_path
    )

    print("\n" + "=" * 70)

    print(
        "CONGESTION-AWARE TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        "Model saved to:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()