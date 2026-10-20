
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


def congestion_scores(x):
    utilization = x[:, 2].clamp(0.0, 1.0)
    occupancy = (x[:, 7] / 32.0).clamp(0.0, 1.0)
    max_queue = (x[:, 8] / 32.0).clamp(0.0, 1.0)
    return (utilization + occupancy + max_queue) / 3.0


def load_model(path, device):
    model = CongestionAgent().to(device)
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_path = Path("models/congestion_agent.pt")
    aware_path = Path("models/congestion_agent_aware.pt")

    for path in (baseline_path, aware_path):
        if not path.exists():
            print(f"ERROR: Missing model: {path}")
            return

    baseline = load_model(baseline_path, device)
    aware = load_model(aware_path, device)

    tests = [
        (100, "results_400-2000_0_0.tar.gz"),
        (200, "results_400-2000_1_1.tar.gz"),
        (300, "results_400-2000_0_0.tar.gz"),
    ]

    builder = CongestionDatasetBuilder()

    for size, archive in tests:
        print("\n" + "=" * 65)
        print(f"DECISION AUDIT — {size} NODES")
        print("=" * 65)

        samples = builder.build_from_archive(
            split="test",
            size=size,
            archive_name=archive,
            simulation_index=0
        )

        counts = {
            "total": 0,
            "same_choice": 0,
            "different_choice": 0,
            "aware_lower": 0,
            "aware_higher": 0,
            "equal_congestion": 0,
            "changed_and_correct": 0,
            "changed_and_wrong": 0,
        }

        congestion_delta_sum = 0.0
        changed_delta_sum = 0.0
        changed_count = 0
        examples = []

        with torch.no_grad():
            for sample in samples:
                candidates = sample["candidates"]
                target = sample["target_index"]

                raw_x = torch.tensor(
                    [c["features"] for c in candidates],
                    dtype=torch.float32,
                    device=device
                )

                x = normalize_features(raw_x)
                congestion = congestion_scores(raw_x)

                base_pred = baseline(x).argmax().item()
                aware_pred = aware(x).argmax().item()

                base_c = congestion[base_pred].item()
                aware_c = congestion[aware_pred].item()
                delta = aware_c - base_c

                counts["total"] += 1
                congestion_delta_sum += delta

                if base_pred == aware_pred:
                    counts["same_choice"] += 1
                else:
                    counts["different_choice"] += 1
                    changed_count += 1
                    changed_delta_sum += delta

                    if aware_c < base_c:
                        counts["aware_lower"] += 1
                    elif aware_c > base_c:
                        counts["aware_higher"] += 1
                    else:
                        counts["equal_congestion"] += 1

                    if aware_pred == target:
                        counts["changed_and_correct"] += 1
                    else:
                        counts["changed_and_wrong"] += 1

                    if len(examples) < 8:
                        examples.append(
                            (
                                sample.get("source"),
                                sample.get("current"),
                                sample.get("destination"),
                                candidates[base_pred]["next_hop"]
                                if "next_hop" in candidates[base_pred]
                                else candidates[base_pred].get("neighbor"),
                                candidates[aware_pred].get("next_hop")
                                if "next_hop" in candidates[aware_pred]
                                else candidates[aware_pred].get("neighbor"),
                                base_c,
                                aware_c,
                                target,
                                aware_pred,
                            )
                        )

        total = counts["total"]
        changed = counts["different_choice"]

        print(f"Total decisions: {total}")
        print(
            f"Same next hop: {counts['same_choice']} "
            f"({100 * counts['same_choice'] / total:.2f}%)"
        )
        print(
            f"Different next hop: {changed} "
            f"({100 * changed / total:.2f}%)"
        )
        print(
            f"Mean congestion delta (aware - baseline): "
            f"{congestion_delta_sum / total:.6f}"
        )

        if changed:
            print(
                f"Among changed decisions, lower congestion: "
                f"{counts['aware_lower']}"
            )
            print(
                f"Among changed decisions, higher congestion: "
                f"{counts['aware_higher']}"
            )
            print(
                f"Among changed decisions, equal congestion: "
                f"{counts['equal_congestion']}"
            )
            print(
                f"Changed and matches dataset target: "
                f"{counts['changed_and_correct']}"
            )
            print(
                f"Changed and differs from dataset target: "
                f"{counts['changed_and_wrong']}"
            )
            print(
                f"Mean congestion delta on changed decisions: "
                f"{changed_delta_sum / changed:.6f}"
            )

        print("\nFirst changed-decision examples:")
        if not examples:
            print("No different choices found.")
        else:
            for example in examples:
                print(example)

        del samples


if __name__ == "__main__":
    main()
