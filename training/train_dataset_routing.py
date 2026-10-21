"""Train the existing 11-feature routing scorer from archived route labels.

This is supervised next-hop imitation. It does not train either DQN because
the archives contain no agent actions or transition tuples.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from congestion_dataset import CongestionDatasetBuilder
from evaluate_routes import CongestionAgent, normalize_features


TRAIN_SIZES = (25, 30, 35, 40, 45, 50)
FEATURE_NAMES = [
    "bandwidth", "weight", "link_utilization", "link_losses",
    "average_packet_size", "queue_utilization", "queue_losses",
    "average_port_occupancy", "maximum_queue_occupancy",
    "queue_average_packet_size", "hop_distance",
]


def make_manifest(root, seed, train_archives_per_size, validation_archives_per_size,
                  simulations_per_archive):
    rng = random.Random(seed)
    manifest = {"seed": seed, "train": [], "validation": []}
    for size in TRAIN_SIZES:
        paths = sorted((root / "train" / str(size)).glob("*.tar.gz"))
        required = train_archives_per_size + validation_archives_per_size
        if len(paths) < required:
            raise ValueError(f"Need {required} train archives for size {size}; found {len(paths)}")
        shuffled = paths[:]
        rng.shuffle(shuffled)
        for split, selected in (
            ("train", shuffled[:train_archives_per_size]),
            ("validation", shuffled[train_archives_per_size:required]),
        ):
            for path in selected:
                manifest[split].append({
                    "size": size,
                    "archive": path.name,
                    "simulation_indices": list(range(simulations_per_archive)),
                })
    return manifest


def collate(samples, device):
    batch_size = len(samples)
    max_candidates = max(len(sample["candidates"]) for sample in samples)
    features = torch.zeros(batch_size, max_candidates, 11, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    targets = torch.tensor([sample["target_index"] for sample in samples], dtype=torch.long)
    for row, sample in enumerate(samples):
        candidate_features = [candidate["features"] for candidate in sample["candidates"]]
        count = len(candidate_features)
        features[row, :count] = torch.tensor(candidate_features, dtype=torch.float32)
        mask[row, :count] = True
    features = normalize_features(features.reshape(-1, 11)).reshape(batch_size, max_candidates, 11)
    return features.to(device), mask.to(device), targets.to(device)


def batches(samples, batch_size, rng=None):
    indices = list(range(len(samples)))
    if rng:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [samples[index] for index in indices[start:start + batch_size]]


def run_epoch(model, optimizer, criterion, simulations, builder, root, split,
              batch_size, device, train_mode, rng):
    model.train(train_mode)
    total_loss = 0.0
    correct = 0
    count = 0
    updates = 0
    order = list(simulations)
    if train_mode:
        rng.shuffle(order)
    with torch.set_grad_enabled(train_mode):
        for entry in order:
            for simulation_index in entry["simulation_indices"]:
                samples = builder.build_from_archive(
                    split="train", size=entry["size"],
                    archive_name=entry["archive"],
                    simulation_index=simulation_index, verbose=False,
                )
                for rows in batches(samples, batch_size, rng if train_mode else None):
                    x, mask, target = collate(rows, device)
                    scores = model(x.reshape(-1, 11)).reshape(x.shape[:2])
                    scores = scores.masked_fill(~mask, -1e9)
                    loss = criterion(scores, target)
                    if train_mode:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        updates += 1
                    total_loss += float(loss.item()) * len(rows)
                    correct += int((scores.argmax(dim=1) == target).sum().item())
                    count += len(rows)
                del samples
    return {"mean_cross_entropy": total_loss / max(count, 1),
            "top1_accuracy": correct / max(count, 1),
            "decisions": count, "optimizer_updates": updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="scalability")
    parser.add_argument("--output-dir", default="results/dataset_training_supervised_seed31415")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-archives-per-size", type=int, default=4)
    parser.add_argument("--validation-archives-per-size", type=int, default=1)
    parser.add_argument("--simulations-per-archive", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_archives_per_size,
           args.validation_archives_per_size, args.simulations_per_archive) < 1:
        parser.error("epoch, batch, archive and simulation counts must be positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    root = Path(args.dataset_root)
    manifest = make_manifest(root, args.seed, args.train_archives_per_size,
                             args.validation_archives_per_size,
                             args.simulations_per_archive)
    output.mkdir(parents=True, exist_ok=True)
    manifest["note"] = "Only scalability/train archives used; test split not opened by this trainer."
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = vars(args) | {"train_sizes": list(TRAIN_SIZES), "feature_names": FEATURE_NAMES,
                          "target": "next hop from archived routing file", "objective": "cross entropy only",
                          "agent_type": "supervised 11-feature routing scorer; not a DQN"}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CongestionAgent(input_dim=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    builder = CongestionDatasetBuilder(str(root))
    history = []
    best_val_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_rng = random.Random(args.seed + epoch)
        train_metrics = run_epoch(model, optimizer, criterion, manifest["train"], builder,
                                  root, "train", args.batch_size, device, True, epoch_rng)
        val_metrics = run_epoch(model, None, criterion, manifest["validation"], builder,
                                root, "train", args.batch_size, device, False, epoch_rng)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["mean_cross_entropy"] < best_val_loss:
            best_val_loss = val_metrics["mean_cross_entropy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": 11,
                "epoch": epoch,
                "feature_names": FEATURE_NAMES,
                "normalization": {"bandwidth_divisor": 100000, "average_packet_size_divisor": 2000,
                                  "queue_average_packet_size_divisor": 2000, "hop_distance_divisor": 50},
                "target_definition": "archived routing file's next hop",
                "objective": "cross entropy; no synthetic congestion reward",
                "training_seed": args.seed,
            }, output / "best_validation_model.pt")
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"epochs": args.epochs, "best_validation_cross_entropy": best_val_loss,
               "runtime_seconds": time.perf_counter() - started, "device": str(device),
               "checkpoint": str(output / "best_validation_model.pt"),
               "optimizer_updates": sum(row["train"]["optimizer_updates"] for row in history)}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
