"""Evaluate a frozen dataset-trained scorer on held-out archives only."""

import argparse
import csv
import json
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from congestion_dataset import CongestionDatasetBuilder
from train_dataset_routing import batches, collate
from evaluate_routes import (
    CongestionAgent, DEVICE, evaluate_model, load_context, load_model,
)


def candidate_accuracy(model, samples):
    correct = total = 0
    with torch.inference_mode():
        for rows in batches(samples, 512):
            x, mask, target = collate(rows, DEVICE)
            scores = model(x.reshape(-1, 11)).reshape(x.shape[:2])
            scores = scores.masked_fill(~mask, -1e9)
            correct += int((scores.argmax(dim=1) == target).sum().item())
            total += len(rows)
    return {"correct": int(correct), "decisions": total,
            "top1_accuracy": correct / total if total else None}


def reference_metrics(graph, route_items, link_state):
    delivered = loops = dead_ends = 0
    hops = []
    congestion = []
    for (source, destination), path in route_items:
        if len(path) != len(set(path)):
            loops += 1
            continue
        if any(not graph.has_edge(u, v) for u, v in zip(path[:-1], path[1:])):
            dead_ends += 1
            continue
        delivered += 1
        hops.append(len(path) - 1)
        values = []
        for u, v in zip(path[:-1], path[1:]):
            state = link_state.get((u, v))
            if state is not None:
                from evaluate_routes import link_congestion
                values.append(link_congestion(state["metrics"]))
        if values:
            congestion.append(sum(values) / len(values))
    return {"evaluated_pairs": len(route_items), "delivered": delivered,
            "delivery_rate": delivered / len(route_items) if route_items else None,
            "loops": loops, "dead_ends": dead_ends,
            "mean_hops": sum(hops) / len(hops) if hops else None,
            "mean_recorded_path_congestion": sum(congestion) / len(congestion) if congestion else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default="models/congestion_agent.pt")
    parser.add_argument("--dataset-root", default="scalability")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 300])
    parser.add_argument("--archives-per-size", type=int, default=1)
    parser.add_argument("--simulations-per-archive", type=int, default=2)
    parser.add_argument("--pairs-per-simulation", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31416)
    parser.add_argument("--output-dir", default="results/dataset_training_eval_seed31415")
    args = parser.parse_args()
    if min(args.archives_per_size, args.simulations_per_archive, args.pairs_per_simulation) < 1:
        parser.error("archive, simulation and pair counts must be positive")
    for path in (args.checkpoint, args.baseline_checkpoint):
        if not Path(path).is_file():
            parser.error(f"checkpoint not found: {path}")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, DEVICE)
    baseline = load_model(args.baseline_checkpoint, DEVICE)
    builder = CongestionDatasetBuilder(args.dataset_root)
    rng = random.Random(args.seed)
    selected = {}
    rows = []
    with (output / "evaluation.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        for size in args.sizes:
            archives = sorted((Path(args.dataset_root) / "test" / str(size)).glob("*.tar.gz"))
            if not archives:
                raise FileNotFoundError(f"No held-out test archives for size {size}")
            chosen = rng.sample(archives, min(args.archives_per_size, len(archives)))
            selected[str(size)] = [path.name for path in chosen]
            for archive in chosen:
                for sim_index in range(args.simulations_per_archive):
                    base = {"size": size, "archive": archive.name,
                            "simulation_index": sim_index, "seed": args.seed}
                    try:
                        samples = builder.build_from_archive(
                            "test", size, archive.name, sim_index, verbose=False
                        )
                        graph, routes, distances, link_state = load_context(
                            "test", size, archive.name, sim_index
                        )
                        pairs = [(key, path) for key, path in routes.items() if len(path) >= 2]
                        if not pairs:
                            raise ValueError("No routed source-destination pairs")
                        sampled = rng.sample(pairs, min(args.pairs_per_simulation, len(pairs)))
                        sampled_routes = dict(sampled)
                        for name, candidate_model, constrained in (
                            ("baseline_checkpoint", baseline, False),
                            ("dataset_trained_scorer", model, False),
                            ("dataset_trained_scorer_distance_constrained", model, True),
                        ):
                            metrics = evaluate_model(name, candidate_model, graph,
                                                     sampled_routes, distances,
                                                     link_state, enforce_progress=constrained)
                            rows.append(base | {"policy": name, **metrics})
                        rows.append(base | {"policy": "archived_route_reference",
                                            **reference_metrics(graph, sampled, link_state)})
                        rows.append(base | {"policy": "dataset_supervised_next_hop_accuracy",
                                            **candidate_accuracy(model, samples)})
                    except (IndexError, ValueError) as exc:
                        rows.append(base | {"policy": "failed_run", "error": str(exc)})
    report = {"dataset_root": args.dataset_root, "split": "test",
              "checkpoint": args.checkpoint,
              "baseline_checkpoint": args.baseline_checkpoint,
              "selection_seed": args.seed, "selected_archives": selected,
              "pairs_per_simulation": args.pairs_per_simulation,
              "test_metrics_are_archived_simulator_measurements_not_new_rollouts": True,
              "runs": rows}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rows:
        with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"output": str(output), "runs": len(rows), "selected_archives": selected}, indent=2))


if __name__ == "__main__":
    main()
