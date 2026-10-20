"""Paired bootstrap and sign-flip analysis of evaluator JSON output."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("delivery_rate", "throughput_mbps", "packet_loss_ratio_offered", "packet_loss",
           "congestion_ratio", "mean_link_utilization", "max_link_utilization",
           "p95_link_utilization", "mean_delay_ms", "mean_hops",
           "recovery_time_steps", "unserved_demand_steps")
COMPARISONS = {
    "congestion_dqn": ("shortest_path", "no_agent"),
    "failure_dqn": ("shortest_path", "no_agent"),
    "orchestrated": ("shortest_path", "no_agent", "congestion_dqn", "failure_dqn", "orchestrated_no_override"),
    "orchestrated_no_override": ("orchestrated", "shortest_path", "no_agent"),
}


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, samples: int) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(values)))
    simulated = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(simulated >= observed)) / (samples + 1))


def holm_adjust(rows: list[dict]) -> None:
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["p_sign_flip"])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * rows[index]["p_sign_flip"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="evaluate_dqn JSON file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("bootstrap-samples must be at least 1000")
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in report["runs"]:
        grouped[(row["scenario"], row["method"])][int(row["seed"])] = row

    rng = np.random.default_rng(args.seed)
    descriptives, paired = [], []
    for (scenario, method), by_seed in sorted(grouped.items()):
        for metric in METRICS:
            values = np.asarray([r[metric] for _, r in sorted(by_seed.items()) if r.get(metric) is not None], dtype=float)
            if not len(values):
                continue
            lo, hi = bootstrap_mean_ci(values, rng, args.bootstrap_samples)
            descriptives.append({"scenario": scenario, "method": method, "metric": metric,
                                 "n": len(values), "mean": float(values.mean()),
                                 "stdev": float(values.std(ddof=1)) if len(values) > 1 else None,
                                 "median": float(np.median(values)), "mean_ci95_low": lo,
                                 "mean_ci95_high": hi})

    for scenario in sorted({scenario for scenario, _ in grouped}):
        for method, baselines in COMPARISONS.items():
            agent_rows = grouped.get((scenario, method), {})
            for baseline in baselines:
                base_rows = grouped.get((scenario, baseline), {})
                shared = sorted(set(agent_rows) & set(base_rows))
                if not shared:
                    continue
                for metric in METRICS:
                    pairs = [(agent_rows[s].get(metric), base_rows[s].get(metric)) for s in shared]
                    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                    if not pairs:
                        continue
                    diffs = np.asarray([a - b for a, b in pairs], dtype=float)
                    lo, hi = bootstrap_mean_ci(diffs, rng, args.bootstrap_samples)
                    pvalue = sign_flip_pvalue(diffs, rng, args.bootstrap_samples)
                    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
                    paired.append({"scenario": scenario, "policy": method, "baseline": baseline,
                                   "metric": metric, "n_pairs": len(diffs),
                                   "mean_paired_difference": float(diffs.mean()),
                                   "paired_difference_ci95_low": lo,
                                   "paired_difference_ci95_high": hi,
                                   "cohen_dz": float(diffs.mean() / sd) if sd else None,
                                   "p_sign_flip": pvalue})
    holm_adjust(paired)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("descriptive_summary.csv", descriptives), ("paired_comparisons.csv", paired)):
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    metadata = {"input": str(Path(args.input)), "bootstrap_samples": args.bootstrap_samples,
                "random_seed": args.seed, "resampling": "paired percentile bootstrap; sign-flip randomization p-values",
                "multiple_testing": "Holm step-down across all emitted policy/scenario/metric comparisons",
                "interpretation": "Simulator-generated outcomes; paired association only, not causal evidence beyond this scenario generator."}
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "descriptive_rows": len(descriptives),
                      "paired_rows": len(paired)}, indent=2))


if __name__ == "__main__":
    main()
