"""Aggregate repeated training seeds and paired held-out evaluation runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyze(run_dirs, out_dir, bootstrap=5000, seed=107):
    rows = []
    for run_index, run_dir in enumerate(run_dirs):
        path = Path(run_dir) / "evaluation_runs.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["training_run"] = run_index
                rows.append(row)
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in ("delivered", "hops", "loop", "invalid_actions", "fallback_actions",
                    "dead_end", "mean_selected_link_utilization"):
            value = row.get(key)
            row[key] = float(value) if value not in ("", None, "None") else None
    scenarios = sorted({r["scenario"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    metrics = ("delivered", "hops", "mean_selected_link_utilization", "loop",
               "invalid_actions", "fallback_actions", "dead_end")
    summary = []
    for scenario in scenarios:
        for policy in policies:
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            for metric in metrics:
                vals = np.asarray([r[metric] for r in group if r[metric] is not None], dtype=float)
                if vals.size:
                    summary.append({"scenario": scenario, "policy": policy, "metric": metric,
                                    "n": int(vals.size), "mean": float(vals.mean()),
                                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0})
    paired = []
    rng = np.random.default_rng(seed)
    by_key = {(r["training_run"], r["scenario"], r["seed"], r["policy"]): r for r in rows}
    comparisons = tuple(policy for policy in policies
                        if policy not in ("shortest_path", "failure_shortest_path"))
    for scenario in scenarios:
        for policy in comparisons:
            references = ("shortest_path", "congestion_aware_baseline", "coordination_disabled")
            for reference in references:
                if policy == reference:
                    continue
                for metric in ("delivered", "hops", "mean_selected_link_utilization",
                               "loop", "invalid_actions", "fallback_actions"):
                    differences_by_eval_seed = {}
                    cases = sorted({(r["training_run"], r["seed"]) for r in rows if r["scenario"] == scenario})
                    for run_index, case_seed in cases:
                        a = by_key.get((run_index, scenario, case_seed, policy))
                        b = by_key.get((run_index, scenario, case_seed, reference))
                        if not a or not b or a[metric] is None or b[metric] is None:
                            continue
                        # Route-cost metrics compare only cases where both policies delivered.
                        if metric in ("hops", "mean_selected_link_utilization") and not (a["delivered"] and b["delivered"]):
                            continue
                        differences_by_eval_seed.setdefault(case_seed, []).append(float(a[metric] - b[metric]))
                    # Average across trained checkpoints within each fixed eval seed;
                    # bootstrap the independent held-out seed units, not duplicate runs.
                    vals = np.asarray([np.mean(diffs) for _, diffs in sorted(differences_by_eval_seed.items())], dtype=float)
                    if not vals.size:
                        continue
                    sampled = rng.choice(vals, size=(bootstrap, vals.size), replace=True).mean(axis=1)
                    paired.append({"scenario": scenario, "policy": policy,
                                   "reference": reference, "metric": metric,
                                   "n_paired": int(vals.size), "mean_difference": float(vals.mean()),
                                   "bootstrap_95_ci_low": float(np.quantile(sampled, 0.025)),
                                   "bootstrap_95_ci_high": float(np.quantile(sampled, 0.975))})
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (("descriptive_summary.csv", summary),
                              ("paired_comparisons.csv", paired)):
        if records:
            with (out_dir / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader(); writer.writerows(records)
    (out_dir / "metadata.json").write_text(json.dumps({
        "input_run_directories": [str(Path(p)) for p in run_dirs],
        "paired_seed_unit": "evaluation seed within each trained checkpoint run",
        "confidence_interval": f"paired percentile bootstrap, {bootstrap} resamples",
        "bootstrap_seed": seed,
        "note": "Generated NetworkX simulator estimates; no Mininet outcomes are included.",
    }, indent=2) + "\n", encoding="utf-8")


def analyze_training(training_runs, out_dir, window_size=100):
    """Write per-window learning diagnostics from raw training episode files."""
    records = []
    for run_dir in training_runs:
        path = Path(run_dir) / "training_episodes.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        groups = {}
        for row in rows:
            episode = int(row["episode"])
            groups.setdefault((episode - 1) // window_size + 1, []).append(row)
        for window, group in sorted(groups.items()):
            def avg(field):
                vals = [float(r[field]) for r in group if r.get(field) not in (None, "")]
                return float(np.mean(vals)) if vals else None
            records.append({
                "run": str(Path(run_dir)), "agent": Path(run_dir).name,
                "window": window, "first_episode": int(group[0]["episode"]),
                "last_episode": int(group[-1]["episode"]), "n": len(group),
                "mean_reward": avg("reward"), "delivery_rate": avg("delivered"),
                "invalid_action_rate": (sum(int(r["invalid_actions"]) for r in group) /
                                         max(1, sum(int(r.get("decisions") or r.get("hops", 0))
                                                    for r in group))),
                "loop_rate": avg("loop"),
                "dead_end_rate": avg("dead_end"), "mean_hops": avg("hops"),
                "mean_selected_background_utilization": avg("mean_selected_link_utilization"),
                "mean_weighted_path_cost": avg("sum_weighted_path_cost"),
                "mean_policy_loss": avg("policy_loss"), "mean_value_loss": avg("value_loss"),
                "mean_entropy": avg("entropy"),
            })
    if records:
        path = Path(out_dir) / "training_windows.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader(); writer.writerows(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=107)
    p.add_argument("--training-runs", nargs="*", default=[])
    a = p.parse_args()
    analyze(a.runs, a.out, a.bootstrap, a.seed)
    if a.training_runs:
        analyze_training(a.training_runs, a.out)


if __name__ == "__main__":
    main()
