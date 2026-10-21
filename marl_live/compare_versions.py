"""Post-hoc paired comparison against preserved v1 raw runs on identical seeds.

The reused historical test seeds make this diagnostic, not a new confirmatory
test. New claims rely on the fresh v4 held-out set.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load(run_dirs):
    rows = []
    for run_idx, directory in enumerate(run_dirs):
        with (Path(directory) / "evaluation_runs.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["run_idx"] = run_idx
                row["seed"] = int(row["seed"])
                for field in ("delivered", "hops", "loop", "mean_selected_link_utilization"):
                    value = row.get(field)
                    row[field] = float(value) if value not in (None, "", "None") else None
                rows.append(row)
    return rows


def compare(old_runs, new_runs, out_dir, bootstrap=5000, seed=907):
    old, new = _load(old_runs), _load(new_runs)
    keys = ("shortest_path", "congestion_aware_baseline", "congestion_policy_only",
            "failure_policy_only", "marl_orchestrated")
    old_map = {(r["run_idx"], r["scenario"], r["seed"], r["policy"]): r for r in old}
    new_map = {(r["run_idx"], r["scenario"], r["seed"], r["policy"]): r for r in new}
    rng = np.random.default_rng(seed)
    output = []
    for scenario in sorted({r["scenario"] for r in old}):
        cases = sorted({(r["run_idx"], r["seed"]) for r in old if r["scenario"] == scenario})
        for policy in keys:
            for metric in ("delivered", "hops", "mean_selected_link_utilization", "loop"):
                diffs_by_seed, old_by_seed, new_by_seed = {}, {}, {}
                for run_idx, case_seed in cases:
                    a = old_map.get((run_idx, scenario, case_seed, policy))
                    b = new_map.get((run_idx, scenario, case_seed, policy))
                    if not a or not b or a[metric] is None or b[metric] is None:
                        continue
                    if metric in ("hops", "mean_selected_link_utilization") and not (a["delivered"] and b["delivered"]):
                        continue
                    diffs_by_seed.setdefault(case_seed, []).append(b[metric] - a[metric])
                    old_by_seed.setdefault(case_seed, []).append(a[metric])
                    new_by_seed.setdefault(case_seed, []).append(b[metric])
                seeds = sorted(diffs_by_seed)
                if not seeds:
                    continue
                diffs = np.asarray([np.mean(diffs_by_seed[s]) for s in seeds], dtype=float)
                old_vals = np.asarray([np.mean(old_by_seed[s]) for s in seeds], dtype=float)
                new_vals = np.asarray([np.mean(new_by_seed[s]) for s in seeds], dtype=float)
                sample = rng.choice(diffs, size=(bootstrap, diffs.size), replace=True).mean(axis=1)
                output.append({
                    "scenario": scenario, "policy": policy, "metric": metric,
                    "n_paired_eval_seeds": len(seeds), "old_mean": float(old_vals.mean()),
                    "new_mean": float(new_vals.mean()), "new_minus_old": float(diffs.mean()),
                    "paired_bootstrap_95_ci_low": float(np.quantile(sample, .025)),
                    "paired_bootstrap_95_ci_high": float(np.quantile(sample, .975)),
                    "historical_test_seed_reused": True,
                })
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "before_after.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    (out_dir / "metadata.json").write_text(json.dumps({
        "old_run_dirs": [str(Path(p)) for p in old_runs],
        "new_run_dirs": [str(Path(p)) for p in new_runs],
        "use": "diagnostic-only: historical held-out seeds were previously inspected and must not be used to choose models",
        "pairing": "same scenario/topology/traffic seed, average three trained model runs within each seed, paired bootstrap across 100 seeds",
        "bootstrap_resamples": bootstrap, "bootstrap_seed": seed,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-runs", nargs=3, required=True)
    p.add_argument("--new-runs", nargs=3, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=907)
    a = p.parse_args()
    compare(a.old_runs, a.new_runs, a.out, a.bootstrap, a.seed)


if __name__ == "__main__":
    main()
