# MARL Routing Performance Diagnosis and Follow-up

All code and newly generated runs for this phase are inside `marl_live/`.
Historical files and prior result directories were preserved. The old
checkpoints remain available for comparison and are rejected by the v2 schema
loader rather than silently treated as compatible.

## 1. Audit of the previous experiment

Inspected implementation: `environment.py`, `features.py`, `policy.py`,
`orchestrator.py`, `train.py`, `evaluate.py`, `analyze.py`, and the old raw files
under `runs/final_seed202611*/`, `runs/final_eval_*/`, and
`runs/final_aggregate/`.

Findings supported by code and raw records:

1. **The old failure was not a loop or an invalid action.** In the old held-out
   raw runs, every policy had zero loops and zero invalid actions. However,
   `candidate_observation()` masked only failed and visited neighbors. It did
   not mask a currently unvisited neighbor whose remaining route to the
   destination required revisiting a visited node. The policy could therefore
   make a locally legal choice that trapped the route. Failed congestion paths
   in `final_eval_*/evaluation_runs.csv` include `[0, 3, 5, 2]` (5/300
   congestion runs) and `[0, 2, 6, 3]`; combined failures also ended at `[0, 3]`
   or `[0, 2]` (3/300). The evaluator reported those as undelivered paths.
2. **The old congestion reward and reported metric differed.** Training used a
   quadratic penalty only above a projected-utilization threshold, adding the
   fixed demand to background load. Evaluation reported average background
   utilization on the selected path. Thus the actor could trade off hop length
   and a thresholded projected value while the report emphasized a different
   quantity. The old weighted baseline used the whole graph and edge cost
   `1 + 3 * utilization`; agents saw only adjacent candidate features and
   unweighted destination progress.
3. **The old Orchestrator did not coordinate the two learned proposals.** It
   dispatched to the congestion actor if there was no failure and to the
   failure actor if *any* failure existed. In the preserved raw runs, its path
   was identical to the failure-only path in every failure and combined
   episode (300/300 each). That actor was trained with hop/delivery rewards,
   not a congestion objective, explaining why it ignored the congestion actor
   during combined conditions.
4. **Pairing was otherwise consistent.** Each policy reset the generated graph
   with the same scenario seed, so topology, capacities, background loads, and
   injected link were matched. The old aggregate combined three trained
   checkpoint pairs over 100 evaluation seeds (300 rows per policy/scenario).
   The old paired bootstrap averaged the model runs within each evaluation seed
   before resampling those 100 seed units.
5. **The utilization statistic is a narrow route metric.** It is the mean
   generated background utilization on the selected route. It is not
   post-admission utilization, queueing, network-wide congestion, throughput,
   loss, delay, or failure recovery time. The weighted baseline has more global
   graph access than the actors, which only receive local candidate features.

## 2. Corrections

- The shared training/inference action mask now also rejects a candidate unless
  the destination remains reachable after removing failed and already visited
  nodes. The route-distance and weighted-cost progress features are computed
  over that same remaining graph. This addresses the verified local-dead-end
  cause while retaining a topology-independent action space.
- Observation schema is `candidate-link-v2`: utilization, utilization-known,
  log capacity ratio, hop progress, failed flag, and weighted-cost progress.
  The weighted progress potential exposes per-candidate downstream cost under
  the same failure-aware graph and cost used by the congestion baseline.
- The congestion actor reward is now exactly negative per-hop cost
  `1 + 3 * background_utilization`, plus a fixed delivery bonus and dead-end
  penalty. This matches the weighted-Dijkstra baseline objective. No action
  can reduce offered demand. The failure actor has a separate per-hop penalty
  and delivery/dead-end reward, with no congestion term.
- Under failures, the rule-based Orchestrator combines the two masked policy
  distributions using equal log-probability weights. Without failures it uses
  the congestion actor. `coordination_disabled` preserves the old failure-first
  dispatch as an ablation. The Orchestrator remains rule-based, not learned.
- Training randomizes cross-link density between 0.15 and 0.45. Validation and
  test use separately seeded generated graphs, with cross-link probabilities
  0.25 and 0.35 respectively. The final test seeds were not used to choose
  checkpoints or tune weights.
- Raw training and evaluation now record reward/loss/entropy, delivery, action
  validity, loops, dead ends, hops, per-route background utilization, weighted
  route cost, fallback counts, and decision sources. The analyzer writes
  100-episode training windows and paired percentile-bootstrap intervals.

## 3. Training and evaluation protocol

Three congestion policies were trained for 1,000 episodes each with seeds
20262220–20262222; three failure policies used seeds 20262230–20262232. Each
episode performed one optimizer update. Training scenario mixtures were
normal/congestion for the congestion actor and failure/combined for the failure
actor. Output checkpoints and episode CSVs are in `runs/v4_train_seed*/`.
Training ran on CPU with Python 3.13.6, PyTorch 2.6.0+cu124, NumPy 2.2.6, and
NetworkX 3.5.

Validation used seeds 97100–97199 at cross-link probability 0.25. Final held-out
test used seeds 99000–99099 at cross-link probability 0.35. All nine policy and
ablation variants were evaluated on identical graph/traffic/failure seeds:

- shortest path and failure-aware shortest path;
- weighted congestion-aware Dijkstra baseline;
- congestion actor only and failure actor only;
- full Orchestrator;
- old failure-first coordination disabled;
- failure actor disabled; and
- congestion actor disabled.

On validation, all Orchestrator routes delivered (100/100 seeds per training
checkpoint), with zero invalid actions, loops, and dead ends. Mean selected
background utilization was 0.0744 vs 0.0710 for the weighted baseline in normal,
0.0919 vs 0.0909 in congestion, 0.0836 vs 0.0801 in failure, and 0.1099 vs
0.1077 in combined conditions. The validation set therefore did not suggest
baseline superiority was resolved; the policy was frozen and the separate
test set run without selecting a seed or changing its weight.

Each summary has 300 episodes per condition (100 held-out seeds × 3 trained
checkpoint pairs). Paired confidence intervals resample the 100 seed units
after averaging results across trained checkpoint pairs. They are descriptive
percentile-bootstrap intervals; comparisons are not adjusted for multiplicity.

## 4. Final held-out test results

All policies delivered 300/300 routes in every test condition. The Orchestrator
had zero invalid actions, fallbacks, loops, or dead ends. Entries below report
mean selected-route background utilization (standard deviation) over 300
test-run/checkpoint combinations; these are generated graph measurements.

| Condition | Shortest path | Weighted congestion baseline | MARL Orchestrator |
|---|---:|---:|---:|
| Normal | 0.155 (0.076) | 0.069 (0.031) | 0.073 (0.036) |
| Congestion | 0.702 (0.830) | 0.088 (0.042) | 0.089 (0.042) |
| Failure | 0.137 (0.070) | 0.075 (0.033) | 0.078 (0.036) |
| Combined | 0.574 (0.640) | 0.109 (0.056) | 0.122 (0.117) |

MARL minus weighted-baseline paired differences in utilization were:

| Condition | Mean difference | 95% paired bootstrap CI | Paired seeds |
|---|---:|---:|---:|
| Normal | +0.00423 | [+0.00221, +0.00675] | 100 |
| Congestion | +0.00059 | [-0.00112, +0.00192] | 100 |
| Failure | +0.00364 | [+0.00217, +0.00537] | 100 |
| Combined | +0.01316 | [+0.00371, +0.02494] | 100 |

The congestion actor is close to, but does not beat, weighted Dijkstra on the
congestion set. The full Orchestrator also remains slightly worse than that
baseline on all four utilization comparisons. Delivery, path safety, and
generalization diagnostics improved; superiority over the strongest baseline
is not supported.

The failure-first ablation is especially informative: on combined tests it
delivered 300/300 but selected links with mean utilization 0.838, compared with
0.122 for the full Orchestrator and 0.109 for weighted Dijkstra. The full
Orchestrator's difference from failure-first coordination was -0.716 (95% CI
[-0.816, -0.619]). Fusion therefore fixes the old coordination conflict, while
the non-RL baseline still has a small advantage.

## 5. Same-seed before/after diagnostic

For a direct comparison with the preserved previous version, v4 was also run on
the already-inspected seeds 93000–93099, cross-link probability 0.25, and the
same policy/reset protocol as the old runs. Because those historical test seeds
were previously inspected, this is **diagnostic only**, not confirmatory
evidence. Paired changes in the Orchestrator were:

| Condition | Delivery old → new | Utilization old → new | Paired 95% CI for utilization change |
|---|---:|---:|---:|
| Normal | 100% → 100% | 0.090 → 0.077 | [-0.01813, -0.00841] |
| Congestion | 98.3% → 100% | 0.150 → 0.095 | [-0.08885, -0.02559] |
| Failure | 100% → 100% | 0.127 → 0.088 | [-0.04773, -0.03095] |
| Combined | 99% → 100% | 0.837 → 0.117 | [-0.80599, -0.63623] |

This isolates the change on identical generated cases but reuses an old test
set. It must not be treated as the fresh held-out test result.

## 6. Artifacts, commands, and remaining limits

Raw final runs and analysis are in `runs/v4_test_*/` and
`runs/v4_test_analysis_final/`. Training curves are
`runs/v4_test_analysis_final/training_windows.csv`. Same-seed historical
diagnostics are in `runs/v4_before_after_reused93000/`.

Exact training commands (each produced 1,000 optimizer updates):

```bash
python -m marl_live.train --agent congestion --episodes 1000 --seed 20262220 --out marl_live/runs/v4_train_seed20262220/congestion
python -m marl_live.train --agent congestion --episodes 1000 --seed 20262221 --out marl_live/runs/v4_train_seed20262221/congestion
python -m marl_live.train --agent congestion --episodes 1000 --seed 20262222 --out marl_live/runs/v4_train_seed20262222/congestion
python -m marl_live.train --agent failure --episodes 1000 --seed 20262230 --out marl_live/runs/v4_train_seed20262230/failure
python -m marl_live.train --agent failure --episodes 1000 --seed 20262231 --out marl_live/runs/v4_train_seed20262231/failure
python -m marl_live.train --agent failure --episodes 1000 --seed 20262232 --out marl_live/runs/v4_train_seed20262232/failure
```

Validation and final test evaluations used the three corresponding checkpoint
pairs and these commands (the test examples shown explicitly for all three
training-seed pairs):

```bash
python -m marl_live.evaluate --congestion-policy marl_live/runs/v4_train_seed20262220/congestion/congestion_policy.pt --failure-policy marl_live/runs/v4_train_seed20262230/failure/failure_policy.pt --episodes 100 --seed 97100 --cross-link-probability 0.25 --out marl_live/runs/v4_validation_20_30
python -m marl_live.evaluate --congestion-policy marl_live/runs/v4_train_seed20262221/congestion/congestion_policy.pt --failure-policy marl_live/runs/v4_train_seed20262231/failure/failure_policy.pt --episodes 100 --seed 97100 --cross-link-probability 0.25 --out marl_live/runs/v4_validation_21_31
python -m marl_live.evaluate --congestion-policy marl_live/runs/v4_train_seed20262222/congestion/congestion_policy.pt --failure-policy marl_live/runs/v4_train_seed20262232/failure/failure_policy.pt --episodes 100 --seed 97100 --cross-link-probability 0.25 --out marl_live/runs/v4_validation_22_32
python -m marl_live.evaluate --congestion-policy marl_live/runs/v4_train_seed20262220/congestion/congestion_policy.pt --failure-policy marl_live/runs/v4_train_seed20262230/failure/failure_policy.pt --episodes 100 --seed 99000 --cross-link-probability 0.35 --out marl_live/runs/v4_test_20_30
python -m marl_live.evaluate --congestion-policy marl_live/runs/v4_train_seed20262221/congestion/congestion_policy.pt --failure-policy marl_live/runs/v4_train_seed20262231/failure/failure_policy.pt --episodes 100 --seed 99000 --cross-link-probability 0.35 --out marl_live/runs/v4_test_21_31
python -m marl_live.evaluate --congestion-policy marl_live/runs/v4_train_seed20262222/congestion/congestion_policy.pt --failure-policy marl_live/runs/v4_train_seed20262232/failure/failure_policy.pt --episodes 100 --seed 99000 --cross-link-probability 0.35 --out marl_live/runs/v4_test_22_32
```

The final aggregate command was:

```bash
python -m marl_live.analyze --runs marl_live/runs/v4_test_20_30 marl_live/runs/v4_test_21_31 marl_live/runs/v4_test_22_32 --training-runs marl_live/runs/v4_train_seed20262220/congestion marl_live/runs/v4_train_seed20262221/congestion marl_live/runs/v4_train_seed20262222/congestion marl_live/runs/v4_train_seed20262230/failure marl_live/runs/v4_train_seed20262231/failure marl_live/runs/v4_train_seed20262232/failure --out marl_live/runs/v4_test_analysis_final --bootstrap 5000 --seed 607
```

Reproduction commands and the unchanged live-runtime blocker are documented in
`README.md`. No live controller or WSL dependency was changed. Throughput,
delay, packet loss, actual recovery time, and Mininet forwarding remain
unverified. The graph family is still a small eight-switch parallel-route
environment with one OD pair and one pre-routing link failure, so transfer to
larger or dynamic networks remains an open problem.
