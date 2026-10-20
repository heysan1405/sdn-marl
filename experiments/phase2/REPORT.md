# Phase 2 reproducibility and generalization report

Date: 2026-10-20
Repository base commit: `1cd5aab7f7f69af1b0280bb785d5c8bcc7910fa6`
This phase reuses the previously trained checkpoints. It does not overwrite or retrain them.

## Baseline audit

The previous five-seed data exists at `results/priority_final_eval_seed9151_9152/evaluation_with_runtime.json` and its CSV companion. It contains 100 runs: four conditions, five policies, and seeds 941–945. The final training summaries record 40 episodes, 480 environment steps, and 465 optimizer updates for each separately trained DQN. Their exact configs, histories, and checkpoints remain in `results/priority_final_congestion_seed9151/` and `results/priority_final_failure_seed9152/`.

Checkpoint SHA-256 values:

| Checkpoint | SHA-256 |
|---|---|
| `results/priority_final_congestion_seed9151/congestion_dqn_final.pt` | `4ce558f4ff227004f3ed5626a6b36a263fb0be484208fd4c421b1b59f21550a1` |
| `results/priority_final_failure_seed9152/failure_dqn_final.pt` | `efb3fc0031a2779af7b864485a2ca9c11f6107417e0832d95a3beeb297d39dfa` |

The prior table is descriptive only. Its five runs are too few to establish generalization or statistical support. The old checkpoint files were loaded successfully by each new evaluator process; each normal-state inference probe returned action 0 for both agents.

## New experiment design

The main Abilene experiment uses 20 fixed seeds (`9301`–`9320`), eight demands per fresh environment, and six conditions. Every policy is evaluated from a reset simulator on the same seed and condition. The policy set is no-agent, immediate shortest path, Congestion DQN alone, Failure DQN alone, the rule-based orchestrator, and both agents without orchestrator overrides. The two single-agent policies are ablations of the other agent; the final policy isolates the orchestrator override rule.

The per-run `runtime_seconds` values sum to 54.71 seconds for the 720 Abilene runs, 103.19 seconds for 360 Cogentco runs, and 56.14 seconds for 360 Iris runs. These sums cover environment creation and episode execution, but omit one-time process startup and analysis time.

Conditions are normal (1x normal generated load), 2x generated heavy load, 8x generated heavy load, one active link failure, two distinct active link failures, and 8x heavy load plus one failure. Failures target the two links carrying the most active flows before injection; this makes failures reproducible, but does not sample failures uniformly. Failures are exposed before the first policy action. The simulator therefore cannot measure detection latency. It models route usability and estimated load, not packet-level time series.

The evaluator writes one CSV row per run and a JSON file containing raw runs, summaries, the seed list, conditions, and checkpoint load metadata. Statistical analysis uses paired percentile bootstrap intervals and paired sign-flip randomization tests; Holm adjustment covers all emitted policy/scenario/metric comparisons. Sign-flip inference assumes symmetric paired differences. With 20 seeds and 846 comparisons, corrected tests are conservative; intervals and p-values should be read as exploratory evidence, not a substitute for independent scenario generation.

### Metric definitions and availability

| Field | Definition / unit | Limitation |
|---|---|---|
| `delivery_rate` | Fraction of original demands with a currently usable route | A route-availability fraction, not packet delivery ratio |
| `throughput_mbps` | Simulator-estimated successfully delivered traffic | Based on assumed/loaded capacities and the simulator's path-loss model |
| `packet_loss_ratio_offered` | `max(0, 1 - throughput / original_offered_Mbps)` | Offered-demand shortfall; includes rate-limited demand and unreachable flows, so it prevents those from appearing as free wins. It is not a packet counter. |
| `packet_loss` | Mean simulator-estimated path loss across demands, with unusable demands counted as loss 1 | Equal weight per flow, not weighted by original offered Mbps |
| `mean_link_utilization`, `max_link_utilization`, `p95_link_utilization` | Current simulator utilization, unitless ratio, over healthy links | Synthetic simulation state, not CAIDA data |
| `congestion_ratio` | Fraction of healthy links at/above the simulator's 0.80 utilization threshold | Does not describe queue occupancy or time spent congested |
| `mean_delay_ms` | Mean estimated path delay for usable paths | Model-derived; jitter is not available |
| `mean_hops`, `loops`, `failed_demands` | Final usable paths only; hops are links traversed | Failed paths are counted separately, and no loops were observed |
| `recovery_time_steps`, `unserved_demand_steps` | Control steps until all demands have usable paths; count of unserved demands accumulated per step | Not wall-clock outage time; unreachable demands can remain unrecovered (`null`) |
| `congestion_actions`, `failure_actions`, `agent_interventions`, `orchestrator_overrides` | Per-run action counts / override count | Intervention actions are policy choices, not confirmed packet-level changes |
| `dropped_packets`, `jitter_ms`, `failure_detection_time_seconds`, `peak_memory_bytes` | `null` | Simulator does not expose packet counters, packet timestamps, detection clock, or isolated process peak memory |

The simulator estimates packet loss as a function of link utilization and estimates throughput from demand, bottleneck capacity, and modeled path loss. These are synthetic simulation results. They are not captured packet traces or measurements from Internet topology datasets.

## Main results: Abilene

All values below are means over 20 matched seeds. Delivery is the route-availability fraction. Offered shortfall uses original offered traffic; modeled path loss is a separate flow-average estimate.

| Condition | Policy | Delivery | Offered shortfall | Modeled path loss | Congested-link ratio | Delay (ms) | Recovery (steps, successful recoveries) |
|---|---|---:|---:|---:|---:|---:|---:|
| Normal | All policies | 1.000 | 0.000 | 0.000 | 0.000 | 2.293 | n/a |
| 2x load | All policies | 1.000 | 0.000 | 0.000 | 0.000 | 2.866 | n/a |
| 8x load | No-agent / shortest path | 1.000 | 0.218 | 0.216 | 0.118 | 51.755 | n/a |
| 8x load | Congestion DQN / orchestrated / no-override | 1.000 | 0.202 | 0.177 | 0.118 | 40.995 | n/a |
| One link failure | No-agent | 0.569 | 0.430 | 0.431 | 0.000 | 2.169 | unrecovered |
| One link failure | Shortest path | 0.969 | 0.032 | 0.031 | 0.000 | 3.175 | 0 (acts immediately) |
| One link failure | Failure DQN / both orchestrators | 0.969 | 0.032 | 0.031 | 0.000 | 3.178 | 1 step when recovered |
| Two link failures | No-agent | 0.356 | 0.643 | 0.644 | 0.000 | 1.889 | unrecovered |
| Two link failures | Shortest path | 0.825 | 0.171 | 0.175 | 0.000 | 3.285 | 0 (acts immediately) |
| Two link failures | Failure DQN / orchestrated | 0.825 | 0.171 | 0.175 | 0.000 | 3.285 | 1 step when recovered |
| Combined | Shortest path | 0.969 | 0.448 | 0.450 | 0.219 | 111.253 | 0 (acts immediately) |
| Combined | Failure DQN / orchestrated | 0.969 | 0.455 | 0.457 | 0.219 | 114.225 | 1 step when recovered |
| Combined | No-override ablation | 0.969 | 0.426 | 0.342 | 0.158 | 77.301 | 1 step when recovered |

The twofold load on Abilene did not cross the 0.80 utilization threshold; it is a lower-intensity control, not a realized congestion condition. At 8x load, the congestion policy chose rate limiting 14 times across 160 control decisions and never chose rerouting. Its paired mean differences versus shortest path were −0.039 for modeled path loss (95% paired bootstrap CI approximately [−0.103, −0.002]) and −10.76 ms for delay (CI approximately [−29.43, −0.29]); offered shortfall changed by −0.016 (CI [−0.047, 0.001]). None of these comparisons remained statistically supported after Holm correction across the complete comparison family (adjusted p=1.0 for these rows). Congestion ratio did not change. This is a small simulator-only effect that may partly reflect the simulator's rate-limiting behavior; it is not evidence of broad congestion control.

The Failure DQN recovers the same final route fraction as immediate shortest path in these failure conditions, but takes one control step after the failure is shown. It is therefore slower. A subset of demands remain unreachable after both policies, consistent with failures that remove available connectivity or exceed the simulator's candidate-path alternatives. No wall-clock recovery speedup is supported.

The rule orchestrator is deterministic and rule-based. In combined Abilene runs, its failure-priority override suppresses congestion actions; the no-override ablation has lower descriptive modeled loss and delay (0.342 and 77.3 ms versus 0.457 and 114.2 ms). This difference is exploratory and does not survive the all-comparisons Holm adjustment. It suggests the override may discard useful congestion actions during combined stress, and coordination has not been shown to add value.

## Topology transfer

The loaded Topology Zoo graphs were connected. Statistics use the repository's duplicate-tolerant GML loader.

| Topology | Nodes | Edges | Degree min / mean / max | Seeds | Heavy-load Congestion DQN result |
|---|---:|---:|---:|---:|---|
| Abilene (training topology for these checkpoints) | 11 | 14 | 2 / 2.545 / 3 | 20 | Some rate limiting; modest descriptive changes |
| Iris | 51 | 64 | 1 / 2.510 / 6 | 10 | Mostly no-op at 2x; rate limiting at 8x without measurable improvement |
| Cogentco | 197 | 243 | 1 / 2.467 / 9 | 10 | No-op in all heavy-load decisions; no policy metric changes |

The checkpoint tensor interfaces handled both unseen graphs, but performance did not transfer reliably. Cogentco also reveals an important limitation of this simulator scenario: estimated offered shortfall was 0.788 at 2x and 0.993 at 8x for every policy. The fixed generated demands and assumed topology capacities create extreme load, and the DQN did not correct it. The failure model matched immediate shortest-path final route availability on these sampled failures, with one extra control step. These are small topology-specific samples, not a broad generalization claim.

Training history indicates the Congestion DQN had maintain/reroute/rate-limit training counts `[186, 82, 212]`, but its held-out evaluation selected no reroute actions. The final checkpoints were trained for 40 episodes only on Abilene. The observed topology transfer failure is real; there is not yet a validated reward or model defect that would justify another training run or checkpoint selection. No retraining was done in this phase.

## CAIDA topology-only scalability

CAIDA was evaluated independently from the traffic simulator. The reused Matrix Market file parses as symmetric coordinate pattern data with 190,914 nodes, 607,610 unique undirected edges, one connected component, zero duplicate entries, and zero self-loops. For each size, deterministic seed 314 selects a connected breadth-first induced subgraph; 1,000/2,000/5,000/10,000-node experiments were rerun into `results/phase2_multiseed/caida_reproduced/`. The full graph was sampled at 100 pairs. Routing uses destination BFS / strictly distance-decreasing forwarding, never an all-pairs matrix.

| Graph | Nodes | Edges | Connected | Degree min / mean / max | Pairs | Delivered | Unreachable | Loops / dead ends | Mean hops | Runtime (s) | Peak RSS (MiB) |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Induced | 1,000 | 5,931 | Yes | 1 / 11.862 / 298 | 500 | 500 | 0 | 0 / 0 | 3.034 | 2.84 | 282.3 |
| Induced | 2,000 | 14,366 | Yes | 1 / 14.366 / 299 | 500 | 500 | 0 | 0 / 0 | 3.072 | 3.20 | 282.5 |
| Induced | 5,000 | 40,105 | Yes | 1 / 16.042 / 299 | 300 | 300 | 0 | 0 / 0 | 3.470 | 3.72 | 282.4 |
| Induced | 10,000 | 85,055 | Yes | 1 / 17.011 / 302 | 200 | 200 | 0 | 0 / 0 | 3.695 | 4.81 | 282.7 |
| Full graph | 190,914 | 607,610 | Yes | 1 / 6.365 / 1,071 | 100 | 100 | 0 | 0 / 0 | 6.880 | 27.90 | 286.8 |

These results say nothing about CAIDA congestion or learned-policy quality. CAIDA has no traffic, capacity, queue, failure, utilization, or packet-loss state. Learned DQN and supervised link-feature policies were not evaluated on it.

## Reproduction commands

Run from repository root with the project's Python dependencies installed. The old five-seed command and outputs remain in `PROJECT_STATUS.md`; the following commands reproduce the Phase 2 raw runs and analysis:

```powershell
$runId = Get-Date -Format yyyyMMdd_HHmmss
$runRoot = "results/phase2_reproduction_$runId"
python training/evaluate_dqn.py --congestion-checkpoint results/priority_final_congestion_seed9151/congestion_dqn_final.pt --failure-checkpoint results/priority_final_failure_seed9152/failure_dqn_final.pt --seeds 9301 9302 9303 9304 9305 9306 9307 9308 9309 9310 9311 9312 9313 9314 9315 9316 9317 9318 9319 9320 --demands 8 --traffic-scale 8 --output "$runRoot/abilene/evaluation.json"
python training/analyze_phase2.py "$runRoot/abilene/evaluation.json" --output-dir "$runRoot/abilene/statistics" --bootstrap-samples 20000 --seed 20261020
python training/evaluate_dqn.py --congestion-checkpoint results/priority_final_congestion_seed9151/congestion_dqn_final.pt --failure-checkpoint results/priority_final_failure_seed9152/failure_dqn_final.pt --topology InternetTopologyZoo/gml/Cogentco.gml --seeds 9401 9402 9403 9404 9405 9406 9407 9408 9409 9410 --demands 8 --traffic-scale 8 --output "$runRoot/cogentco/evaluation.json"
python training/analyze_phase2.py "$runRoot/cogentco/evaluation.json" --output-dir "$runRoot/cogentco/statistics" --bootstrap-samples 20000 --seed 20261020
python training/evaluate_dqn.py --congestion-checkpoint results/priority_final_congestion_seed9151/congestion_dqn_final.pt --failure-checkpoint results/priority_final_failure_seed9152/failure_dqn_final.pt --topology InternetTopologyZoo/gml/Iris.gml --seeds 9501 9502 9503 9504 9505 9506 9507 9508 9509 9510 --demands 8 --traffic-scale 8 --output "$runRoot/iris/evaluation.json"
python training/analyze_phase2.py "$runRoot/iris/evaluation.json" --output-dir "$runRoot/iris/statistics" --bootstrap-samples 20000 --seed 20261020
python scalability/evaluate_large_topology.py --size 1000 --seed 314 --pairs 500 --output-dir "$runRoot/caida"
python scalability/evaluate_large_topology.py --size 2000 --seed 314 --pairs 500 --output-dir "$runRoot/caida"
python scalability/evaluate_large_topology.py --size 5000 --seed 314 --pairs 300 --output-dir "$runRoot/caida"
python scalability/evaluate_large_topology.py --size 10000 --seed 314 --pairs 200 --output-dir "$runRoot/caida"
python scalability/evaluate_large_topology.py --full-graph --seed 314 --pairs 100 --output-dir "$runRoot/caida"
python -m pytest -q
python -m compileall -q agents simulator training scalability evaluation tests
git diff --check
```

Raw outcomes and console logs are saved in `results/phase2_final/`, `results/phase2_final_cogentco/`, `results/phase2_final_iris/`, and `results/phase2_multiseed/`. Existing `priority_*`, `large_topology`, checkpoint, and prior training artifacts were left intact.

## Conclusions and remaining work

- The corrected checkpoint evaluation confirms a narrow, modest heavy-load effect on Abilene with no congestion-ratio improvement; it is not statistically supported after multiple-comparison correction.
- Failure DQN improves route availability over no-agent for injected active-link failures, but matches rather than beats immediate shortest path and has a one-step control delay.
- The rule orchestrator remains a deterministic baseline. Its failure override can suppress potentially useful congestion actions; the ablation is suggestive but not statistically established.
- Transfer to Iris and Cogentco is weak or absent. More diverse training/validation scenarios and a clearer simulator objective are needed before retraining and selecting a new model.
- CAIDA graph-scale routing succeeded through the full graph at 100 sampled pairs. No traffic-performance claim is supported for CAIDA.
- The simulator cannot provide packet counts, packet-level jitter, wall-clock failure detection/recovery, or trustworthy real-network performance. It uses generated demands, assumptions/defaults for missing link parameters, and simplified load/loss calculations.
