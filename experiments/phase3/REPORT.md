# Phase 3: Transfer Diagnosis and Bidirectional Graph-State Correction

Date: 2026-10-20 (runs completed after the Phase 2 baseline; execution environment clock is recorded in raw run metadata). Baseline commit: `53d9a4a`. Phase 3 outputs are under `results/phase3_eval/`, `results/phase3_train_congestion_seed20261021/`, and `results/phase3_train_failure_seed20261022/`. No prior result or checkpoint was overwritten.

## Phase 2 evidence audit

The Phase 2 raw rows are in `results/phase2_final/` and the Cogentco/Iris transfer rows are in their corresponding `results/phase2_final_*` directories. The report and paired analysis are in `experiments/phase2/REPORT.md` and `results/phase2_final/statistics/paired_comparisons.csv`. The Abilene final comparison used 20 matched seeds, six conditions, six policies, and 720 rows. Transfer used 10 seeds per topology and 360 rows each. Analysis paired policy differences by seed/condition and applied Holm correction across the comparison family.

The evidence remains negative or limited: at 8x generated heavy traffic the Phase 2 congestion policy reduced modeled path loss and delay descriptively but did not change congested-link ratio; none of those comparisons survived Holm correction. Its two-link failure recovery matched immediate shortest-path delivery but acted one control step later. Cogentco had no policy effect and Iris had intermittent interventions without consistent gain. The rule-based Orchestrator did not establish coordination value; in combined Abilene stress its failure override suppressed congestion proposals, and the no-override ablation's apparent gains were not significant after correction. CAIDA was graph-only, not a QoS evaluation.

The saved Phase 2 training configuration/history is inconsistent with the checked-in trainer's default topology list: the two priority checkpoints' histories show Abilene episodes, while the current default list is a set of 18 other topologies. The Phase 3 configuration now saves the explicit topology and traffic-scale lists used.

## Diagnosed transfer limitation

`SDNEnvironment._build_state` represented each NetworkX undirected physical link with one directed `edge_index` message edge. The GAT encoder uses directed source-to-target message passing. Measurement on Abilene, Iris, and Cogentco confirmed zero reverse message edges (14/14, 64/64, and 243/243 physical links respectively). Which endpoint sent a message was therefore determined by NetworkX edge iteration / node ordering, although the physical link and its measurement are bidirectional. This is a concrete state-encoding defect and plausible source of transfer instability. It is not proven to be the only cause.

Other inspected causes/limits:

* No absolute node ID or fixed graph dimension enters the feature tensors. Q heads produce three global actions, not node-indexed actions; action numbering is consistent across graphs. The distance-constraint policy from the separate supervised route evaluator is destination-specific and is not used by these global DQN policies.
* Per-graph normalized degree and coordinates are relative features; topology-global mean pooling can dilute local congestion signals. Raw edge capacity/delay and offered demand remain unnormalized. Abilene's 14 link capacities are read as 9,953.28 Mbps OC-192 values; Cogentco/Iris inputs use the loader's named 1,000 Mbps default capacity. These feature regimes differ materially. The loader records default sources, so missing measurements are not silently reported as observed capacity.
* The action mask permits reroute/rate-limit only when an observed simulator edge reaches 0.80 utilization. Consequently, on states with no threshold-level congestion, maintain is the only valid policy choice; this is desirable and explains no-op results in some conditions. Under congestion, all three actions are available.
* Replay buffers are not saved in checkpoints. Historical replay diversity and training-time Q-value distributions cannot be reconstructed from saved artifacts. The per-episode training histories show only Abilene in Phase 2; Phase 2 evaluation covers separate topologies.
* Traffic, capacities, utilization, loss, and delay are simulator-generated. CAIDA lacks these states and was not used to create them.

The first justified correction was therefore to serialize every undirected link as two directed message edges with duplicated, identical five-feature measurements. The feature semantics/dimensions and DQN action space stayed unchanged. This preserves checkpoint load compatibility, but old weights were optimized against the one-direction message graph and should not be treated as equivalent models after the fix.

## Phase 3 training

The existing `training/train.py` gained explicit `--topologies` and episode-rotated `--traffic-scales`; defaults preserve the existing behavior. Two single-agent runs used the same three connected training graphs (UsCarrier, Colt, GtsCe), heavy generated traffic, 12 demands, 12 steps, batch 16, CUDA, and traffic scales 2x/8x. The Congestion DQN had no injected failures; the Failure DQN injected one active link per episode. No validation topology was used for checkpoint selection. Held-out Abilene, Cogentco, and Iris were evaluated only after both runs completed.

| Agent | Seed | Episodes | Environment steps | Optimizer updates | Runtime | Mean episode reward | Training actions [0,1,2] | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Congestion DQN | 20261021 | 48 | 576 | 561 | 135.68 s | -3.9898 | [217,184,175] | `results/phase3_train_congestion_seed20261021/congestion_dqn_final.pt` |
| Failure DQN | 20261022 | 48 | 576 | 561 | 156.72 s | -3.4224 | [242,185,149] | `results/phase3_train_failure_seed20261022/failure_dqn_final.pt` |

Configs, per-episode reward/loss/action histories, summaries, and checkpoints are retained beside each checkpoint. A loss/update count is not evidence of a useful policy. Both checkpoints were loaded by the fresh-process evaluator; inference probes are recorded in the raw JSON files.

Reproduction commands (run from repository root):

```powershell
python training/train.py --episodes 48 --max-steps 12 --num-demands 12 --batch-size 16 --seed 20261021 --scenario heavy --traffic-scales 2 8 --failure-probability 0 --train-agent congestion --topologies UsCarrier.gml Colt.gml GtsCe.gml --output-dir results/phase3_train_congestion_seed20261021
python training/train.py --episodes 48 --max-steps 12 --num-demands 12 --batch-size 16 --seed 20261022 --scenario heavy --traffic-scales 2 8 --failure-probability 1 --train-agent failure --topologies UsCarrier.gml Colt.gml GtsCe.gml --output-dir results/phase3_train_failure_seed20261022
python training/evaluate_dqn.py --congestion-checkpoint results/phase3_train_congestion_seed20261021/congestion_dqn_final.pt --failure-checkpoint results/phase3_train_failure_seed20261022/failure_dqn_final.pt --topology InternetTopologyZoo/gml/Cogentco.gml --seeds 9701 9702 9703 9704 9705 --demands 8 --traffic-scale 8 --output results/phase3_eval/failure_new_cogentco/evaluation.json
python training/analyze_phase2.py results/phase3_eval/failure_new_cogentco/evaluation.json --output-dir results/phase3_eval/failure_new_cogentco/statistics --bootstrap-samples 5000 --seed 20261023
python -m pytest -q
python -m compileall -q agents models simulator training scalability evaluation tests
git diff --check
```

## Held-out transfer results

The fresh-process evaluator ran five paired seeds (9701–9705), eight demands, and all six existing conditions/policies on Abilene, Cogentco, and Iris. Each run starts a fresh simulator. All link-state values below are generated by the simplified simulator. New and old checkpoints were evaluated under the corrected bidirectional encoder; the old checkpoint comparison is therefore a compatibility diagnostic, not an exact replication of its former input semantics.

| Topology | Congestion model | 8x heavy congested-link ratio | Offered shortfall | Mean delay (ms) | Congestion actions over 40 decisions |
|---|---|---:|---:|---:|---|
| Abilene | Phase 2 checkpoint | 0.100 | 0.178 | 50.738 | see raw CSV |
| Abilene | Phase 3 checkpoint | 0.100 | 0.194 | 53.198 | [40,0,0] |
| Cogentco | Phase 2 checkpoint | 0.272 | 0.995 | 1,125.0 | [40,0,0] |
| Cogentco | Phase 3 checkpoint | 0.272 | 0.995 | 1,125.0 | [40,0,0] |
| Iris | Phase 2 checkpoint | 0.447 | 0.973 | 550.0 | [0,40,0] |
| Iris | Phase 3 checkpoint | 0.406 | 0.969 | 487.5 | [40,0,0] |

The new model did not learn to intervene on held-out Cogentco or Iris stress runs, and on Abilene it had slightly higher offered shortfall and delay than the old checkpoint. Iris's lower synthetic congestion ratio versus the old model is confounded by the new model taking no action and by run seeds; the no-agent/shortest baselines are available in the same raw files and must be used for proper policy effects. This run does not establish a reliable improvement. Per-condition paired statistical comparisons and confidence intervals are retained in the raw per-seed rows; only five seeds were used here, so they are exploratory. No checkpoint was selected on these final tests.

Paired comparisons of the Phase 3 Congestion DQN against shortest path within each new evaluation had exactly zero difference in congested-link ratio, offered shortfall, throughput, and delay on all three topologies (n=5, sign-flip p=1.0, Holm p=1.0); the learned action was maintain for all 40 high-load decisions on Abilene, Cogentco, and Iris. Thus the only positive case in the Phase 3 evaluation is equivalence to the baseline, not learned congestion control. The apparent cross-checkpoint Iris difference in the table is not a paired policy effect and is not interpreted causally.

## Phase 3 failure check and coordination

The new multi-topology Failure DQN checkpoint reloaded and ran on five matched held-out seeds (9701–9705), with one- and two-link failures as well as combined stress. For link-failure-only conditions, Cogentco delivery was 0.825 for both immediate shortest path and DQN; mean recovery was 0 versus 1 step and unserved-demand steps were 12.6 versus 14.4. With two failures, delivery was 0.775 for both and recovery remained 0 versus 1. On Iris, single-failure delivery was 0.850 for both; mean DQN recovery was 1.5 steps and unserved-demand steps 15.6 versus shortest path's 0 steps and 10.8. For two failures, delivery was 0.775 for both; DQN recovery averaged 1.5 steps versus 0. The combined-stress DQN did not match shortest path on every topology: Cogentco delivery was 0.775 versus 0.825 and Iris was 0.575 versus 0.850. Orchestrator override restored Iris combined delivery to 0.850, but it remained a rule baseline and no broad coordination benefit is established. Full per-seed rows, summaries, and paired Phase 2-style statistical outputs are in each `failure_new_*` evaluation directory. These five-seed samples are exploratory; they do not establish better recovery.

For DQN minus shortest path, paired differences in unserved-demand steps were +1.8 (95% paired bootstrap CI [0.6, 3.0], Cohen's dz 1.21, sign-flip p=.124, Holm p=1.0) for Cogentco single failure and +4.8 ([3.2, 6.6], dz 2.21, p=.063, Holm p=1.0) for Iris single failure. The two-failure differences were +1.6 ([0.6,3.0], dz 1.06, p=.128, Holm 1.0) on Cogentco and +4.6 ([2.8,6.6], dz 1.91, p=.065, Holm 1.0) on Iris. The combined-stress delivery-rate differences were -0.050 ([−0.150, 0.000], dz −0.45, p=1.0, Holm 1.0) on Cogentco and -0.275 ([−0.375, −0.125], dz −1.69, p=.127, Holm 1.0) on Iris. The test family is corrected for multiplicity; with five seeds, the estimates are imprecise and none is statistically supported after correction.

The repository's separate simulator dataset is indeed under `scalability/`: `scalability/train/` contains archived runs for 25–50-node cases and `scalability/test/` contains cases from 50 through 300 nodes. These are not the CAIDA graph input and the DQN trainer did not consume these archives in Phase 3. CAIDA's Matrix Market file remains under `tech-RL-caida/`. Phase 3 used the existing Topology Zoo + NetworkSimulator workflow for DQN training/evaluation and preserved the archived scalability data; it did not treat archived trace fields as interchangeable with the DQN observation without a validated adapter.

## Failure and coordination

Phase 2 already tested one and two active-link failures across 20 Abilene seeds and 10 seeds each on Cogentco/Iris, under normal failure and combined load/failure. Failure DQN recovered the same final route fraction as immediate shortest path where recovery was possible, but at one control step versus the reference's immediate action. The rule orchestrator and no-override ablation did not demonstrate significant coordination benefit. The Phase 3 shared-edge correction justified retraining the Failure DQN with the same multi-topology regime; its held-out failure outcomes are saved in `results/phase3_eval/failure_new_{cogentco,iris}/`. This model still cannot reduce the immediate baseline's zero-step decision latency. Multiple simultaneous failures and detection wall-time remain limited by the simulator: failures are exposed before the first policy action and recovery is measured in control steps.

## CAIDA and limitations

The full graph-only CAIDA benchmark remains preserved from Phase 2 (190,914 nodes, 607,610 unique edges, full-graph 100-pair sample). No additional CAIDA evaluation was run because no new graph algorithm/scalability question was identified; there is no compatible link-state data for learned QoS comparisons. No congestion, throughput, delay, or failure claim is made for CAIDA.

The simulator does not model packet traces, queues as time series, real loss counters, wall-clock failure detection, or measured CAIDA capacities. Its default topology capacities can make transferred utilization distributions extreme, and the reward/action system still has a graph-level global Q output. Training on three additional graphs fixes neither the capacity-domain mismatch nor the underlying simulator's realism. The evidence supports correcting the undirected message representation, but does not support claiming a defensible congestion-control gain.

## Verification

The targeted regression tests cover symmetric edge serialization, identical feature values in both directions, variable graph sizes, and valid DQN inference/action outputs. Full test suite, syntax compile, and `git diff --check` results are recorded in `PROJECT_STATUS.md` after Phase 3 completion.
