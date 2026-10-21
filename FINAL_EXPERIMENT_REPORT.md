# Final Experiment Report

Date: 2026-10-20

This report records the dataset-based follow-up after the Phase 2 and Phase 3 reports. Earlier simulator experiments and checkpoint outcomes remain in [experiments/phase2/REPORT.md](experiments/phase2/REPORT.md) and [experiments/phase3/REPORT.md](experiments/phase3/REPORT.md); no earlier checkpoints or results were overwritten.

## Dataset audit

The dataset is under `scalability/`, separate from the CAIDA graph in `tech-RL-caida/`. The `train/` tree contains 800 archives at each of 25, 30, 35, 40, 45, and 50 nodes (4,800 archives total). `test/` contains archives across 50–300 nodes, with a varying number per size. A training archive normally contains 25 simulation records; the selected test archives each contained one record.

Each archive includes `input_files.txt`, `traffic.txt`, `linkUsage.txt`, `simulationResults.txt`, and `stability.txt`. The input rows associate a run ID with a graph and routing file. Traffic rows describe per-source/destination traffic generator settings and parameters, with a global intensity field. `simulationResults.txt` stores aggregate packet/loss/delay statistics and per-pair output. `stability.txt` records run duration/intensity/status and simulator metadata. Selected runs were `OK`; the parser now preserves this status and the sample builder refuses non-OK records.

The routing files provide source-to-destination paths. The names are typically `Routing-...-SP-k-...`; these are labels for the supplied routing configuration, not labels for congestion-optimal decisions. This supports supervised imitation of the archived route's next hop. `linkUsage.txt` provides measured simulator link and queue features as a flattened row-major N-by-N matrix. The correct index for directed arc `(u,v)` is `u*N+v`; non-links use `-1`. Each measured link row supplies link utilization/loss/average packet size and queue utilization/loss/average port occupancy/maximum queue occupancy/average packet size. Adding bandwidth, weight, and distance-to-destination yields the established 11-feature input.

The previous builder assigned the compact list of link measurements to `NetworkX.edges()` order. That order differs from the row-major matrix: in the audited 25-node example, the first graph arc was `0->24`, but the first measured matrix entry was `0->2`. The wrong utilization and queue measurements were therefore associated with edges. The new shared mapper uses node-pair matrix offsets and tests cover the ordering mismatch.

The archives do **not** record DQN-selected actions, action-conditioned next states, rewards, or injected link failures. Although aggregate simulator outcomes exist, they are not linked into DQN transitions. Therefore neither Congestion DQN nor Failure DQN has valid offline-RL experience in these archives. No actions, rewards, transitions, labels, congestion values, or failure events were invented.

## Dataset-trained model

The strongest supported training target is the existing supervised 11-feature candidate-link scorer. It was trained from scratch with cross-entropy to imitate next hops in archived routing files. The congestion penalty from the older training script was intentionally excluded because the route labels do not identify a congestion-optimal action. This run is a supervised routing scorer, not either DQN agent.

The trainer selected 4 train archives and 1 validation archive per size (25–50 nodes), held out by archive, then used simulation records 0–2 from each selected archive. This yielded 72 training records and 18 validation records. The test split was not opened by the trainer or used for checkpoint selection. The seed was 31415; batch size 128; learning rate 0.001; five epochs; CUDA. The selected epoch was 4 by minimum validation cross-entropy.

| Measure | Result |
|---|---:|
| Training next-hop decisions per epoch | 251,756 |
| Validation decisions per epoch | 61,952 |
| Optimizer updates | 10,045 |
| Runtime | 144.99 s |
| Best validation cross-entropy | 0.16096 |
| Best validation top-1 next-hop accuracy | 92.66% |
| Checkpoint | `results/dataset_training_supervised_seed31415/best_validation_model.pt` |

The split manifest, normalization and feature order, config, per-epoch train/validation curves, and summary are in `results/dataset_training_supervised_seed31415/`. The Phase 3 checkpoints were not used as training inputs.

Commands:

```powershell
python training/train_dataset_routing.py --epochs 5 --batch-size 128 --train-archives-per-size 4 --validation-archives-per-size 1 --simulations-per-archive 3 --seed 31415 --output-dir results/dataset_training_supervised_seed31415
python training/evaluate_dataset_routing.py --checkpoint results/dataset_training_supervised_seed31415/best_validation_model.pt --baseline-checkpoint models/congestion_agent.pt --sizes 50 100 200 300 --archives-per-size 2 --simulations-per-archive 1 --pairs-per-simulation 150 --seed 31416 --output-dir results/dataset_training_eval_seed31416
python -m pytest -q
python -m compileall -q agents models simulator training scalability evaluation tests
git diff --check
```

## Held-out evaluation

After checkpoint selection, the evaluator loaded eight test archives (two each at 50, 100, 200, and 300 nodes), sampling 150 routed source/destination pairs per archive: 1,200 matched route rollouts per policy. Next-hop accuracy was evaluated over all candidate decisions in those eight records (1,001,736 decisions). The evaluation used the Phase 2 supervised checkpoint as a baseline, the archived route as a reference, the new scorer, and the new scorer with the destination-specific distance-decreasing constraint.

| Test nodes | Candidate next-hop accuracy | Baseline delivery | New scorer delivery | New scorer loops | Distance-constrained delivery / loops |
|---:|---:|---:|---:|---:|---:|
| 50 | 93.79% | 100% | 100% | 0 | 100% / 0 |
| 100 | 94.19% | 100% | 100% | 0 | 100% / 0 |
| 200 | 91.44% | 100% | 86.33% | 41 | 100% / 0 |
| 300 | 93.19% | 100% | 96.33% | 11 | 100% / 0 |
| Overall | 92.72% | 100% | 95.67% | 52 / 1,200 pairs | 100% / 0 |

The constrained policy restricts each selected next hop to strictly decrease shortest-path distance to that specific destination. It restored delivery and eliminated loops in these samples, at the same mean hop count as the archived reference. The unconstrained learned scorer introduced loops at 200 and 300 nodes.

The newly scored paths had mean recorded congestion 0.2402 versus 0.2309 for the existing baseline over these samples. These values are features measured in the archive's original simulation. Routing changes were not re-simulated, so this is not a counterfactual congestion comparison and does not demonstrate a QoS improvement. The defensible result is supervised label imitation with roughly 92.7% held-out next-hop accuracy, plus evidence that unconstrained rollout can loop and the destination-specific constraint prevents it in the tested routes. The eight-archive sample is not a broad statistical claim.

Raw rows, model comparison output, and evaluator logs are in `results/dataset_training_eval_seed31416/`. The initial evaluation attempt exposed a missing local variable in shared feature construction; it was corrected before the successful held-out run. A subsequent attempt asked for a second simulation from one-record test archives; those invalid indices were recorded, then corrected by selecting multiple archives with one record each.

## Agent status and professor requirement

Dataset-based supervised training was completed for the existing 11-feature routing scorer. **Neither the Congestion DQN nor the Failure DQN was trained from this dataset.** DQN offline training remains unsupported because the required agent actions, transitions/rewards, and failure events are missing. The Orchestrator remains a rule-based baseline; it was not trained. Therefore the dataset-training requirement is **partially satisfied** if supervised training of the supported routing scorer is acceptable, but **not satisfied for training either DQN agent from the dataset**. No valid method exists to claim DQN dataset training from these fields without fabricating experience.

## Verification

The full test suite passed: 30 tests. Syntax compilation via `python -m compileall -q agents models simulator training scalability evaluation tests`, both new CLI help commands, and `git diff --check` completed successfully. Tests emitted only third-party PyTorch Geometric annotation deprecation warnings.
