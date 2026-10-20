# SDN-MARL verified project status

Audit date: 2026-10-20. This status records repository evidence and executed checks, not the intended final research outcome.

## Components

| Component | Verified status |
|---|---|
| Congestion routing checkpoint | Three small supervised candidate scoring checkpoints exist under `models/`. The evaluator defines a 4-layer MLP over exactly 11 candidate-link values. They are not the GNN DQN agent checkpoints. |
| Congestion DQN | `agents/congestion_agent.py` is an online/target GNN DQN with replay buffer, epsilon-greedy selection and optimizer updates. Training experience comes from simulator steps. This is an RL implementation, distinct from the supervised 11-feature checkpoints. |
| Failure DQN | `agents/failure_agent.py` has the same DQN structure and a 3-action output. Its presence does not prove effective failure learning; no compatible saved agent checkpoint was found in the inspected checkpoint locations. |
| Orchestrator | Added `RuleBasedOrchestrator` as an explicit safety baseline, wired into training and learned-agent evaluation. It prioritizes rerouting when the state marks any failed component and suppresses unsupported failure proposals otherwise. It is not learned. |
| Simulation | `SDNEnvironment` wraps `NetworkSimulator`, generated demands, path reroute/rate limit/isolation actions, graph-state observations, synthetic utilization/loss, and a shared scalar reward. The model is simplified and assigns defaults for missing capacity/delay; it is not a packet-level or queue simulator. |
| Conventional baseline | `evaluation/baseline.py` provides a shortest-path workload baseline. |
| CAIDA scalability | See `scalability/README.md`. Graph-only 1k/2k sampled-pair experiments ran; model feature inputs are unavailable from CAIDA. |

## Learning and evaluation limitations

The two DQN agents have separate policies and optimizers but receive the same state and scalar reward, and proposals are executed in sequence by the environment. This is a multi-agent training setup, but not evidence of effective coordination or improved performance. The orchestration layer now changes which actions reach the environment; replay entries are stored using those executed actions, so agent proposals may receive weak or confounded credit. A future learning study should make coordinator effects explicit and compare it against ablations.

The 11-feature supervised routing checkpoints are separate from the GNN agents. Their features include bandwidth, weight, utilization, loss, packet size, queue/port occupancy and destination distance. CAIDA contains none of the measured load/queue fields. The topology-only experiments consequently do not run those checkpoints and make no congestion claims.

The simulator can model random/synthetic demands, failures, rerouting, rate limiting, isolation and aggregate utilization/loss outcomes. Its default capacity and delay values are assumptions, and queue occupancy is not modeled as a measured time series. Results from it must be labeled simulator-generated. They are not real CAIDA measurements.

## Tests and execution

The full test suite passed after installing PyTorch Geometric in the active Python environment. PyTorch and PyTorch Geometric are now declared in `requirements.txt`; platform-specific PyTorch installation should still follow the official package instructions when a default pip wheel is unavailable.

## DQN training and held-out simulator evaluation

Two isolated 30-episode runs were executed on Abilene (11 nodes, 14 links), with 8 generated heavy-traffic demands per episode and batch size 8. The Congestion Agent trained with no injected failures; the Failure Agent trained with one failed link chosen from the most-used active route in every episode. Both runs completed 240 environment steps and 233 optimizer updates for the trained agent. Runtime was about 20 seconds per run on CUDA. Training episode rewards and mean losses are in the per-episode CSV; configs and summary JSON are stored beside the weights.

The fresh-process evaluation loads both saved checkpoints and runs three fixed seeds (101, 202, 303), 12 demands per scenario, across heavy and single-hotspot traffic plus heavy traffic with a used-link failure. The learned congestion policy selected no-op for all 72 no-failure decisions; metrics matched shortest path/no-agent exactly, so there is no demonstrated congestion benefit. The Failure Agent selected reroute for all 36 decisions over the three failed-link evaluations and restored all demands in all three cases. Its mean restoration step was 11 of 12, with 39.3 unserved demand-steps on average; shortest-path recovery restored service before step 1. No-agent delivery was 0.472 on average in those failure runs; shortest-path recovery, Failure DQN and the rule orchestrator all delivered 1.0 with zero final recorded packet loss. This is a small single-topology result, not evidence of broad generalization.

During this work, reward normalization was corrected to use original offered demand after rate limiting, and simulator packet-loss averaging was corrected to count isolated/unrouted demands as full loss. The Failure Agent's reroute action now chooses a shortest healthy route and permits retaining an already-valid route. These changes address metric gaming and failure-action semantics; the test suite includes regression coverage.

## Acceptance status

- Reproducible 1,000/2,000 node graph-only experiments: completed.
- Existing GNN-DQN agent implementations: present and unit-tested; learning effectiveness not established.
- Rule-based orchestration: implemented, unit-tested, and exercised with both trained checkpoints in evaluation; it remains a deterministic rule.
- Learned orchestrator: not implemented.
- Failure recovery controlled experiment: completed for one topology, three fixed seeds; broader tests/ablations remain.
- Congestion improvement: not demonstrated; trained policy matched no-op/shortest path.
- Full-graph shortest-path route evaluation: not run; no all-pairs computation attempted.

## Priority DQN diagnosis and corrected rerun (2026-10-20)

Repository state at the start of the follow-up was already dirty (the earlier
evaluation, CAIDA, DQN, and status work was uncommitted). Base commit recorded
for this run: `25cbd915eefe9d9a341294d0a4535eabe58516e1`. Existing results and
checkpoints were left in place; the artifacts below use `priority_*` paths.

### Verified fixes

- The earlier 72 no-op congestion decisions came from evaluations with zero
  congestion. Training also used a single high-load regime, and all actions
  were selectable in every state. The DQN now masks reroute/rate-limit unless
  a measured simulator edge is at or above the simulator's 0.80 utilization
  threshold. The same mask is used for exploratory actions, greedy actions,
  and the next-state max in DQN targets. Under uncongested traffic the learned
  policy now selects maintain, which is the appropriate action.
- Failure rerouting formerly acted on only the current demand even though a
  failed link could break several installed flows. It now repairs all active
  demands in one control step using healthy shortest alternatives.
- The failure isolate action formerly deleted broken demands before the
  recovery loop could see them. Isolation now retains affected flow records
  (unusable paths carry no traffic), then reroutes the broken flows.
- Added explicit `--traffic-scale` to create labeled simulator-generated load.
  Reward normalization uses original offered demand, so reducing offered load
  cannot score as full delivery. This is a synthetic stress control; it does
  not represent CAIDA measurements.
- There are no learned-agent action masks in the stored replay entries; masks
  are derived from each state's edge features, so the next-state bootstrap is
  masked consistently without changing checkpoint tensor shapes.

### Smoke and final DQN training

Smoke commands each completed 3 episodes, 12 environment steps and 9 optimizer
updates. Their checkpoints are retained under `results/priority_smoke_*` and
were used only for the initial check. The corrected final runs were:

```powershell
python training/train.py --episodes 40 --max-steps 12 --num-demands 12 --batch-size 16 --seed 9151 --scenario heavy --traffic-scale 8 --failure-probability 0 --train-agent congestion --output-dir results/priority_final_congestion_seed9151
python training/train.py --episodes 40 --max-steps 12 --num-demands 12 --batch-size 16 --seed 9152 --scenario heavy --failure-probability 1 --train-agent failure --output-dir results/priority_final_failure_seed9152
```

| Agent | Episodes | Steps | Optimizer updates | Runtime | Mean episode reward | Training action counts `[0,1,2]` | Checkpoint |
|---|---:|---:|---:|---:|---:|---|---|
| Congestion | 40 | 480 | 465 | 75.08 s | -0.1783 | maintain 186, reroute 82, rate-limit 212 | `results/priority_final_congestion_seed9151/congestion_dqn_final.pt` |
| Failure | 40 | 480 | 465 | 79.24 s | 4.5905 | no-action 149, reroute 155, isolate-and-reroute 176 | `results/priority_final_failure_seed9152/failure_dqn_final.pt` |

`training_summary.json`, `multi_agent_history.csv`, and the captured console
logs contain configurations, per-episode rewards/losses, action counts, step
counts, update totals and runtimes. The fresh-process evaluator successfully
loaded both final checkpoints and ran inference. Its probe was a normal state,
where both agents correctly selected action 0.

### Five-seed controlled simulator evaluation

Command:

```powershell
python training/evaluate_dqn.py --congestion-checkpoint results/priority_final_congestion_seed9151/congestion_dqn_final.pt --failure-checkpoint results/priority_final_failure_seed9152/failure_dqn_final.pt --seeds 941 942 943 944 945 --demands 8 --traffic-scale 8 --output results/priority_final_eval_seed9151_9152/evaluation_with_runtime.json
```

Each cell below is a mean over five identical-seed runs per policy/scenario.
“Heavy congestion” and “combined” use 8x generated simulator traffic; normal
and failure-only use 1x. The simulator calculates throughput/loss/utilization
from assumed link capacities and simplified utilization formulas. These are
simulator outcomes, not packet traces or CAIDA link measurements.

| Scenario | Policy | Delivery | Throughput Mbps | Packet loss | Congested-link ratio | Delay ms | Hops | Recovery steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | No-agent / shortest / congestion DQN / failure DQN / orchestrated | 1.000 | 538.3 | 0.000 | 0.000 | 2.4 | 2.33 | n/a |
| Heavy congestion | No-agent / shortest | 1.000 | 21,107.5 | 0.139 | 0.086 | 32.7 | 2.25 | n/a |
| Heavy congestion | Congestion DQN / orchestrated | 1.000 | 20,779.6 | 0.101 | 0.071 | 25.2 | 2.25 | n/a |
| Link failure | No-agent | 0.625 | 1,966.6 | 0.375 | 0.000 | 2.1 | 1.94 | not recovered |
| Link failure | Shortest path | 1.000 | 3,107.5 | 0.000 | 0.000 | 3.1 | 2.77 | 0 (baseline acts immediately) |
| Link failure | Failure DQN / orchestrated | 1.000 | 3,107.5 | 0.000 | 0.000 | 3.1 | 2.77 | 1 control step |
| Combined | No-agent | 0.625 | 15,732.5 | 0.375 | 0.015 | 20.1 | 1.94 | not recovered |
| Combined | Shortest path | 1.000 | 15,601.2 | 0.373 | 0.231 | 101.7 | 2.77 | 0 (baseline acts immediately) |
| Combined | Failure DQN / orchestrated | 1.000 | 15,230.2 | 0.387 | 0.246 | 105.2 | 2.77 | 1 control step |

On stress without failures, the congestion policy reduced mean congestion ratio
from 0.086 to 0.071, loss from 0.139 to 0.101, and delay from 32.7 ms to 25.2
ms, with about 1.6% lower throughput. This is a modest five-seed simulator
result, not evidence of generalization. Normal traffic had no policy action
other than maintain, avoiding the earlier unnecessary demand reduction. Failure
DQN restored delivery in five of five failure-only and combined runs, but takes
one control step after failure injection; shortest-path is an immediate
reference and therefore retains a one-step latency advantage. Combined stress
metrics remain worse under Failure DQN than under the immediate shortest-path
reference. Failure detection latency in seconds is unavailable: the simulator
injects and exposes the failed link before the first decision. Per-run elapsed
time and step recovery are in `evaluation_with_runtime.json` and `.csv`.

The Orchestrator remains a deterministic rule baseline. It was not trained;
it delegates to the DQNs when its failure rule does not override their choices.

### CAIDA scale results

CAIDA input was reused from `tech-RL-caida/tech-RL-caida.mtx`; no download or
dataset rebuild was performed. The parser verified 190,914 nodes, 607,610
unique undirected edges, one connected component, zero self-loops and zero
duplicate edge entries. Seed 314 generated connected induced subgraphs. Run
JSON files and topology-selection mappings are under `results/priority_scale/`.
Graph-only sampled pairs all delivered with zero loops/dead ends for both
shortest-path reference and distance-decreasing forwarding:

| Size | Edges | Pairs | Mean hops | Runtime | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 5,931 | 500 | 3.034 | 2.88 s | 283 MiB |
| 2,000 | 14,366 | 500 | 3.072 | 3.26 s | 283 MiB |
| 5,000 | 40,105 | 300 | 3.470 | 3.81 s | 283 MiB |
| 10,000 | 85,055 | 200 | 3.695 | 4.87 s | 283 MiB |
| Full graph, 190,914 | 607,610 | 20 | 6.200 | 7.56 s | 286 MiB |

No all-pairs table was created. Learned-agent comparisons on CAIDA remain
unsupported: there is no CAIDA traffic, failure state, utilization, capacity,
queue, or per-link monitoring data. The graph-only results make no congestion,
throughput, delay, or packet-loss claims. Synthetic Abilene simulations are a
separate experiment group.

### Verification and current conclusion

The final test command `python -m pytest -q` passed 23 tests. Syntax/import
validation `python -m compileall -q agents simulator training scalability
evaluation tests` completed successfully. PyTorch Geometric emitted 196
third-party deprecation warnings; there were no test failures.

The congestion agent now demonstrates a small stress-scenario improvement and
correctly holds on normal traffic, but benefit is not consistent enough to call
it a broadly useful policy. The failure agent now restores all sampled flows
after an injected active-link failure, one control step behind the immediate
shortest-path reference. Full-graph learned-policy testing is still blocked by
missing topology state/traffic measurements, while the full graph-only sampled
route test completed successfully.

## Phase 2 reproducibility and generalization (2026-10-20)

See [`experiments/phase2/REPORT.md`](experiments/phase2/REPORT.md) for metric
definitions, reproducibility commands, per-policy tables, statistical analysis,
topology transfer, CAIDA reruns, and limitations. This phase did not replace
the trained checkpoints or earlier raw outputs. It extended the evaluator to
include two simultaneous active-link failures and an orchestrator-without-
overrides ablation, and records offered-demand shortfall so rate-limiting is
not counted as free delivery. Metrics unsupported by the simulator are saved
as `null`.

New raw evaluations are under `results/phase2_final/`,
`results/phase2_final_cogentco/`, and `results/phase2_final_iris/`; paired
bootstrap/sign-flip analysis is saved in each `statistics/` subdirectory.
Abilene used 20 matched seeds and 720 policy/scenario runs. Cogentco (197 nodes,
243 edges) and Iris (51 nodes, 64 edges) used 10 seeds each. CAIDA subgraphs at
1k/2k/5k/10k nodes were rerun under `results/phase2_multiseed/caida_reproduced/`,
and the full 190,914-node graph completed 100 sampled pairs in 27.90 seconds at
286.8 MiB peak RSS. All CAIDA pairs delivered without loops or dead ends.

The Abilene congestion result is modest and did not survive Holm correction;
the policy acted only by rate-limiting and did not reduce the congested-link
ratio. Failure DQN matched shortest-path delivery but needed one control step.
Cogentco showed no congestion-policy effect; Iris showed limited actions but
no consistent benefit. Retraining was not performed because investigation did
not identify a verified implementation defect to fix or a validated checkpoint
selection criterion. The orchestrator remains rule-based, and its override
benefit is unproven. The full test suite after this phase passed 25 tests;
syntax compilation and `git diff --check` passed.

